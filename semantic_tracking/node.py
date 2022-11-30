import cv2
import numpy as np
import rclpy
import typing

from time import time as get_time

from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import MarkerArray, Marker

from yolo_eth_msgs.msg import BoundingBoxes, BoundingBox
from obstacle_detector.msg import Obstacles, CircleObstacle, SegmentObstacle


from semantic_tracking.camera import *
from semantic_tracking.utils import *


class SemanticObstacleMatcher(Node):

    def __init__(self):
        super().__init__('semantic_tracking')
        default_debug_channels = ['list', 'of', 'topics']
        self.declare_parameter('cameras', ['list', 'of', 'topics'])
        self.declare_parameter('camera_links', ['list', 'of', 'tf', 'links'])
        self.declare_parameter('image_floor_margin', 0.9)
        self.declare_parameter('intersection_threshold', 0.5)
        self.declare_parameter('lidar_height', -2000.0)
        self.declare_parameter('image_plane_threshold', 0.3)
        self.declare_parameter('yolo_pixel_confidence_margin', 20)
        self.declare_parameter('debug_topics', default_debug_channels)
        self.declare_parameter('rectified_input', False)


        camera_names = self.get_parameter('cameras').value
        camera_links = self.get_parameter('camera_links').value
        self.image_floor_margin = self.get_parameter('image_floor_margin').value

        self.cameras = list()
        for name, link in zip(camera_names, camera_links):
            self.cameras.append(Camera(name, link))

        self.intersection_threshold = self.get_parameter('intersection_threshold').value
        self.lidar_height = self.get_parameter('lidar_height').value
        self.image_plane_threshold = self.get_parameter('image_plane_threshold').value
        self.yolo_pixel_confidence_margin = self.get_parameter('yolo_pixel_confidence_margin').value
        self.debug_topics = self.get_parameter('debug_topics').value
        if self.debug_topics == default_debug_channels: self.debug_topics = []
        self.rectified_input = self.get_parameter('rectified_input').value

        self.info_subs = []
        for camera in self.cameras:
            sub = self.create_subscription(
                CameraInfo, camera.name + '/camera_info', get_info_callback(camera), 10)
            self.info_subs.append(sub)

        self.debug_sub = []
        for mtopic in self.debug_topics:
            dcamera = get_camera_from_topic(self.cameras, mtopic)
            if dcamera is None: continue
            self.get_logger().info("Displaying debug camera stream for topic " + mtopic + " of camera " + dcamera.name)
            compressed = bool('compressed' in mtopic or 'Compressed' in mtopic)
            mType = CompressedImage if compressed else Image
            self.debug_sub.append(self.create_subscription(
                mType, mtopic, get_image_callback(dcamera, compressed), 10
            ))
            
        # if len(self.debug_topics) > 0:
        #     self.get_logger().info("Displaying debug camera stream")
        #     debug_camera = self.cameras[self.debug_topics]
        #     self.debug_sub = self.create_subscription(
        #     CompressedImage, debug_camera.name + '/image_color/compressed', get_image_callback(debug_camera), 10)

        self.yolo_topic = "/yolov5"
        self.yolo_sub = self.create_subscription(BoundingBoxes, self.yolo_topic, self.yolo_callback, 10)

        self.obstacle_topic = "/tracked_obstacles"
        self.obstacle_sub = self.create_subscription(Obstacles, self.obstacle_topic, self.obstacle_callback, 10)
        self.obstacle_frame = None

        self.vis_obstacle_topic = "/tracked_obstacles_visualization"
        self.vis_obstacle_sub = self.create_subscription(MarkerArray, self.vis_obstacle_topic, self.vis_obstacle_callback, 10)

        self.semantic_obstacle_topic = "/tracked_semantic_obstacles"
        self.obstacle_pub = self.create_publisher(Obstacles, self.semantic_obstacle_topic, 10)

        self.semantic_obstacle_topic_vis = "/tracked_semantic_obstacles_visualization"
        self.vis_obstacle_pub = self.create_publisher(MarkerArray, self.semantic_obstacle_topic_vis, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.call_time_sum = 0
        self.call_time_n = 0

        self.assigned_obstacles = {}

    def yolo_callback(self, yolo_msg: BoundingBoxes):
        camera = get_camera(self.cameras, yolo_msg.image_header.frame_id)
        if camera is None: return
        camera.current_yolo = yolo_msg.bounding_boxes

    def get_tf(self, target, source):
        try:
            transform = self.tf_buffer.lookup_transform(
                        target_frame=target,
                        source_frame=source,
                        time=rclpy.time.Time())
        except TransformException as ex:
            # self.get_logger().info(f'Could not get transform from {source} to {target}: {ex}')
            return None

        return transform.transform
        

    def match_class(self, points: np.ndarray):
        matching_yolo = None
        for camera in self.cameras:
            # if len(camera.current_yolo) < 1: continue
            if camera.R is None: 
                self.get_logger().info(f'TF not yet available for camera ' + str(camera.name), throttle_duration_sec=3)
                continue
            # lidar is ~ 40 cm off the ground?
            if self.lidar_height > -1000:
                points[:, 2] = self.lidar_height
            # obst_frame -> camera_frame TF
            points_c = do_transform_points(points, camera.R, camera.t)

            # get rid of points behind the image plane
            points_c = points_c[points_c[:, 2] > self.image_plane_threshold]
            olen, nlen = points.shape[0], points_c.shape[0]
            # skip obstacle if less then 30% of points are ahead of the camera
            if nlen < olen * self.intersection_threshold: continue

            # project points from image plane to pixel coordinates
            dist = np.zeros_like(camera.d) if self.rectified_input else camera.d
            res1, _ = cv2.projectPoints(points_c, np.zeros((3,1)), np.zeros((3,1)), camera.k, dist)
            res1 = res1.astype(int).reshape((nlen, 2))
            # if the point is too low, bring it up to minimum eye level
            y_tresh = int(self.image_floor_margin * camera.height)
            res1[res1[:, 1] > y_tresh, 1] = y_tresh

            debug_det = Detection()
            debug_det.pixel_coords = res1
            max_score = 0
            c = self.yolo_pixel_confidence_margin
            for yolo in camera.current_yolo:
                x1, x2, y1, y2 = yolo.xmin-c, yolo.xmax+c, yolo.ymin-c, yolo.ymax+c
                # I would expect the lidar obstacle to be at the bottom of the bounding box (e.g. in the bottom 30%)
                y1 = int(y2 - 0.3*(y2 - y1))
                # obj_intersection, width_occupancy = intersection(res1, x1, x2, y1, y2)
                # score_1 is intersection * percentage of box x-space occupied by obstacle
                #     if we have two boxes with the same center and obj intersection is equal,
                #     favors the smaller box (which is usually the one in front)
                # score_1 = obj_intersection * width_occupancy
                obj_intersection, ioux = intersection_2(res1, x1, x2, y1, y2)
                score = ioux
                if obj_intersection > self.intersection_threshold and max_score < score:
                    # min some intersection (easy to tune)
                    max_score = score
                    matching_yolo = yolo

                    debug_det.semclass = yolo.object_class 
            camera.debug_detections.append(debug_det)
        return matching_yolo

    def assign_class(self, obstacle_ref: typing.Union[CircleObstacle, SegmentObstacle], yolo_detection):
        # aliases for readability
        uid = obstacle_ref.uid

        # if assignment in memory is better than what we have for this frame -> override
        # if uid in self.assigned_obstacles:
        #     if yolo_detection is None or yolo_detection.confidence < self.assigned_obstacles[uid].confidence:
        #         yolo_detection = self.assigned_obstacles[uid]

        if yolo_detection is not None:
            self.assigned_obstacles[uid] = yolo_detection
            obstacle_ref.semclass = str(yolo_detection.object_class)
            obstacle_ref.confidence = yolo_detection.probability

    def vis_obstacle_callback(self, markers: MarkerArray):
        for marker in markers.markers:
            if marker.id in self.assigned_obstacles and marker.type == Marker.TEXT_VIEW_FACING:
                marker.text = marker.text + " " + str(self.assigned_obstacles[marker.id].object_class)
        # republish annotated markers to new topic
        self.vis_obstacle_pub.publish(markers)

    def obstacle_callback(self, obstacle_msg: Obstacles):
        start = get_time()
        self.obstacle_frame = obstacle_msg.header.frame_id
        for camera in self.cameras: 
            tf_msg = self.get_tf(target=camera.link, source=self.obstacle_frame)
            if tf_msg is not None: camera.R, camera.t = decode_tf(tf_msg)
            camera.debug_detections = []
            
        for circle in obstacle_msg.circles:
            # get reference to obstacles
            cc = circle.center
            points = sample_circle(np.array([cc.x, cc.y, cc.z]), circle.radius, n=20)
            # match a class in the current frame
            yolo_detection = self.match_class(points)
            # assign class (in place)
            self.assign_class(obstacle_ref=circle, yolo_detection=yolo_detection)

        for segment in obstacle_msg.segments:
            f, l = segment.last_point, segment.first_point
            points = sample_line(np.array([f.x, f.y, f.z]), np.array([l.x, l.y, l.z]), 20)
            yolo_detection = self.match_class(points)
            self.assign_class(obstacle_ref=segment, yolo_detection=yolo_detection)

        # republish obstacle array, now with semantic class and confidence
        self.obstacle_pub.publish(obstacle_msg)
        self.call_time_sum += (get_time()-start) * 1000
        self.call_time_n += 1


def main(args=None):
    rclpy.init(args=args)

    semantic_tracker = SemanticObstacleMatcher()
    while rclpy.ok():
        rclpy.spin_once(semantic_tracker)
        avg_time = semantic_tracker.call_time_sum/(semantic_tracker.call_time_n + 1)
        msg = "avg time of obstacle callback (ms): {:.2f}".format(avg_time)
        semantic_tracker.get_logger().info(msg, throttle_duration_sec=5)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    semantic_tracker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()