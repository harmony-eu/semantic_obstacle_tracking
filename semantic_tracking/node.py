import cv2
import numpy as np
import rclpy
import typing

from time import time as get_time

from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import MarkerArray, Marker

from nmcl_msgs.msg import YoloCombinedArray, YoloArray, Yolo
from obstacle_detector.msg import Obstacles, CircleObstacle, SegmentObstacle


from semantic_tracking.camera import *
from semantic_tracking.utils import *


class SemanticObstacleMatcher(Node):

    def __init__(self):
        super().__init__('semantic_tracking')
        # camera_0 = Camera('/kinect_master/rgb', 'kinect_master_rgb_camera_link_rotated', 1)
        # camera_0 = Camera('/kinect_master/rgb', 'kinect_master_camera_visor', 1)
        camera_1 = Camera('/multicam/right', 'flir_right_link', 1)
        camera_2 = Camera('/multicam/back', 'flir_back_link', 2)
        camera_3 = Camera('/multicam/left', 'flir_left_link', 3)
        self.cameras = {1:camera_1, 2:camera_2, 3:camera_3}

        self.pixel_y_threshold = 1000
        self.intersection_threshold = 0.5
        self.lidar_height = 0.4
        self.image_plane_threshold = 0.3
        self.yolo_x_scaling_factor = 1440/640
        self.yolo_y_scaling_factor = 1080/480
        self.yolo_pixel_confidence_margin = 20
        self.debug_vis_projection = True

        self.class_ids_to_string = ['sink', 'door', 'oven', 'whiteboard', 'table', 'cardboard', 'plant', 'drawers', 'sofa', 'storage', 'chair', 'extinguisher', 'people', 'desk']

        self.info_subs = []
        for camera in self.cameras.values():
            sub = self.create_subscription(
                CameraInfo, camera.name + '/camera_info', get_info_callback(camera), 10)
            self.info_subs.append(sub)
            
        if self.debug_vis_projection:
            self.debug_sub = self.create_subscription(
            CompressedImage, camera_2.name + '/image_color/compressed', get_image_callback(camera_2), 10)

        self.yolo_topic = "/yolov5"
        self.yolo_sub = self.create_subscription(YoloCombinedArray, self.yolo_topic, self.yolo_callback, 10)

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
        self.call_time = []

        self.assigned_obstacles = {}

    def yolo_callback(self, yolo_msg: YoloCombinedArray):
        for view in yolo_msg.views:
            if view.cam_id not in self.cameras: continue
            camera = self.cameras[view.cam_id]
            for det in view.detections:
                # yolo bounding boxes are not scaled properly
                det.xmin = int(det.xmin * self.yolo_x_scaling_factor)
                det.xmax = int(det.xmax * self.yolo_x_scaling_factor)
                det.ymin = int(det.ymin * self.yolo_y_scaling_factor)
                det.ymax = int(det.ymax * self.yolo_y_scaling_factor)

            camera.current_yolo = view.detections

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
        for id, camera in self.cameras.items():
            if camera.current_yolo is None: continue
            if camera.R is None: self.get_logger().info(f'TF not yet available for camera ' + str(camera.name))
            # lidar is ~ 40 cm off the ground?
            points[:, 2] = self.lidar_height
            # obst_frame -> camera_frame TF
            points_c = do_transform_points(points, camera.R, camera.t)

            # get rid of points behind the image plane
            points_c = points_c[points_c[:, 2] > self.image_plane_threshold]
            olen, nlen = points.shape[0], points_c.shape[0]
            # skip obstacle if less then 30% of points are ahead of the camera
            if nlen < olen * self.intersection_threshold: continue

            # project points from image plane to pixel coordinates
            res1, _ = cv2.projectPoints(points_c, np.zeros((3,1)), np.zeros((3,1)), camera.k, camera.d)
            res1 = res1.astype(int).reshape((nlen, 2))
            # if the point is too low, bring it up to minimum eye level
            res1[res1[:, 1] > self.pixel_y_threshold, 1] = self.pixel_y_threshold

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

                    debug_det.semclass = yolo.semclass 
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
            obstacle_ref.semclass = yolo_detection.semclass
            obstacle_ref.confidence = yolo_detection.confidence

    def vis_obstacle_callback(self, markers: MarkerArray):
        for marker in markers.markers:
            if marker.id in self.assigned_obstacles and marker.type == Marker.TEXT_VIEW_FACING:
                semclass_id = self.assigned_obstacles[marker.id].semclass
                marker.text = marker.text + " " + str(self.class_ids_to_string[semclass_id])
        # republish annotated markers to new topic
        self.vis_obstacle_pub.publish(markers)

    def obstacle_callback(self, obstacle_msg: Obstacles):
        start = get_time()
        self.obstacle_frame = obstacle_msg.header.frame_id
        for camera in self.cameras.values(): 
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
        self.call_time.append((get_time()-start) * 1000)


def main(args=None):
    rclpy.init(args=args)

    semantic_tracker = SemanticObstacleMatcher()
    i = 0
    while rclpy.ok():
        rclpy.spin_once(semantic_tracker)
        if i%100 == 0 and len(semantic_tracker.call_time) > 0: 
            print("avg time")
            print(sum(semantic_tracker.call_time) / len(semantic_tracker.call_time))
        i += 1

    """
    TODO: right now I just do single-shot assignment
    but we have tracked obstacles
    does the semantic class persist in the tracked obstacle?
        i.e. is the semantic class field copied over when the tracked obstacle is identified?
        if yes, then the sem class should persist in the lidar obstacles even when the object is
        occluded in the camera view
    TODO: where should the semantic assignment happen?
    option 1 -> in the raw obstacles -> use class for obstacle tracking
    option 2 -> in the tracked obstacles -> just republish, keep a list of obstacle_ids-to-semantic class
        in this node and update it online if we get something better (more confident or more intersection)
        Also solves TODO_1
    TODO: test with better data??
    """

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    print(sum(semantic_tracker.call_time) / len(semantic_tracker.call_time))
    semantic_tracker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()