import cv2
import numpy as np

from cv_bridge import CvBridge

from sensor_msgs.msg import CameraInfo, CompressedImage


class Camera:
    def __init__(self, name, link, id=0) -> None:
        self.name = name
        self.link = link
        self.id = id

        self.frame_id = None
        self.k = None
        self.d = None
        self.height = None
        self.width = None

        self.current_yolo = []

        self.debug_detections = []

        self.R = None
        self.t = None


def get_info_callback(camera: Camera):
    def info_callback(info: CameraInfo):
        if camera.k is None: 
            camera.k, camera.d = np.array(info.k), np.array(info.d)
            camera.k.resize((3,3))
            camera.frame_id = str(info.header.frame_id)
            camera.height = info.height
            camera.width = info.width
    return info_callback


def get_camera_from_topic(cameras, topic: str):
    for camera in cameras:
        if camera.name in topic: return camera
    return None


def get_image_callback(camera, compressed):

    def image_callback(img):
        mBridge = CvBridge().compressed_imgmsg_to_cv2 if compressed else CvBridge().imgmsg_to_cv2
        image = mBridge(img)
        # bridge = CvBridge()
        # image = bridge.compressed_imgmsg_to_cv2(img)
        
        for det in camera.debug_detections:
            if det.semclass is None: color = np.array([255, 255, 255])
            elif det.semclass == 'people': color = np.array([255, 0, 0])
            else: color = np.array([0, 0, 255])
            npix = det.pixel_coords.shape[0]
            det.pixel_coords.resize(npix, 2)
            r = 2
            for idx in range(npix):
                x, y = det.pixel_coords[idx, 0], det.pixel_coords[idx, 1]
                image[y-r:y+r, x-r:x+r, :] = color
        for det in camera.current_yolo:
            x1, y1 = det.xmin, det.ymin
            x2, y2 = det.xmax, det.ymax
            cv2.rectangle(image,(x1,y1), (x2,y2), (0, 0, 255), 2)
            
        cv2.namedWindow(camera.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(camera.name, 720, 480)
        cv2.imshow(camera.name, image)
        cv2.waitKey(1)
    return image_callback