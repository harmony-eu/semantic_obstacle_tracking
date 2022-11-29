import cv2
import numpy as np

from cv_bridge import CvBridge

from sensor_msgs.msg import CameraInfo, CompressedImage


class Camera:
    def __init__(self, name, link, id) -> None:
        self.name = name
        self.link = link
        self.id = id

        self.k = None
        self.d = None

        self.current_yolo = None

        self.debug_detections = []

        self.R = None
        self.t = None


def get_info_callback(camera):
    def info_callback(info: CameraInfo):
        camera.k, camera.d = np.array(info.k), np.array(info.d)
        camera.k.resize((3,3))
    return info_callback


def get_image_callback(camera):
    def image_callback(img: CompressedImage):
        bridge = CvBridge()
        image = bridge.compressed_imgmsg_to_cv2(img)
        
        for det in camera.debug_detections:
            if det.semclass is None: color = np.array([255, 255, 255])
            elif det.semclass == 12: color = np.array([255, 0, 0])
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
            
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", 720, 480)
        cv2.imshow("output", image)
        cv2.waitKey(1)
    return image_callback