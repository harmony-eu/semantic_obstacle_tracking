import cv2
import numpy as np

from scipy.spatial.transform import Rotation


def decode_tf(transform_msg):
    """
    convert a ros transform_msg to a Rotation matrix (3x3) 
    translation vector object (t.x, t.y, tz)
    """
    q = transform_msg.rotation
    R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    t = transform_msg.translation
    return R, t


def get_camera(cameras, frame_id):
    for camera in cameras:
        if camera.frame_id == frame_id: return camera
    return None


def do_transform_points(point_np, r, t):
    """
    point_np: np.ndarray: pointcloud of shape (N, 3)
    r, t: rotation matrix and translation object (from get_transform)
    return: transform pointcloud (N, 3)
    """
    point_np = np.transpose(point_np)[:3, :]
    point_tf = np.matmul(r, point_np) + np.array([[float(t.x)], [float(t.y)], [float(t.z)]]) 
    return np.transpose(point_tf)


def intersection(points: np.ndarray, x1: float, x2: float, y1: float, y2: float) -> float:
    """
    points: 2d points array of shape Nx2
    x1, x2, y1, y2: rectangle corners
    return: percentage of points inside the rectangle
    """
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    inside_x = (x1 < points[:, 0]) * (points[:, 0] < x2)
    inside_y = (y1 < points[:, 1]) * (points[:, 1] < y2)
    inside = inside_x * inside_y
    intersection_percentage = inside.sum() / points.shape[0]
    if intersection_percentage > 0:
        box_width = x2 - x1
        intersection_normalized_width = (np.max(points[inside, 0]) - np.min(points[inside, 0])) / box_width
    else:
        intersection_normalized_width = 0
    
    return intersection_percentage, intersection_normalized_width


def intersection_2(points: np.ndarray, x1: float, x2: float, y1: float, y2: float) -> float:
    """
    points: 2d points array of shape Nx2
    x1, x2, y1, y2: rectangle corners
    return: percentage of points inside the rectangle, intersection_over_union_x
    """
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    inside_x = (x1 < points[:, 0]) * (points[:, 0] < x2)
    inside_y = (y1 < points[:, 1]) * (points[:, 1] < y2)
    inside = inside_x * inside_y
    normalized_intersection = inside.sum() / points.shape[0]
    box_width = x2 - x1

    # approximate intersection over union by only considering x coordinate
    if normalized_intersection == 0:
        ioux = 0
    elif normalized_intersection == 1:
        obstacle_intersection_width = np.max(points[inside, 0]) - np.min(points[inside, 0])
        ioux = obstacle_intersection_width / box_width
    else:
        obstacle_intersection_width = np.max(points[inside, 0]) - np.min(points[inside, 0])
        obstacle_width = np.max(points[:, 0]) - np.min(points[:, 0])
        ioux = obstacle_intersection_width / (box_width + obstacle_width - obstacle_intersection_width)

    return normalized_intersection, ioux


def sample_circle(center: np.array, radius: float, n) -> np.ndarray:
    theta = np.linspace(0, 2*3.141592, n)
    res = np.zeros((n, 3))
    radiuses = np.random.random(n) * radius
    res[:, 0] = np.cos(theta) * radiuses + center[0]
    res[:, 1] = np.sin(theta) * radiuses + center[1]
    return res


def sample_line(p1: np.array, p2: np.array, n) -> np.ndarray:
    res = np.zeros((n, 3))
    res[:, 0] = np.linspace(p1[0], p2[0], n)
    res[:, 1] = np.linspace(p1[1], p2[1], n)
    res[:, 2] = np.linspace(p1[2], p2[2], n)
    return res


class Detection:
    def __init__(self) -> None:
        self.semclass = None
        self.position = None
        self.pixel_coords = None
        self.intersection = 0
    
    def is_empty(self):
        return not self.semclass is None
