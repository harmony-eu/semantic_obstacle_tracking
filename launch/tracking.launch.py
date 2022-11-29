from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument as LA
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration as LC


def generate_launch_description():
    largs = []
    cameras = [
        '/multicam/back',
        '/multicam/right',
        '/multicam/left'
    ]
    camera_links = [
        'flir_back_link',
        'flir_right_link',
        'flir_left_link'
    ]
    # cameras and camera_links should be in the same order
    largs.append(LA('cameras',default_value=str(cameras)))
    largs.append(LA('camera_links',default_value=str(camera_links)))

    largs.append(LA('yolo_topic',default_value="yolo/yolo_eth_ros/detections"))
    largs.append(LA('obstacles_topic',default_value="/tracked_obstacles"))
    largs.append(LA('obstacles_topic_vis',default_value="/tracked_obstacles_visualization"))
    largs.append(LA('semantic_obstacles_topic',default_value="/tracked_semantic_obstacles"))
    largs.append(LA('semantic_obstacles_topic_vis',default_value="/tracked_semantic_obstacles_visualization"))

    node = Node(
        package='semantic_tracking',
        executable='semantic_tracking',
        name='semantic_tracking',
        output='screen',
        parameters=[
            {'cameras': LC('cameras')},
            {'camera_links': LC('camera_links')},
        ],
        remappings=[
            ('yolov5', LC('yolo_topic')),
            ("tracked_obstacles", LC('obstacles_topic')),
            ("tracked_obstacles_visualization", LC('obstacles_topic_vis')),
            ("tracked_semantic_obstacles", LC('semantic_obstacles_topic')),
            ("tracked_semantic_obstacles_visualization", LC('semantic_obstacles_topic_vis')),
        ]
    )
    largs.append(node)

    return LaunchDescription(largs)
