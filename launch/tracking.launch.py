import launch

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # tracking_marker_frame = DeclareLaunchArgument('tracking_marker_frame', default_value="tag_2")
    cameras = DeclareLaunchArgument('cameras', default_value="tag_2")
    obstacles_in = DeclareLaunchArgument('obstacles_in', default_value="tag_2")
    obstacles_out = DeclareLaunchArgument('obstacles_out', default_value="tag_2")
    yolo_in = DeclareLaunchArgument('yolo_in', default_value="tag_2")

    node = Node(
            package='semantic_tracking',
            executable='semantic_tracking',
            name='semantic_tracking',
            output='screen',
            parameters=[
                {'tracking_base_frame': LaunchConfiguration('tracking_base_frame')},
            ]
            remap=[
                ('obstacles_in': LaunchConfiguration('obstacles_in')),
                ('obstacles_out': LaunchConfiguration('obstacles_out')),
                ('yolo_in': LaunchConfiguration('yolo_in'))
            ]
    )    

    ll = list()
    ll.append(cameras)
    ll.append(obstacles_in)
    ll.append(obstacles_out)
    ll.append(yolo_in)
    ll.append(calibration_type)
    ll.append(calibration_node)

    return LaunchDescription(ll)