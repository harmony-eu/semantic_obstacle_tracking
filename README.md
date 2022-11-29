# Semantic Tracking
**Description**: assign tracked Lidar obstacles to a semantic class, by integrating YOLO from cameras.

**Container**: harmony `base` Humble container

## Working Principle
Project each Lidar obstacle into camera space. For each obstacle, calculate:
- the normalized intersection between the obstacle projection and each YOLO bounding box
- the IOU between the obstacle projection and each YOLO bounding box (approximated to only horizontal direction)

An obstacle is assigned to a bounding box if their normalized intersection is higher than the `intersection_threshold`. If an obstacle could be assigned to multiple bounding boxes, it will be assigned to the bounding box that maximises the IOU.

## Assumptions


<!-- ## Required components
`/tf` and `/tf_static` topics:
- localization: provided by Bonn
- TF tree: provided by robot description
`/multicam/left...right...back/image_rect` and `multicam/left...right...back/camera_info`
- **rectified** camera streams: (mapping Azure Kinect and 3 x FLIR Firefly cameras)
`/scan_merged_eth`
- merged lidar scan (as **PointCloud2**): provided by scan_merger_eth node
`/map` and `/map_inflated`:
- map server (actual map and inflated map for filtering) -->



## Usage
- start YOLO node
- start obstacle tracking (see [moving_object_tracking](https://github.com/harmony-eu/moving_object_tracking))
- start node:
    ```bash
    ros2 launch bla bla
    ```


**Parameters:**


**Topics:**
- /yolov5 topic (message_type `nmcl_msgs.msg.YoloCombinedArray`)
- /tf (robot needs to be localized, TF from map to camera is needed)
- /tracked_obstacles (message_type `obstacle_detector.msg.Obstacles`)