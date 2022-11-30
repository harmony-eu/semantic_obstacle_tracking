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



## Usage with obstacle tracking
See documentation in [https://github.com/harmony-eu/moving_object_tracking/tree/humble-devel](https://github.com/harmony-eu/moving_object_tracking/tree/humble-devel).

## Standalone usage
Standalone launch file is:
```bash
ros2 launch semantic_tracking tracking.launch.py
```

**Arguments:**
- cameras: list of all camera namespaces (list of strings)
- camera_links: list of all the camera TF links (list of strings)
- yolo_topic: yolo topic (input)
- obstacles_topic: topic of tracked obstacles (input)
- obstacles_topic_vis: topic of tracked obstacles visualization (input)
- semantic_obstacles_topic (output)
- semantic_obstacles_topic_vis (output)

**Other node params:**
- **image_floor_margin**: \
YOLO bounding boxes for object that cross the edge of the image typically don't reach the edge pixels. To account for this, "lift" the obstacle projections to image space along the vertical axis until they reach the image_floor_margin. Normalized value between 0 (top line) and 1 (bottom line), default is 0.9.
- **intersection_threshold**: \
An obstacle belongs to a bounding box if at least a fraction of its points are projected inside the box. Normalized value between 0 (no points) and 1 (all points), default is 0.5.
- **lidar_height**: \
If for some reason the z-coordinate of the obstacles is wrong, then you can set the correct value here and it will be used to overwrite the z in the obstacle (0.32 meters for the ABB robot - lidar height of the ridgeback is 0.23). To NOT overwrite the values, set lidar_height=-2000 (default is -2000 i.e. OFF).
- **image_plane_threshold**: \
An object cannot "belong" to a camera if its z-coordinate in camera-frame is less than image_plane_threshold (i.e. if image_plane_threshold = 0, excludes all obstacles behind the camera plane). Default is 0.3 meters.
- **yolo_pixel_confidence_margin**: \
Enlarge all YOLO boxes by this amount (in pixels) in each direction. Default is 20 pixels, increase if yolo boxes are low-quality.
- **debug_topics**: \
Display color-coded obstacle projections onto an image stream for debug purposes. List of topics (list of strings). Default is ['none'].
- **rectified_input**: \
True if the yolo bounding boxes are from rectified images, False otherwise. Default is False.


**Notes:**
- Needs /tf topic (robot needs to be localized, TF from object frame to camera frame is needed)