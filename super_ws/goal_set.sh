#!/bin/bash

TOPIC_NAME="/move_base_simple/goal"
TARGET_X=5.0
TARGET_Y=0.0
TARGET_Z=1.5

echo "=========================================="
echo "  正在向 $TOPIC_NAME 发送目标点..."
echo "  目标坐标: X=$TARGET_X, Y=$TARGET_Y, Z=$TARGET_Z"
echo "=========================================="

rostopic pub -1 $TOPIC_NAME geometry_msgs/PoseStamped "
header:
  frame_id: 'world'
pose:
  position:
    x: $TARGET_X
    y: $TARGET_Y
    z: $TARGET_Z
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0"
