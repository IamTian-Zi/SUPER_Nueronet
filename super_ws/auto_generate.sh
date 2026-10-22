#!/bin/bash
# ==========================================
# 自动化轨迹数据采集脚本
# ==========================================

# 1. 环境配置
source /opt/ros/noetic/setup.bash
cd ~/super_ws
source devel/setup.bash
export LD_LIBRARY_PATH=~/libtorch/lib:$LD_LIBRARY_PATH

# 2. 创建数据存储目录
mkdir -p /home/tianzi/super_ws/dataset/raw
TOTAL_RUNS=50  # 设定你想生成多少个独立场景的数据

for i in $(seq 1 $TOTAL_RUNS); do
    echo "=========================================="
    echo ">>> 正在生成第 $i / $TOTAL_RUNS 组数据..."
    
    # 清理上一轮残留
    killall -9 roscore rosmaster rosout perfect_drone fsm_node rviz 2>/dev/null
    rm -f /tmp/current_traj.csv
    sleep 2

    # 启动仿真与节点
    roslaunch perfect_drone_sim dense.launch > /dev/null 2>&1 &
    sleep 3
    rosrun super_planner fsm_node /home/tianzi/super_ws/src/SUPER/super_planner/config/static_dense.yaml > /tmp/fsm_log.txt 2>&1 &
    sleep 5 # 等待 SLOPE 模型加载

    # 随机生成一个目标点 (x: 2~6米, y: -4~4米, z: 1.5米)
    X=$(awk -v min=2 -v max=6 'BEGIN{srand(); print min+rand()*(max-min)}')
    Y=$(awk -v min=-4 -v max=4 'BEGIN{srand(); print min+rand()*(max-min)}')
    Z=1.5

    echo ">>> 发送目标点: X=$X, Y=$Y, Z=$Z"
    timeout 1 rostopic pub -1 /goal geometry_msgs/PoseStamped "
    header:
      frame_id: 'world'
    pose:
      position:
        x: $X
        y: $Y
        z: $Z
      orientation:
        w: 1.0" > /dev/null 2>&1

    # 等待规划完成写入数据
    sleep 15

    # 保存数据（带上时间戳或序号防止覆盖）
    if [ -f /tmp/current_traj.csv ]; then
        mv /tmp/current_traj.csv "/home/tianzi/super_ws/dataset/raw/scene_$i.csv"
        echo ">>> 成功保存 scene_$i.csv"
    else
        echo ">>> [警告] 第 $i 次规划失败，未生成数据"
    fi
done

# 结束清理
killall -9 roscore rosmaster rosout perfect_drone fsm_node rviz 2>/dev/null
echo "=========================================="
echo ">>> 全部采集完成！数据位于 dataset/raw/"