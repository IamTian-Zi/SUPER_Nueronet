#!/bin/bash

# ==========================================
# SUPER 自动对比测试脚本
# ==========================================

# 1. 设置环境
source /opt/ros/noetic/setup.bash
cd ~/super_ws
source devel/setup.bash
export LD_LIBRARY_PATH=~/libtorch/lib:$LD_LIBRARY_PATH

# 2. 清理残留进程
killall -9 roscore rosmaster rosout perfect_drone fsm_node rviz 2>/dev/null
sleep 2

# 3. 启动核心环境 (启动仿真、Rviz 和你的规划节点)
# 注意：确保这里的 yaml 文件是你正在用的
echo ">>> 正在启动仿真环境..."
roslaunch perfect_drone_sim dense.launch > /dev/null 2>&1 &
rosrun super_planner fsm_node /home/tianzi/super_ws/src/SUPER/super_planner/config/static_dense.yaml > /tmp/fsm_log.txt 2>&1 &
sleep 5 # 等待节点初始化完毕

# 4. 定义 10 道考题 (X Y Z)
# 假设起点是无人机当前位置(0,0,1.5)，以下坐标为相对偏移，距离控制在 3~8 米
GOALS=(
    "3.0  0.0  1.5"
    "-3.0 0.0  1.5"
    "0.0  3.0  1.5"
    "0.0 -3.0  1.5"
    "4.0  4.0  1.5"
    "-4.0 -4.0 1.5"
    "5.0 -2.0  1.5"
    "-2.0 5.0  1.5"
    "6.0  0.0  1.5"
    "0.0  6.0  1.5"
)

# 5. 开始考试
echo "=========================================="
echo "  开始执行 10 道考题..."
echo "=========================================="

for i in "${!GOALS[@]}"; do
    read -r X Y Z <<< "${GOALS[$i]}"
    
    echo ">>> 考题 $((i+1))/10: 目标点 ($X, $Y, $Z)"
    
    # ===== 新增：让无人机悬停复位 =====
    # 发送一个和当前位置相同的点，或者等待它停下来
    # 这里我们假设它执行完上一道题后会悬停，我们多等一会儿
    echo "等待无人机复位悬停..."
    sleep 10  # 上一题飞完可能需要时间，必须等它彻底停下
    
    # 发送目标点
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
    
    # 等待规划完成（抓取数据的时机）
    sleep 2 
    
    # 提取数据
    grep "\[BENCHMARK_RESULT\]" /tmp/fsm_log.txt | tail -n 1 >> /tmp/benchmark_results.txt
    sed -i '/\[BENCHMARK_RESULT\]/d' /tmp/fsm_log.txt
done
# 6. 结束清理
echo "=========================================="
echo "  测试结束，关闭环境"
killall -9 roscore rosmaster rosout perfect_drone fsm_node rviz 2>/dev/null

echo "测试数据已保存至 /tmp/benchmark_results.txt"
cat /tmp/benchmark_results.txt