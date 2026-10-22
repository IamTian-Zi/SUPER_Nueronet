#pragma once
#ifdef logit
#undef logit
#endif

#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include "rog_map_ros/rog_map_ros1.hpp"

class SlopePruner {
public:
    SlopePruner(const std::string& model_path, float prune_threshold = 2.0f);

    // 评估某个节点是否应该被保留
    bool shouldKeep(const rog_map::Vec3i& current_id,
        const rog_map::Vec3i& goal_id,
        rog_map::ROGMapROS::Ptr map_ptr);

private:
    torch::jit::script::Module module_;
    torch::Device device_{ torch::kCPU };
    float threshold_;

    // 提取局部栅格 (3, 8, 8, 4)
    torch::Tensor extractLocalGrid(const rog_map::Vec3i& center_id, rog_map::ROGMapROS::Ptr map_ptr);
};