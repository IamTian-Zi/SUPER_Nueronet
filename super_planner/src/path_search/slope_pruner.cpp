#include "path_search/slope_pruner.h"
#include <iostream>

SlopePruner::SlopePruner(const std::string& model_path, float prune_threshold)
    : threshold_(prune_threshold) {
    try {
        module_ = torch::jit::load(model_path, device_);
        module_.eval();
        std::cout << "[SLOPE] Model loaded successfully from " << model_path << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "[SLOPE] Error loading model: " << e.what() << std::endl;
    }
}

torch::Tensor SlopePruner::extractLocalGrid(const rog_map::Vec3i& center_id, rog_map::ROGMapROS::Ptr map_ptr) {
    int L = 8, W = 8, H = 4;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor grid_tensor = torch::zeros({ 1, 3, L, W, H }, options);
    auto accessor = grid_tensor.accessor<float, 5>();

    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int k = 0; k < H; ++k) {
                rog_map::Vec3i id = center_id + rog_map::Vec3i(i - L / 2, j - W / 2, k - H / 2);

                // 改用公开接口获取物理坐标
                rog_map::Vec3f pos;
                map_ptr->infMapGlobalIndexToPos(id, pos);

                if (!map_ptr->insideLocalMap(pos)) continue;

                // 改用 getInfGridType 判断是否占据
                rog_map::GridType type = map_ptr->getInfGridType(pos);
                accessor[0][0][i][j][k] = (type == rog_map::OCCUPIED) ? 1.0f : 0.0f;

                // SUPER 没有直接的 ESDF 接口，这里填 0 或使用高度
                accessor[0][1][i][j][k] = 0.0f;
                accessor[0][2][i][j][k] = pos.z();
            }
        }
    }
    return grid_tensor;
}

bool SlopePruner::shouldKeep(const rog_map::Vec3i& current_id,
    const rog_map::Vec3i& goal_id,
    rog_map::ROGMapROS::Ptr map_ptr) {
    torch::Tensor grid = extractLocalGrid(current_id, map_ptr);

    rog_map::Vec3f current_pos, goal_pos;
    // 改用公开接口获取物理坐标
    map_ptr->infMapGlobalIndexToPos(current_id, current_pos);
    map_ptr->infMapGlobalIndexToPos(goal_id, goal_pos);

    auto goal_tensor = torch::tensor({
        goal_pos.x() - current_pos.x(),
        goal_pos.y() - current_pos.y(),
        goal_pos.z() - current_pos.z()
        }, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0);

    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(grid.to(device_));
    inputs.push_back(goal_tensor.to(device_));

    float slope_dist = module_.forward(inputs).toTensor().item<float>();
    return slope_dist < threshold_;
}
