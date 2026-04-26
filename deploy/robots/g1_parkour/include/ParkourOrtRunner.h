#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "isaaclab/algorithms/algorithms.h"

namespace isaaclab
{

class ParkourOrtRunner : public Algorithms
{
public:
    struct VectorStats
    {
        float min = 0.0f;
        float max = 0.0f;
        float mean = 0.0f;
    };

    ParkourOrtRunner(const std::string& depth_encoder_path, const std::string& actor_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "parkour_onnx")
    {
        session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        depth_encoder_session_ = std::make_unique<Ort::Session>(env_, depth_encoder_path.c_str(), session_options_);
        actor_session_ = std::make_unique<Ort::Session>(env_, actor_path.c_str(), session_options_);

        depth_input_shape_ = depth_encoder_session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        depth_input_size_ = tensor_size(depth_input_shape_);
        actor_input_shape_ = actor_session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        actor_input_size_ = tensor_size(actor_input_shape_);
        actor_output_shape_ = actor_session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        depth_input_name_ = "input";
        depth_output_name_ = "output";
        actor_input_name_ = "input";
        actor_output_name_ = "output";

        action.resize(static_cast<size_t>(actor_output_shape_.at(1)));
    }

    std::vector<float> last_policy_order_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return last_policy_order_action_;
    }

    std::vector<float> last_deploy_order_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return last_deploy_order_action_;
    }

    VectorStats last_depth_stats()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return last_depth_stats_;
    }

    VectorStats last_proprio_stats()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return last_proprio_stats_;
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) override
    {
        const auto& proprio = obs.at("proprio");
        const auto& depth_image = obs.at("depth_image");
        if (depth_image.size() != static_cast<size_t>(depth_input_size_)) {
            throw std::runtime_error("Unexpected depth image size for parkour encoder input.");
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::array<const char*, 1> depth_input_names{depth_input_name_.c_str()};
        std::array<const char*, 1> depth_output_names{depth_output_name_.c_str()};
        std::vector<Ort::Value> depth_inputs;
        depth_inputs.emplace_back(
            Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(depth_image.data()),
                depth_input_size_,
                depth_input_shape_.data(),
                depth_input_shape_.size()
            )
        );
        auto depth_outputs = depth_encoder_session_->Run(
            Ort::RunOptions{nullptr},
            depth_input_names.data(),
            depth_inputs.data(),
            depth_inputs.size(),
            depth_output_names.data(),
            depth_output_names.size()
        );

        const auto latent_info = depth_outputs.front().GetTensorTypeAndShapeInfo().GetShape();
        const size_t latent_size = static_cast<size_t>(tensor_size(latent_info));
        const float* latent_ptr = depth_outputs.front().GetTensorData<float>();

        std::vector<float> policy_order_proprio = proprio;
        reorder_deploy_order_joint_blocks_to_policy_order(policy_order_proprio);

        std::vector<float> actor_input;
        actor_input.reserve(policy_order_proprio.size() + latent_size);
        actor_input.insert(actor_input.end(), policy_order_proprio.begin(), policy_order_proprio.end());
        actor_input.insert(actor_input.end(), latent_ptr, latent_ptr + latent_size);
        if (actor_input.size() != static_cast<size_t>(actor_input_size_)) {
            throw std::runtime_error("Unexpected actor input size for parkour policy input.");
        }

        std::array<const char*, 1> actor_input_names{actor_input_name_.c_str()};
        std::array<const char*, 1> actor_output_names{actor_output_name_.c_str()};
        std::vector<Ort::Value> actor_inputs;
        actor_inputs.emplace_back(
            Ort::Value::CreateTensor<float>(
                memory_info,
                actor_input.data(),
                actor_input.size(),
                actor_input_shape_.data(),
                actor_input_shape_.size()
            )
        );
        auto actor_outputs = actor_session_->Run(
            Ort::RunOptions{nullptr},
            actor_input_names.data(),
            actor_inputs.data(),
            actor_inputs.size(),
            actor_output_names.data(),
            actor_output_names.size()
        );

        auto* output_ptr = actor_outputs.front().GetTensorMutableData<float>();
        std::vector<float> policy_order_action(action.size(), 0.0f);
        std::memcpy(policy_order_action.data(), output_ptr, policy_order_action.size() * sizeof(float));
        std::vector<float> deploy_order_action = policy_action_to_deploy_action(policy_order_action);
        std::lock_guard<std::mutex> lock(act_mtx_);
        action = std::move(deploy_order_action);
        last_policy_order_action_ = policy_order_action;
        last_deploy_order_action_ = action;
        last_depth_stats_ = compute_vector_stats(depth_image);
        last_proprio_stats_ = compute_vector_stats(proprio);
        static std::atomic<int> debug_log_count{0};
        const int debug_index = debug_log_count.fetch_add(1);
        if (debug_index == 0 || (debug_index < 400 && debug_index % 50 == 0)) {
            const auto depth_minmax = std::minmax_element(depth_image.begin(), depth_image.end());
            const auto depth_mean = depth_image.empty()
                ? 0.0f
                : std::accumulate(depth_image.begin(), depth_image.end(), 0.0f) / static_cast<float>(depth_image.size());
            const auto action_minmax = std::minmax_element(action.begin(), action.end());
            auto mean_abs = [](const auto begin, const auto end) {
                if (begin >= end) {
                    return 0.0f;
                }
                float sum = 0.0f;
                size_t count = 0;
                for (auto it = begin; it != end; ++it) {
                    sum += std::abs(*it);
                    ++count;
                }
                return count > 0 ? sum / static_cast<float>(count) : 0.0f;
            };
            const size_t joint_dim = action.size();
            const size_t hist = 8;
            const size_t base_ang_begin = 3 * (hist - 1);
            const size_t proj_grav_begin = 3 * hist + 3 * (hist - 1);
            const size_t vel_cmd_begin = 3 * hist + 3 * hist + 3 * (hist - 1);
            const size_t joint_pos_begin = 3 * hist + 3 * hist + 3 * hist + joint_dim * (hist - 1);
            const size_t joint_vel_begin = 3 * hist + 3 * hist + 3 * hist + joint_dim * hist + joint_dim * (hist - 1);
            const size_t last_action_begin = 3 * hist + 3 * hist + 3 * hist + joint_dim * hist + joint_dim * hist + joint_dim * (hist - 1);
            std::cout
                << "ORDER_PARITY_OK"
                << " observation_order_source=ONNX_POLICY_JOINT_NAMES"
                << " action_order_source=ONNX_POLICY_JOINT_NAMES->TRAINING_JOINT_NAMES"
                << " joint_ids_map=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28"
                << " policy_to_robot_joint_indices=" << policy_to_robot_joint_indices_csv()
                << std::endl;
            std::cout
                << "[parkour debug] step=" << debug_index
                << " proprio_size=" << proprio.size()
                << " depth_size=" << depth_image.size()
                << " actor_input_size=" << actor_input.size()
                << " depth[min,max,mean]=["
                << (depth_minmax.first != depth_image.end() ? *depth_minmax.first : 0.0f) << ","
                << (depth_minmax.second != depth_image.end() ? *depth_minmax.second : 0.0f) << ","
                << depth_mean << "]"
                << " latest_base_ang_vel=["
                << proprio[base_ang_begin] << "," << proprio[base_ang_begin + 1] << "," << proprio[base_ang_begin + 2] << "]"
                << " latest_projected_gravity=["
                << proprio[proj_grav_begin] << "," << proprio[proj_grav_begin + 1] << "," << proprio[proj_grav_begin + 2] << "]"
                << " latest_velocity_commands=["
                << proprio[vel_cmd_begin] << "," << proprio[vel_cmd_begin + 1] << "," << proprio[vel_cmd_begin + 2] << "]"
                << " mean_abs[joint_pos,joint_vel,last_action]=["
                << mean_abs(proprio.begin() + joint_pos_begin, proprio.begin() + joint_pos_begin + joint_dim) << ","
                << mean_abs(proprio.begin() + joint_vel_begin, proprio.begin() + joint_vel_begin + joint_dim) << ","
                << mean_abs(proprio.begin() + last_action_begin, proprio.begin() + last_action_begin + joint_dim) << "]"
                << " action[min,max]=["
                << (action_minmax.first != action.end() ? *action_minmax.first : 0.0f) << ","
                << (action_minmax.second != action.end() ? *action_minmax.second : 0.0f) << "]"
                << " policy_order_action_head=";
            for (size_t i = 0; i < std::min<size_t>(10, policy_order_action.size()); ++i) {
                std::cout << policy_order_action[i] << (i + 1 < std::min<size_t>(10, policy_order_action.size()) ? "," : "");
            }
            std::cout << " deploy_order_action_head=";
            for (size_t i = 0; i < std::min<size_t>(10, action.size()); ++i) {
                std::cout << action[i] << (i + 1 < std::min<size_t>(10, action.size()) ? "," : "");
            }
            std::cout << std::endl;
        }
        return action;
    }

private:
    // ONNX_POLICY_JOINT_NAMES -> TRAINING_JOINT_NAMES/deploy motor order.
    // Must stay in lockstep with src/parkour/contract.py.
    static constexpr std::array<int, 29> kPolicyToDeployJointIndices{
        15, 22, 14, 16, 23, 13, 17, 24, 12, 18,
        25, 0, 6, 19, 26, 1, 7, 20, 27, 2,
        8, 21, 28, 3, 9, 4, 10, 5, 11
    };

    static void reorder_deploy_order_chunk_to_policy_order(
        const std::vector<float>& deploy_order_chunk,
        std::vector<float>& policy_order_chunk)
    {
        policy_order_chunk.assign(kPolicyToDeployJointIndices.size(), 0.0f);
        for (size_t policy_index = 0; policy_index < kPolicyToDeployJointIndices.size(); ++policy_index) {
            const auto deploy_index = static_cast<size_t>(kPolicyToDeployJointIndices[policy_index]);
            if (deploy_index >= deploy_order_chunk.size()) {
                throw std::runtime_error("Parkour joint-order mapping exceeds deploy-order chunk size.");
            }
            policy_order_chunk[policy_index] = deploy_order_chunk[deploy_index];
        }
    }

    static void reorder_deploy_order_joint_blocks_to_policy_order(std::vector<float>& proprio)
    {
        constexpr size_t hist = 8;
        constexpr size_t joint_dim = kPolicyToDeployJointIndices.size();
        constexpr size_t joint_pos_group_begin = 3 * hist + 3 * hist + 3 * hist;
        constexpr size_t joint_vel_group_begin = joint_pos_group_begin + joint_dim * hist;
        constexpr size_t last_action_group_begin = joint_vel_group_begin + joint_dim * hist;

        auto reorder_group = [&](size_t group_begin) {
            for (size_t h = 0; h < hist; ++h) {
                const size_t chunk_begin = group_begin + joint_dim * h;
                if (chunk_begin + joint_dim > proprio.size()) {
                    throw std::runtime_error("Unexpected parkour proprio layout while applying joint-order mapping.");
                }
                std::vector<float> deploy_order_chunk(proprio.begin() + chunk_begin, proprio.begin() + chunk_begin + joint_dim);
                std::vector<float> policy_order_chunk;
                reorder_deploy_order_chunk_to_policy_order(deploy_order_chunk, policy_order_chunk);
                std::copy(policy_order_chunk.begin(), policy_order_chunk.end(), proprio.begin() + chunk_begin);
            }
        };

        reorder_group(joint_pos_group_begin);
        reorder_group(joint_vel_group_begin);
        reorder_group(last_action_group_begin);
    }

    static std::vector<float> policy_action_to_deploy_action(const std::vector<float>& policy_order_action)
    {
        std::vector<float> deploy_order_action(policy_order_action.size(), 0.0f);
        for (size_t policy_index = 0; policy_index < kPolicyToDeployJointIndices.size(); ++policy_index) {
            const auto deploy_index = static_cast<size_t>(kPolicyToDeployJointIndices[policy_index]);
            if (policy_index >= policy_order_action.size() || deploy_index >= deploy_order_action.size()) {
                throw std::runtime_error("Parkour action-order mapping exceeds action vector size.");
            }
            deploy_order_action[deploy_index] = policy_order_action[policy_index];
        }
        return deploy_order_action;
    }

    static std::string policy_to_robot_joint_indices_csv()
    {
        std::string text;
        for (size_t i = 0; i < kPolicyToDeployJointIndices.size(); ++i) {
            if (i > 0) {
                text += ",";
            }
            text += std::to_string(kPolicyToDeployJointIndices[i]);
        }
        return text;
    }

    static VectorStats compute_vector_stats(const std::vector<float>& values)
    {
        if (values.empty()) {
            return {};
        }
        const auto minmax = std::minmax_element(values.begin(), values.end());
        const float mean = std::accumulate(values.begin(), values.end(), 0.0f)
            / static_cast<float>(values.size());
        return {*minmax.first, *minmax.second, mean};
    }

    static int64_t tensor_size(const std::vector<int64_t>& shape)
    {
        int64_t size = 1;
        for (const auto dim : shape) {
            if (dim <= 0) {
                throw std::runtime_error("Dynamic ONNX shapes are not supported in ParkourOrtRunner.");
            }
            size *= dim;
        }
        return size;
    }

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> depth_encoder_session_;
    std::unique_ptr<Ort::Session> actor_session_;
    std::vector<int64_t> depth_input_shape_;
    std::vector<int64_t> actor_input_shape_;
    std::vector<int64_t> actor_output_shape_;
    int64_t depth_input_size_ = 0;
    int64_t actor_input_size_ = 0;
    std::string depth_input_name_;
    std::string depth_output_name_;
    std::string actor_input_name_;
    std::string actor_output_name_;
    std::vector<float> last_policy_order_action_;
    std::vector<float> last_deploy_order_action_;
    VectorStats last_depth_stats_;
    VectorStats last_proprio_stats_;
};

} // namespace isaaclab
