#include "ParkourObservations.h"
#include "ParkourArticulation.h"
#include "FSM/State_RLBase.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    reset_on_tick_rewind_ = cfg["reset_on_tick_rewind"].as<bool>(false);
    ensureParkourObservationRegistration();
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    const auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::ParkourArticulation<LowState_t::SharedPtr>>(
            FSMState::lowstate,
            param::sim_autostart_parkour
        )
    );
    depth_provider_ = std::make_unique<ParkourDepthProvider>(deploy_cfg);
    if (depth_provider_->enabled()) {
        depth_provider_->initialize(env->robot.get());
    }
    env->alg = std::make_unique<isaaclab::ParkourOrtRunner>(
        (policy_dir / "exported" / "0-depth_encoder.onnx").string(),
        (policy_dir / "exported" / "actor.onnx").string()
    );

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

namespace
{

std::string json_vector(const std::vector<float>& values)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << std::setprecision(9) << values[i];
    }
    os << "]";
    return os.str();
}

std::string json_vector3(const Eigen::Vector3f& values)
{
    std::ostringstream os;
    os << "["
       << std::setprecision(9) << values[0] << ","
       << std::setprecision(9) << values[1] << ","
       << std::setprecision(9) << values[2] << "]";
    return os.str();
}

std::string json_stats(const isaaclab::ParkourOrtRunner::VectorStats& stats)
{
    std::ostringstream os;
    os << "{\"min\":" << std::setprecision(9) << stats.min
       << ",\"max\":" << std::setprecision(9) << stats.max
       << ",\"mean\":" << std::setprecision(9) << stats.mean
       << "}";
    return os.str();
}

uint32_t lowstate_tick(const LowState_t::SharedPtr& lowstate)
{
    std::lock_guard<std::mutex> lowstate_lock(lowstate->mutex_);
    return lowstate->msg_.tick();
}

} // namespace

void State_RLBase::open_gait_record_if_requested()
{
    close_gait_record();
    if (param::gait_record_jsonl.empty()) {
        return;
    }
    std::filesystem::path output_path = param::gait_record_jsonl;
    if (output_path.is_relative()) {
        output_path = param::proj_dir / output_path;
    }
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    gait_record_stream_.open(output_path, std::ios::out | std::ios::trunc);
    if (!gait_record_stream_) {
        spdlog::error("Failed to open parkour gait record JSONL: {}", output_path.string());
        return;
    }
    spdlog::info(
        "Recording parkour gait parity JSONL to {} every {} policy step(s).",
        output_path.string(),
        param::gait_record_every
    );
}

void State_RLBase::close_gait_record()
{
    if (gait_record_stream_.is_open()) {
        gait_record_stream_.close();
    }
}

void State_RLBase::record_gait_sample(float blend_alpha)
{
    if (!gait_record_stream_.is_open()) {
        return;
    }
    const int every = std::max(param::gait_record_every, 1);
    if (gait_record_step_ % static_cast<size_t>(every) != 0) {
        ++gait_record_step_;
        return;
    }

    auto* runner = dynamic_cast<isaaclab::ParkourOrtRunner*>(env->alg.get());
    const auto raw_action_deploy = env->action_manager->action();
    const auto processed_action_deploy = env->action_manager->processed_actions();
    const auto raw_action_policy = runner ? runner->last_policy_order_action() : std::vector<float>{};
    const auto depth_stats = runner ? runner->last_depth_stats() : isaaclab::ParkourOrtRunner::VectorStats{};
    const auto proprio_stats = runner ? runner->last_proprio_stats() : isaaclab::ParkourOrtRunner::VectorStats{};
    const auto tick = lowstate_tick(lowstate);

    const std::vector<float> joint_pos(
        env->robot->data.joint_pos.data(),
        env->robot->data.joint_pos.data() + env->robot->data.joint_pos.size()
    );
    const std::vector<float> joint_vel(
        env->robot->data.joint_vel.data(),
        env->robot->data.joint_vel.data() + env->robot->data.joint_vel.size()
    );
    const std::vector<float> command{
        param::sim_observed_command_x.load(),
        param::sim_observed_command_y.load(),
        param::sim_observed_command_yaw.load(),
    };

    gait_record_stream_
        << "{\"source\":\"cpp_dds_ctrl\""
        << ",\"step\":" << gait_record_step_
        << ",\"elapsed_seconds\":" << std::setprecision(9) << (env->episode_length * env->step_dt)
        << ",\"lowstate_tick\":" << tick
        << ",\"startup_blend_alpha\":" << std::setprecision(9) << blend_alpha
        << ",\"command\":" << json_vector(command)
        << ",\"base_ang_vel\":" << json_vector3(env->robot->data.root_ang_vel_b)
        << ",\"projected_gravity\":" << json_vector3(env->robot->data.projected_gravity_b)
        << ",\"joint_pos_deploy_order\":" << json_vector(joint_pos)
        << ",\"joint_vel_deploy_order\":" << json_vector(joint_vel)
        << ",\"raw_action_policy_order\":" << json_vector(raw_action_policy)
        << ",\"raw_action_deploy_order\":" << json_vector(raw_action_deploy)
        << ",\"applied_action_deploy_order\":" << json_vector(raw_action_deploy)
        << ",\"target_q_deploy_order\":" << json_vector(processed_action_deploy)
        << ",\"depth_stats\":" << json_stats(depth_stats)
        << ",\"proprio_stats\":" << json_stats(proprio_stats)
        << "}\n";
    gait_record_stream_.flush();
    ++gait_record_step_;
}

void State_RLBase::run()
{
    std::vector<float> action;
    {
        std::lock_guard<std::mutex> env_lock(env_mutex_);
        action = env->action_manager->processed_actions();
    }
    static std::atomic<bool> logged_once{false};
    if (!logged_once.exchange(true)) {
        const auto minmax = std::minmax_element(action.begin(), action.end());
        std::cout << "[parkour debug] processed_action[min,max]=["
                  << (minmax.first != action.end() ? *minmax.first : 0.0f) << ","
                  << (minmax.second != action.end() ? *minmax.second : 0.0f) << "] head=";
        for (size_t i = 0; i < std::min<size_t>(10, action.size()); ++i) {
            std::cout << action[i] << (i + 1 < std::min<size_t>(10, action.size()) ? "," : "");
        }
        std::vector<float> motor_order(action.size(), 0.0f);
        for (size_t i = 0; i < action.size(); ++i) {
            const auto motor_index = static_cast<size_t>(env->robot->data.joint_ids_map[i]);
            if (motor_index < motor_order.size()) {
                motor_order[motor_index] = action[i];
            }
        }
        std::cout << " motor_order=";
        for (size_t i = 0; i < motor_order.size(); ++i) {
            std::cout << motor_order[i] << (i + 1 < motor_order.size() ? "," : "");
        }
        std::cout << std::endl;
    }
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
