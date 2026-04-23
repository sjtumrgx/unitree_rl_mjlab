#include "ParkourObservations.h"
#include "ParkourArticulation.h"
#include "FSM/State_RLBase.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"

#include <algorithm>
#include <atomic>
#include <iostream>

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
        std::make_shared<unitree::ParkourArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
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

void State_RLBase::run()
{
    std::vector<float> action;
    {
        std::lock_guard<std::mutex> env_lock(env_mutex_);
        action = env->action_manager->processed_actions();
    }
    const float blend_alpha = policy_blend_duration_s_ > 0.0f
        ? std::clamp(policy_blend_elapsed_s_.load() / policy_blend_duration_s_, 0.0f, 1.0f)
        : 1.0f;
    if (!policy_blend_start_action_.empty() && policy_blend_start_action_.size() == action.size() && blend_alpha < 1.0f) {
        std::vector<float> blended_action = action;
        for (size_t i = 0; i < action.size(); ++i) {
            blended_action[i] = (1.0f - blend_alpha) * policy_blend_start_action_[i] + blend_alpha * action[i];
        }
        action = std::move(blended_action);
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
