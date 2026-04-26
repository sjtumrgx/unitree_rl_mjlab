#pragma once

#include "FSM/FSMState.h"
#include "ParkourDepthProvider.h"
#include "ParkourOrtRunner.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "FSM/rl_reset_utils.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);

    void pre_run() override
    {
        FSMState::pre_run();

        uint32_t current_tick = 0;
        {
            std::lock_guard<std::mutex> lowstate_lock(lowstate->mutex_);
            current_tick = lowstate->msg_.tick();
        }

        const bool reset_on_tick_rewind_enabled =
            reset_on_tick_rewind_ || param::sim_loopback_interactive || param::sim_autostart_parkour;
        if (reset_on_tick_rewind_enabled && lowstate_tick_initialized_ && rl_reset::tick_rewound(last_lowstate_tick_, current_tick))
        {
            std::lock_guard<std::mutex> env_lock(env_mutex_);
            env->robot->update();
            env->reset();
            policy_blend_start_action_ = env->action_manager->processed_actions();
            policy_blend_elapsed_s_ = 0.0f;
            if (depth_provider_ && depth_provider_->enabled()) {
                depth_provider_->reset(env->robot.get());
            }
            ++reset_epoch_;
            lowstate_tick_history_.clear();
            last_replay_applied_ = false;
            last_replay_mode_ = "off";
            last_replay_source_step_ = -1;
            policy_wall_start_time_ = std::chrono::high_resolution_clock::now();
            spdlog::info(
                "State_{} detected lowstate tick rewind ({} -> {}); resetting deploy env/depth/action history.",
                getStateString(),
                last_lowstate_tick_,
                current_tick
            );
        }

        last_lowstate_tick_ = current_tick;
        lowstate_tick_initialized_ = true;
    }

    void enter()
    {
        spdlog::info(
            "ENTERED_PARKOUR state={} depth_provider_enabled={} policy_blend_duration_s={} policy_tick_sync={}",
            getStateString(),
            depth_provider_ && depth_provider_->enabled(),
            policy_blend_duration_s_,
            param::policy_tick_sync
        );
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        {
            std::lock_guard<std::mutex> env_lock(env_mutex_);
            env->robot->update();
            env->reset();
            policy_blend_start_action_ = env->action_manager->processed_actions();
            policy_blend_elapsed_s_ = 0.0f;
            gait_record_step_ = 0;
            ++reset_epoch_;
            lowstate_tick_history_.clear();
            last_replay_applied_ = false;
            last_replay_mode_ = "off";
            last_replay_source_step_ = -1;
            policy_wall_start_time_ = std::chrono::high_resolution_clock::now();
            open_gait_record_if_requested();
            load_gait_replay_if_requested();
            if (depth_provider_ && depth_provider_->enabled()) {
                depth_provider_->reset(env->robot.get());
            }
        }

        {
            std::lock_guard<std::mutex> lowstate_lock(lowstate->mutex_);
            last_lowstate_tick_ = lowstate->msg_.tick();
        }
        lowstate_tick_initialized_ = true;

        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);
            const auto tick_dt = static_cast<uint32_t>(std::max(1, static_cast<int>(std::llround(env->step_dt / 1.0e-3f))));
            auto sleepTill = clock::now() + dt;
            bool tick_sync_initialized = false;
            uint32_t next_policy_tick = 0;

            while (policy_thread_running)
            {
                if (param::policy_tick_sync) {
                    while (policy_thread_running) {
                        uint32_t current_tick = 0;
                        {
                            std::lock_guard<std::mutex> lowstate_lock(lowstate->mutex_);
                            current_tick = lowstate->msg_.tick();
                        }
                        if (!tick_sync_initialized) {
                            next_policy_tick = current_tick;
                            tick_sync_initialized = true;
                            break;
                        }
                        if (current_tick >= next_policy_tick) {
                            break;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                    if (!policy_thread_running) {
                        break;
                    }
                }
                {
                    std::lock_guard<std::mutex> env_lock(env_mutex_);
                    if (depth_provider_ && depth_provider_->enabled()) {
                        depth_provider_->update(env->robot.get());
                        if (!depth_provider_->has_valid_frame()) {
                            if (param::policy_tick_sync) {
                                next_policy_tick += tick_dt;
                            } else {
                                std::this_thread::sleep_until(sleepTill);
                                sleepTill += dt;
                            }
                            continue;
                        }
                    }
                    env->step();
                    apply_gait_replay_raw_action();
                    const float next_blend_elapsed = policy_blend_elapsed_s_.load() + env->step_dt;
                    const float blend_alpha = policy_blend_duration_s_ > 0.0f
                        ? std::clamp(next_blend_elapsed / policy_blend_duration_s_, 0.0f, 1.0f)
                        : 1.0f;
                    if (blend_alpha < 1.0f) {
                        auto blended_raw_action = env->action_manager->action();
                        for (auto& value : blended_raw_action) {
                            value *= blend_alpha;
                        }
                        env->action_manager->process_action(blended_raw_action);
                    }
                    apply_gait_replay_target_q();
                    policy_blend_elapsed_s_.store(next_blend_elapsed);
                    record_gait_sample(blend_alpha);
                }
                if (param::policy_tick_sync) {
                    next_policy_tick += tick_dt;
                } else {
                    std::this_thread::sleep_until(sleepTill);
                    sleepTill += dt;
                }
            }
        });
    }

    void run();

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
        close_gait_record();
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
    std::unique_ptr<ParkourDepthProvider> depth_provider_;
    std::mutex env_mutex_;
    uint32_t last_lowstate_tick_ = 0;
    bool lowstate_tick_initialized_ = false;
    bool reset_on_tick_rewound_legacy_unused_ = false;
    bool reset_on_tick_rewind_ = false;
    std::vector<float> policy_blend_start_action_;
    float policy_blend_duration_s_ = 1.0f;
    std::atomic<float> policy_blend_elapsed_s_{0.0f};
    struct GaitReplaySample
    {
        int source_step = -1;
        std::vector<float> raw_action_policy_order;
        std::vector<float> target_q_deploy_order;
    };

    std::ofstream gait_record_stream_;
    size_t gait_record_step_ = 0;
    std::vector<GaitReplaySample> gait_replay_samples_;
    bool gait_replay_loaded_ = false;
    std::string last_replay_mode_ = "off";
    int last_replay_source_step_ = -1;
    bool last_replay_applied_ = false;
    std::deque<uint32_t> lowstate_tick_history_;
    uint64_t reset_epoch_ = 0;
    std::chrono::high_resolution_clock::time_point policy_wall_start_time_;

    void open_gait_record_if_requested();
    void close_gait_record();
    void load_gait_replay_if_requested();
    bool apply_gait_replay_raw_action();
    bool apply_gait_replay_target_q();
    const GaitReplaySample* current_gait_replay_sample() const;
    std::vector<float> target_q_to_raw_action(const std::vector<float>& target_q) const;
    void record_gait_sample(float blend_alpha);
};

REGISTER_FSM(State_RLBase)
