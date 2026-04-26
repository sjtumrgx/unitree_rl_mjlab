#pragma once

#include "FSM/FSMState.h"
#include "ParkourDepthProvider.h"
#include "ParkourOrtRunner.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "FSM/rl_reset_utils.h"
#include <atomic>
#include <mutex>
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

        if (reset_on_tick_rewind_ && lowstate_tick_initialized_ && rl_reset::tick_rewound(last_lowstate_tick_, current_tick))
        {
            std::lock_guard<std::mutex> env_lock(env_mutex_);
            env->reset();
            if (depth_provider_ && depth_provider_->enabled()) {
                depth_provider_->reset(env->robot.get());
            }
            spdlog::info(
                "State_{} detected lowstate tick rewind ({} -> {}); resetting deploy env history.",
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
            "ENTERED_PARKOUR state={} depth_provider_enabled={} policy_blend_duration_s={}",
            getStateString(),
            depth_provider_ && depth_provider_->enabled(),
            policy_blend_duration_s_
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
            auto sleepTill = clock::now() + dt;

            while (policy_thread_running)
            {
                {
                    std::lock_guard<std::mutex> env_lock(env_mutex_);
                    if (depth_provider_ && depth_provider_->enabled()) {
                        depth_provider_->update(env->robot.get());
                        if (!depth_provider_->has_valid_frame()) {
                            std::this_thread::sleep_until(sleepTill);
                            sleepTill += dt;
                            continue;
                        }
                    }
                    env->step();
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
                    policy_blend_elapsed_s_.store(next_blend_elapsed);
                }
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
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
};

REGISTER_FSM(State_RLBase)
