// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "SupportGeometryProvider.h"
#include "rl_reset_utils.h"
#include <mutex>

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

        if (lowstate_tick_initialized_ && rl_reset::tick_rewound(last_lowstate_tick_, current_tick))
        {
            std::lock_guard<std::mutex> env_lock(env_mutex_);
            env->reset();
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
        // set gain
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
        }

        {
            std::lock_guard<std::mutex> lowstate_lock(lowstate->mutex_);
            last_lowstate_tick_ = lowstate->msg_.tick();
        }
        lowstate_tick_initialized_ = true;

        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;

            while (policy_thread_running)
            {
                {
                    std::lock_guard<std::mutex> env_lock(env_mutex_);
                    if (support_geometry_provider && support_geometry_provider->enabled()) {
                        support_geometry_provider->update(env->robot.get());
                    }
                    env->step();
                }

                // Sleep
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
    std::unique_ptr<SupportGeometryProvider> support_geometry_provider;
    std::mutex env_mutex_;
    uint32_t last_lowstate_tick_ = 0;
    bool lowstate_tick_initialized_ = false;
};

REGISTER_FSM(State_RLBase)
