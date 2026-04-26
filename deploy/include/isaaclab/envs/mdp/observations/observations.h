// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>

#include <eigen3/Eigen/Dense>

#include "isaaclab/envs/manager_based_rl_env.h"
#include "param.h"

namespace isaaclab
{
namespace mdp
{
namespace detail
{

inline float wrap_to_pi(float angle)
{
    while (angle > static_cast<float>(M_PI)) {
        angle -= 2.0f * static_cast<float>(M_PI);
    }
    while (angle < -static_cast<float>(M_PI)) {
        angle += 2.0f * static_cast<float>(M_PI);
    }
    return angle;
}

inline float yaw_from_quaternion(const Eigen::Quaternionf& q)
{
    return std::atan2(
        2.0f * (q.w() * q.z() + q.x() * q.y()),
        1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z())
    );
}

} // namespace detail

inline std::vector<float> keyboard_velocity_command(ManagerBasedRLEnv* env)
{
    std::vector<float> obs(3, 0.0f);
    if (!param::keyboard_control || !param::active_keyboard)
    {
        return obs;
    }

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];
    const std::string key = param::active_keyboard->key();
    const auto keyboard_cfg = env->cfg["commands"]["base_velocity"]["keyboard"];
    if (keyboard_cfg)
    {
        static std::vector<float> parkour_keyboard_command(3, 0.0f);
        static bool command_initialized = false;
        static std::string last_logged_key = "";
        const float lin_vel_x_min = keyboard_cfg["lin_vel_x_min"].as<float>();
        const float lin_vel_x_max = keyboard_cfg["lin_vel_x_max"].as<float>();
        const float ang_vel_z_min = keyboard_cfg["ang_vel_z_min"].as<float>();
        const float ang_vel_z_max = keyboard_cfg["ang_vel_z_max"].as<float>();
        const float lin_vel_step = keyboard_cfg["lin_vel_step"].as<float>(0.1f);
        const float cruise_speed = std::clamp(
            std::abs(param::sim_command_x) > 1.0e-4f ? std::abs(param::sim_command_x) : lin_vel_step,
            lin_vel_x_min,
            lin_vel_x_max
        );
        const float idle_speed = param::sim_loopback_interactive
            ? std::clamp(param::sim_keyboard_idle_command_x, -std::max(1.0f, std::abs(lin_vel_x_max)), lin_vel_x_max)
            : 0.0f;

        if (!command_initialized)
        {
            parkour_keyboard_command[0] = idle_speed;
            command_initialized = true;
        }

        if (key == "w" || key == "up")
        {
            parkour_keyboard_command[0] = cruise_speed;
        }
        else if (key == "s" || key == "down" || key == "x" || key == " ")
        {
            parkour_keyboard_command[0] = idle_speed;
            parkour_keyboard_command[1] = 0.0f;
            parkour_keyboard_command[2] = 0.0f;
        }
        else if (key == "a" || key == "left" || key == "q")
        {
            parkour_keyboard_command[2] = ang_vel_z_max;
        }
        else if (key == "d" || key == "right" || key == "e")
        {
            parkour_keyboard_command[2] = ang_vel_z_min;
        }
        else if (key == "c")
        {
            parkour_keyboard_command[2] = 0.0f;
        }
        else if (key == "+" || key == "=")
        {
            parkour_keyboard_command[0] = std::clamp(
                parkour_keyboard_command[0] + lin_vel_step,
                lin_vel_x_min,
                lin_vel_x_max
            );
        }
        else if (key == "-")
        {
            parkour_keyboard_command[0] = std::clamp(
                parkour_keyboard_command[0] - lin_vel_step,
                lin_vel_x_min,
                lin_vel_x_max
            );
        }

        if (key != last_logged_key && !key.empty())
        {
            std::cout << "[parkour keyboard] key=" << key
                      << " command=[" << parkour_keyboard_command[0] << ","
                      << parkour_keyboard_command[1] << ","
                      << parkour_keyboard_command[2] << "]" << std::endl;
            last_logged_key = key;
        }
        if (key.empty())
        {
            last_logged_key.clear();
        }

        return parkour_keyboard_command;
    }

    if (key == "w")
    {
        obs[0] = cfg["lin_vel_x"][1].as<float>();
    }
    else if (key == "s")
    {
        obs[0] = cfg["lin_vel_x"][0].as<float>();
    }
    else if (key == "a")
    {
        obs[1] = cfg["lin_vel_y"][1].as<float>();
    }
    else if (key == "d")
    {
        obs[1] = cfg["lin_vel_y"][0].as<float>();
    }
    else if (key == "q")
    {
        obs[2] = cfg["ang_vel_z"][1].as<float>();
    }
    else if (key == "e")
    {
        obs[2] = cfg["ang_vel_z"][0].as<float>();
    }

    return obs;
}

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(velocity_commands)
{
    if (param::sim_autostart_parkour)
    {
        float yaw_command = param::sim_command_yaw;
        if (param::sim_heading_lock)
        {
            const float current_yaw = detail::yaw_from_quaternion(env->robot->data.root_quat_w);
            const float yaw_error = detail::wrap_to_pi(param::sim_heading_target_yaw - current_yaw);
            const float correction = std::clamp(
                param::sim_heading_kp * yaw_error,
                -std::abs(param::sim_heading_max_yaw),
                std::abs(param::sim_heading_max_yaw)
            );
            yaw_command = std::clamp(
                param::sim_command_yaw + correction,
                -1.0f,
                1.0f
            );

            static size_t heading_log_count = 0;
            if (heading_log_count++ % 250 == 0)
            {
                std::cout << "SIM_HEADING_LOCK yaw=" << current_yaw
                          << " target_yaw=" << param::sim_heading_target_yaw
                          << " yaw_error=" << yaw_error
                          << " yaw_command=" << yaw_command
                          << " kp=" << param::sim_heading_kp
                          << " max_yaw=" << param::sim_heading_max_yaw
                          << std::endl;
            }
        }
        std::vector<float> obs{
            param::sim_command_x,
            param::sim_command_y,
            yaw_command,
        };
        param::sim_observed_command_x.store(obs[0]);
        param::sim_observed_command_y.store(obs[1]);
        param::sim_observed_command_yaw.store(obs[2]);
        static bool logged_sim_command = false;
        if (!logged_sim_command)
        {
            std::cout << "NONZERO_COMMAND x=" << obs[0]
                      << " y=" << obs[1]
                      << " yaw=" << obs[2]
                      << " source=sim-autostart"
                      << " heading_lock=" << param::sim_heading_lock
                      << " target_yaw=" << param::sim_heading_target_yaw
                      << std::endl;
            logged_sim_command = true;
        }
        return obs;
    }

    if (param::keyboard_control)
    {
        auto obs = keyboard_velocity_command(env);
        param::sim_observed_command_x.store(obs[0]);
        param::sim_observed_command_y.store(obs[1]);
        param::sim_observed_command_yaw.store(obs[2]);
        return obs;
    }

    std::vector<float> obs(3);
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());
    param::sim_observed_command_x.store(obs[0]);
    param::sim_observed_command_y.store(obs[1]);
    param::sim_observed_command_yaw.store(obs[2]);

    return obs;
}


REGISTER_OBSERVATION(support_depth)
{
    const auto sensor_name = params["sensor_name"].as<std::string>("support_depth");
    const auto expected_size = params["expected_size"].as<int>(0);
    auto it = env->robot->data.named_observations.find(sensor_name);
    if (it != env->robot->data.named_observations.end()) {
        return it->second;
    }
    if (expected_size <= 0) {
        return {};
    }
    return std::vector<float>(expected_size, 0.0f);
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    auto cmd = isaaclab::mdp::velocity_commands(env, params);
    float cmd_norm = std::sqrt(
        cmd[0] * cmd[0] +
        cmd[1] * cmd[1] +
        cmd[2] * cmd[2]
    );

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);

    if (cmd_norm < 0.1f)
    {
        obs[0] = 0.0f;
        obs[1] = 0.0f;
    }

    return obs;
}

}
}
