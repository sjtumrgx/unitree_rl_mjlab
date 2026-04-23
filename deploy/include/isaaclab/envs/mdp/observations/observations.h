// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <iostream>

#include "isaaclab/envs/manager_based_rl_env.h"
#include "param.h"

namespace isaaclab
{
namespace mdp
{

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
        static std::string last_logged_key = "";
        const float lin_vel_x_min = keyboard_cfg["lin_vel_x_min"].as<float>();
        const float lin_vel_x_max = keyboard_cfg["lin_vel_x_max"].as<float>();
        const float ang_vel_z_min = keyboard_cfg["ang_vel_z_min"].as<float>();
        const float ang_vel_z_max = keyboard_cfg["ang_vel_z_max"].as<float>();
        const float lin_vel_step = keyboard_cfg["lin_vel_step"].as<float>(0.1f);

        if (key == "w")
        {
            parkour_keyboard_command[0] = std::clamp(
                parkour_keyboard_command[0] + lin_vel_step,
                lin_vel_x_min,
                lin_vel_x_max
            );
        }
        else if (key == "f")
        {
            parkour_keyboard_command[2] = ang_vel_z_max;
        }
        else if (key == "g")
        {
            parkour_keyboard_command[2] = ang_vel_z_min;
        }
        else if (key == "s")
        {
            parkour_keyboard_command[2] = 0.0f;
        }
        else if (key == "x")
        {
            parkour_keyboard_command[0] = 0.0f;
            parkour_keyboard_command[1] = 0.0f;
            parkour_keyboard_command[2] = 0.0f;
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
    if (param::keyboard_control)
    {
        return keyboard_velocity_command(env);
    }

    std::vector<float> obs(3);
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

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
