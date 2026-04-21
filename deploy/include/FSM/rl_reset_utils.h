// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace rl_reset
{

inline bool tick_rewound(uint32_t previous_tick, uint32_t current_tick)
{
    return previous_tick != 0 && current_tick < previous_tick;
}

inline std::vector<float> reset_processed_actions(
    int action_dim,
    const std::vector<float>& offset,
    const std::vector<std::vector<float>>& clip)
{
    std::vector<float> processed_actions(action_dim, 0.0f);
    for (int i = 0; i < action_dim; ++i)
    {
        if (!offset.empty())
        {
            processed_actions[i] += offset[i];
        }
        if (!clip.empty())
        {
            processed_actions[i] = std::clamp(processed_actions[i], clip[i][0], clip[i][1]);
        }
    }
    return processed_actions;
}

} // namespace rl_reset
