#include "ParkourObservations.h"
#include "isaaclab/envs/mdp/observations/observations.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <unitree/dds_wrapper/robots/go2/go2.h>

namespace isaaclab
{
namespace mdp
{
namespace
{

using RoutePoseSub = unitree::robot::go2::subscription::SportModeState;

const std::vector<std::pair<float, float>>& parkour_route_waypoints()
{
    static const std::vector<std::pair<float, float>> waypoints{
        {0.0f, 0.0f},
        {2.0f, 0.0f},
        {4.8f, 0.0f},
        {7.345f, 0.0f},
        {10.8f, 0.0f},
        {13.2f, 0.0f},
        {15.5f, 0.0f},
        {17.8f, 0.0f},
        {19.645f, 0.0f},
        {22.0f, 0.0f},
        {25.2f, 0.0f},
    };
    return waypoints;
}

std::optional<std::pair<float, float>> latest_route_pose_xy()
{
    static std::shared_ptr<RoutePoseSub> sport_state_sub = [] {
        auto sub = std::make_shared<RoutePoseSub>("rt/sportmodestate");
        sub->set_timeout_ms(500);
        return sub;
    }();
    static bool warned_timeout = false;

    if (!sport_state_sub || sport_state_sub->isTimeout()) {
        if (!warned_timeout) {
            warned_timeout = true;
            spdlog::warn(
                "SIM_ROUTE_FOLLOW waiting for rt/sportmodestate pose; falling back to fixed sim command."
            );
        }
        return std::nullopt;
    }
    warned_timeout = false;

    std::lock_guard<std::mutex> lock(sport_state_sub->mutex_);
    const auto position = sport_state_sub->position();
    return std::make_pair(position[0], position[1]);
}

size_t target_route_index(float x)
{
    const auto& waypoints = parkour_route_waypoints();
    const float target_x = x + param::sim_route_lookahead;
    for (size_t index = 1; index < waypoints.size(); ++index) {
        if (waypoints[index].first >= target_x) {
            return index;
        }
    }
    return waypoints.size() - 1;
}

std::vector<float> parkour_route_velocity_commands(ManagerBasedRLEnv* env)
{
    const auto pose = latest_route_pose_xy();
    if (!pose || env == nullptr || env->robot == nullptr) {
        return velocity_commands(env, YAML::Node{});
    }

    const auto& waypoints = parkour_route_waypoints();
    const float pos_x = pose->first;
    const float pos_y = pose->second;
    const size_t target_index = target_route_index(pos_x);
    const auto [target_x, target_y] = waypoints[target_index];
    const float final_x = waypoints.back().first;
    const bool route_completed = target_index == waypoints.size() - 1 && pos_x >= final_x;

    if (route_completed) {
        static bool logged_completed = false;
        if (!logged_completed) {
            std::cout << "SIM_ROUTE_COMPLETED x=" << pos_x
                      << " y=" << pos_y
                      << " final_x=" << final_x
                      << std::endl;
            logged_completed = true;
        }
        param::sim_observed_command_x.store(0.0f);
        param::sim_observed_command_y.store(0.0f);
        param::sim_observed_command_yaw.store(0.0f);
        return {0.0f, 0.0f, 0.0f};
    }

    const float yaw = detail::yaw_from_quaternion(env->robot->data.root_quat_w);
    const float speed = std::max(0.05f, param::sim_command_x);
    const float delta_x = target_x - pos_x;
    const float delta_y = target_y - pos_y;
    const float distance = std::max(1.0e-6f, std::hypot(delta_x, delta_y));
    float desired_heading = std::atan2(delta_y, delta_x);
    float yaw_error = detail::wrap_to_pi(desired_heading - yaw);
    float desired_world_x = speed * delta_x / distance;
    float desired_world_y = speed * delta_y / distance;
    const float cos_yaw = std::cos(yaw);
    const float sin_yaw = std::sin(yaw);
    const float body_x = cos_yaw * desired_world_x + sin_yaw * desired_world_y;
    const float body_y = -sin_yaw * desired_world_x + cos_yaw * desired_world_y;

    std::vector<float> obs{
        std::clamp(body_x, 0.05f, speed),
        std::clamp(body_y, -param::sim_route_max_lateral_speed, param::sim_route_max_lateral_speed),
        std::clamp(
            param::sim_route_yaw_gain * yaw_error + param::sim_command_yaw,
            -param::sim_route_max_yaw_rate,
            param::sim_route_max_yaw_rate
        ),
    };

    param::sim_observed_command_x.store(obs[0]);
    param::sim_observed_command_y.store(obs[1]);
    param::sim_observed_command_yaw.store(obs[2]);

    static bool logged_nonzero_command = false;
    if (!logged_nonzero_command) {
        std::cout << "NONZERO_COMMAND x=" << obs[0]
                  << " y=" << obs[1]
                  << " yaw=" << obs[2]
                  << " source=sim-route-follow"
                  << " route_speed=" << speed
                  << " lookahead=" << param::sim_route_lookahead
                  << std::endl;
        logged_nonzero_command = true;
    }

    static size_t route_log_count = 0;
    if (route_log_count++ % 250 == 0) {
        std::cout << "SIM_ROUTE_FOLLOW x=" << pos_x
                  << " y=" << pos_y
                  << " yaw=" << yaw
                  << " target_index=" << target_index
                  << " target_x=" << target_x
                  << " target_y=" << target_y
                  << " distance=" << distance
                  << " command_x=" << obs[0]
                  << " command_y=" << obs[1]
                  << " command_yaw=" << obs[2]
                  << std::endl;
    }

    return obs;
}

std::vector<float> parkour_velocity_commands(ManagerBasedRLEnv* env, YAML::Node params)
{
    if (param::sim_autostart_parkour && param::sim_route_follow) {
        return parkour_route_velocity_commands(env);
    }
    return velocity_commands(env, params);
}

} // namespace

REGISTER_OBSERVATION(depth_image)
{
    const auto sensor_name = params["sensor_name"].as<std::string>("depth_image");
    const auto expected_size = params["expected_size"].as<int>(0);
    const auto it = env->robot->data.named_observations.find(sensor_name);
    if (it != env->robot->data.named_observations.end()) {
        return it->second;
    }
    if (expected_size <= 0) {
        return {};
    }
    return std::vector<float>(expected_size, 0.0f);
}

} // namespace mdp
} // namespace isaaclab

void ensureParkourObservationRegistration()
{
    isaaclab::observations_map()["velocity_commands"] = isaaclab::mdp::parkour_velocity_commands;
    (void)isaaclab::observations_map();
}
