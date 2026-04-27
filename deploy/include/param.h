// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <stdint.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <memory>
#include <iomanip>
#include "isaaclab/devices/keyboard/keyboard.h"

/* ---------- logger ---------- */
namespace spdlog
{
inline void create_logger(std::string log_path)
{
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(log_path, 5 * 1024 * 1024, 5);

    std::vector<spdlog::sink_ptr> sinks {console_sink, rotating_sink};
    auto logger = std::make_shared<spdlog::logger>("unitree", sinks.begin(), sinks.end());

    logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");
    logger->flush_on(spdlog::level::info);

    spdlog::set_default_logger(logger);
}

} // namespace spdlog


namespace param
{
inline std::string VERSION = "1.0.0.1";
inline std::filesystem::path bin_path;
inline std::filesystem::path proj_dir;
inline std::filesystem::path config_dir;
inline YAML::Node config;
inline bool keyboard_control = false;
inline std::shared_ptr<Keyboard> active_keyboard = nullptr;
inline bool sim_autostart_parkour = false;
inline bool sim_loopback_interactive = false;
inline float sim_command_x = 0.30f;
inline float sim_keyboard_idle_command_x = -0.15f;
inline float sim_command_y = 0.0f;
inline float sim_command_yaw = 0.0f;
inline bool sim_route_follow = true;
inline float sim_route_lookahead = 1.0f;
inline float sim_route_max_lateral_speed = 0.25f;
inline float sim_route_max_yaw_rate = 0.8f;
inline float sim_route_yaw_gain = 1.0f;
inline bool sim_heading_lock = true;
inline float sim_heading_target_yaw = 0.0f;
inline float sim_heading_kp = 1.0f;
inline float sim_heading_max_yaw = 0.8f;
inline std::atomic<float> sim_observed_command_x{0.0f};
inline std::atomic<float> sim_observed_command_y{0.0f};
inline std::atomic<float> sim_observed_command_yaw{0.0f};
inline std::filesystem::path gait_record_jsonl;
inline int gait_record_every = 1;
inline std::filesystem::path gait_replay_jsonl;
inline std::string gait_replay_mode = "off";
inline int gait_replay_start_step = 0;
inline int gait_replay_max_steps = 0;
inline std::string joint_vel_source = "sensor";
inline bool policy_tick_sync = false;
inline bool no_policy_tick_sync = false;
inline bool parkour_live_depth_blend_override = false;
inline float parkour_live_depth_blend = 0.0f;
inline bool parkour_live_depth_baseline_override = false;
inline float parkour_live_depth_baseline = 0.5f;
inline bool parkour_constant_depth_override = false;
inline float parkour_constant_depth = 0.5f;
inline bool parkour_depth_artifact_floor_override = false;
inline float parkour_depth_artifact_floor = 0.0f;

inline std::filesystem::path get_bin_path() {
    std::vector<char> path(1024);
    ssize_t len = readlink("/proc/self/exe", &path[0], path.size());
    if (len != -1) {
        path[len] = '\0';  // Null-terminate the result
        return std::filesystem::path(&path[0]);
    } else {
        spdlog::error("Failed to get executable path.");
        exit(1);
    }
}

/* ---------- config.yaml ---------- */
inline void load_config_file()
{
    assert(std::filesystem::exists(bin_path)); // run param::helper before this function
    if(bin_path.parent_path().filename() == "bin" || bin_path.parent_path().filename() == "build")
    {
        proj_dir = bin_path.parent_path().parent_path();
        config_dir = proj_dir / "config";
    }
    else
    {
        proj_dir = bin_path.parent_path();
        config_dir = proj_dir;
    }

    try {
        std::string config_file = (config_dir / "config.yaml").string();
        if(std::filesystem::exists(config_file))
        {
            config = YAML::LoadFile(config_file);
        }
    } catch (const YAML::BadFile& e) {
        spdlog::error("Failed to load config.yaml: {}", e.what());
        exit(1);
    }
}

inline std::filesystem::path parser_policy_dir(std::filesystem::path policy_dir)
{
    // Load Policy
    if (policy_dir.is_relative()) {
        policy_dir = param::proj_dir / policy_dir;
    }

    // If there is no `exported` folder in this folder,
    // then sort all the folders under this folder and take the last folder
    if (!std::filesystem::exists(policy_dir / "exported")) {
        auto dirs = std::filesystem::directory_iterator(policy_dir);
        std::vector<std::filesystem::path> dir_list;
        for (const auto& entry : dirs) {
            if (entry.is_directory()) {
                dir_list.push_back(entry.path());
            }
        }
        if (!dir_list.empty()) {
            std::sort(dir_list.begin(), dir_list.end());
            // Check if there is an `exported` folder starting from the last folder
            for (auto it = dir_list.rbegin(); it != dir_list.rend(); ++it) {
                if (std::filesystem::exists(*it / "exported")) {
                    policy_dir = *it;
                    break;
                }
            }
        }
    }
    spdlog::info("Policy directory: {}", policy_dir.string());
    return policy_dir;
}

/* ---------- Command Line Parameters ---------- */
namespace po = boost::program_options;

//※ This function must be called at the beginning of main() function
inline po::variables_map helper(int argc, char** argv) 
{
    bin_path = get_bin_path();
    load_config_file();

    po::options_description desc("Unitree Controller");
    desc.add_options()
        ("help,h", "produce help message")
        ("version,v", "show version")
        ("log", "record log file")
        ("network,n", po::value<std::string>()->default_value(""), "dds network interface")
        ("keyboard,k", "enable keyboard control")
        ("sim-autostart-parkour", po::bool_switch(&sim_autostart_parkour)->default_value(false),
            "simulation-only: start directly in Parkour mode; requires --network=lo")
        ("no-sim-loopback-interactive", po::bool_switch()->default_value(false),
            "simulation-only: opt out of the default safe interactive keyboard mode used by --network=lo")
        ("no-sim-autostart-parkour", po::bool_switch()->default_value(false),
            "simulation-only: legacy alias for --no-sim-loopback-interactive")
        ("sim-command-x", po::value<float>(&sim_command_x)->default_value(0.30f),
            "simulation-only forward velocity command used with --sim-autostart-parkour")
        ("sim-idle-command-x", po::value<float>(&sim_keyboard_idle_command_x)->default_value(-0.15f),
            "simulation-only keyboard idle/stop forward command used by default loopback interactive mode")
        ("sim-command-y", po::value<float>(&sim_command_y)->default_value(0.0f),
            "simulation-only lateral velocity command used with --sim-autostart-parkour")
        ("sim-command-yaw", po::value<float>(&sim_command_yaw)->default_value(0.0f),
            "simulation-only yaw velocity command used with --sim-autostart-parkour")
        ("no-sim-route-follow", po::bool_switch()->default_value(false),
            "simulation-only: disable terrain-route following and use fixed sim command")
        ("sim-route-lookahead", po::value<float>(&sim_route_lookahead)->default_value(1.0f),
            "simulation-only route follower lookahead distance in meters")
        ("sim-route-max-lateral-speed", po::value<float>(&sim_route_max_lateral_speed)->default_value(0.25f),
            "simulation-only route follower body-frame lateral velocity clamp")
        ("sim-route-max-yaw-rate", po::value<float>(&sim_route_max_yaw_rate)->default_value(0.8f),
            "simulation-only route follower yaw-rate clamp")
        ("sim-route-yaw-gain", po::value<float>(&sim_route_yaw_gain)->default_value(1.0f),
            "simulation-only route follower heading proportional gain")
        ("sim-heading-lock", po::bool_switch(&sim_heading_lock)->default_value(true),
            "simulation-only: stabilize heading to --sim-heading-target-yaw while walking")
        ("no-sim-heading-lock", po::bool_switch()->default_value(false),
            "simulation-only: disable heading stabilization")
        ("sim-heading-target-yaw", po::value<float>(&sim_heading_target_yaw)->default_value(0.0f),
            "simulation-only heading target in radians used by --sim-heading-lock")
        ("sim-heading-kp", po::value<float>(&sim_heading_kp)->default_value(1.0f),
            "simulation-only heading proportional gain")
        ("sim-heading-max-yaw", po::value<float>(&sim_heading_max_yaw)->default_value(0.8f),
            "simulation-only absolute heading correction clamp")
        ("gait-record-jsonl", po::value<std::filesystem::path>(&gait_record_jsonl),
            "record per-policy-step parkour gait/action/joint samples as JSONL")
        ("gait-record-every", po::value<int>(&gait_record_every)->default_value(1),
            "record one gait sample every N policy steps when --gait-record-jsonl is set")
        ("gait-replay-jsonl", po::value<std::filesystem::path>(&gait_replay_jsonl),
            "simulation-only diagnostic: replay Python gait JSONL action/target sequence instead of actor output")
        ("gait-replay-mode", po::value<std::string>(&gait_replay_mode)->default_value("off"),
            "simulation-only diagnostic replay mode: off, raw-action-policy, or target-q-deploy")
        ("gait-replay-start-step", po::value<int>(&gait_replay_start_step)->default_value(0),
            "simulation-only diagnostic: first source sample index to replay from --gait-replay-jsonl")
        ("gait-replay-max-steps", po::value<int>(&gait_replay_max_steps)->default_value(0),
            "simulation-only diagnostic: maximum replayed policy steps; 0 means all available samples")
        ("joint-vel-source", po::value<std::string>(&joint_vel_source)->default_value("sensor"),
            "simulation-only diagnostic joint velocity source: sensor, finite-diff-policy, or finite-diff-lowstate")
        ("policy-tick-sync", po::bool_switch(&policy_tick_sync)->default_value(false),
            "simulation-only diagnostic: gate the 50 Hz policy loop on simulator lowstate tick instead of wall-clock only")
        ("no-policy-tick-sync", po::bool_switch(&no_policy_tick_sync)->default_value(false),
            "simulation-only diagnostic: disable the default lowstate-tick-synced policy loop used by --sim-autostart-parkour")
        ("live-depth-blend", po::value<float>(&parkour_live_depth_blend),
            "parkour simulation-only: blend live depth into policy input, 0=constant baseline, 1=full live depth")
        ("live-depth-baseline", po::value<float>(&parkour_live_depth_baseline),
            "parkour simulation-only: normalized baseline depth used when --live-depth-blend is below 1")
        ("constant-depth", po::value<float>(&parkour_constant_depth),
            "parkour diagnostic: ignore live depth and feed a constant normalized policy-depth value")
        ("depth-artifact-floor", po::value<float>(&parkour_depth_artifact_floor),
            "parkour live-depth diagnostic: clamp normalized depth pixels below this floor")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    keyboard_control = vm.count("keyboard") > 0;
    const std::string network = vm["network"].as<std::string>();
    const bool explicit_sim_autostart = vm["sim-autostart-parkour"].as<bool>();
    const bool disable_loopback_interactive =
        vm["no-sim-loopback-interactive"].as<bool>() || vm["no-sim-autostart-parkour"].as<bool>();
    if (network == "lo" && !explicit_sim_autostart && !disable_loopback_interactive)
    {
        sim_loopback_interactive = true;
        keyboard_control = true;
        spdlog::warn(
            "--network=lo defaulting to interactive simulation mode: "
            "start_state=Parkour with idle hold, live depth, keyboard-gated terrain route following, "
            "cruise speed={} m/s. Pass --no-sim-route-follow for fixed keyboard velocity commands. "
            "Pass --no-sim-loopback-interactive for the legacy joystick/FSM flow.",
            sim_command_x
        );
    }
    if (vm["no-sim-heading-lock"].as<bool>())
    {
        sim_heading_lock = false;
    }
    if (vm["no-sim-route-follow"].as<bool>())
    {
        sim_route_follow = false;
    }
    sim_route_lookahead = std::max(0.05f, sim_route_lookahead);
    sim_route_max_lateral_speed = std::abs(sim_route_max_lateral_speed);
    sim_route_max_yaw_rate = std::abs(sim_route_max_yaw_rate);
    if (gait_record_every < 1)
    {
        gait_record_every = 1;
    }
    if (gait_replay_start_step < 0)
    {
        gait_replay_start_step = 0;
    }
    if (gait_replay_max_steps < 0)
    {
        gait_replay_max_steps = 0;
    }
    if (gait_replay_jsonl.empty())
    {
        gait_replay_mode = "off";
    }
    if (gait_replay_mode != "off"
        && gait_replay_mode != "raw-action-policy"
        && gait_replay_mode != "target-q-deploy")
    {
        throw std::runtime_error("--gait-replay-mode must be off, raw-action-policy, or target-q-deploy");
    }
    if (joint_vel_source != "sensor"
        && joint_vel_source != "finite-diff-policy"
        && joint_vel_source != "finite-diff-lowstate")
    {
        throw std::runtime_error("--joint-vel-source must be sensor, finite-diff-policy, or finite-diff-lowstate");
    }
    parkour_live_depth_blend_override = vm.count("live-depth-blend") > 0;
    parkour_live_depth_baseline_override = vm.count("live-depth-baseline") > 0;
    parkour_constant_depth_override = vm.count("constant-depth") > 0;
    parkour_depth_artifact_floor_override = vm.count("depth-artifact-floor") > 0;
    if (parkour_live_depth_blend_override)
    {
        parkour_live_depth_blend = std::clamp(parkour_live_depth_blend, 0.0f, 1.0f);
    }
    if (parkour_live_depth_baseline_override)
    {
        parkour_live_depth_baseline = std::clamp(parkour_live_depth_baseline, 0.0f, 1.0f);
    }
    if (parkour_constant_depth_override)
    {
        parkour_constant_depth = std::clamp(parkour_constant_depth, 0.0f, 1.0f);
    }
    if (parkour_depth_artifact_floor_override)
    {
        parkour_depth_artifact_floor = std::clamp(parkour_depth_artifact_floor, 0.0f, 1.0f);
    }
    if (sim_loopback_interactive && !parkour_live_depth_blend_override && !parkour_constant_depth_override)
    {
        parkour_live_depth_blend_override = true;
        parkour_live_depth_blend = 1.0f;
    }
    if (sim_autostart_parkour && !parkour_live_depth_blend_override && !parkour_constant_depth_override)
    {
        parkour_live_depth_blend_override = true;
        parkour_live_depth_blend = 1.0f;
        spdlog::warn(
            "--sim-autostart-parkour defaulting to --live-depth-blend=1.0; "
            "pass --live-depth-blend=0.0 or --constant-depth=<value> for proprioception-only diagnostics."
        );
    }
    if ((sim_autostart_parkour || sim_loopback_interactive) && !no_policy_tick_sync)
    {
        policy_tick_sync = true;
    }
    if (no_policy_tick_sync)
    {
        policy_tick_sync = false;
    }

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(0);
    }
    if (vm.count("version"))
    {
        std::cout << "Version: " << VERSION << std::endl;
        exit(0);
    }

#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#else
    spdlog::set_level(spdlog::level::info);
#endif
    if(vm.count("log"))
    {
        std::filesystem::create_directories(proj_dir / "log");
        spdlog::create_logger(proj_dir.string() + "/log/log.txt");
    }

    return vm;
}

}
