// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <stdint.h>
#include <atomic>
#include <chrono>
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
inline float sim_command_x = 0.25f;
inline float sim_command_y = 0.0f;
inline float sim_command_yaw = 0.0f;
inline bool sim_heading_lock = true;
inline float sim_heading_target_yaw = 0.0f;
inline float sim_heading_kp = 1.0f;
inline float sim_heading_max_yaw = 0.8f;
inline std::atomic<float> sim_observed_command_x{0.0f};
inline std::atomic<float> sim_observed_command_y{0.0f};
inline std::atomic<float> sim_observed_command_yaw{0.0f};
inline std::filesystem::path gait_record_jsonl;
inline int gait_record_every = 1;

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
        ("sim-command-x", po::value<float>(&sim_command_x)->default_value(0.25f),
            "simulation-only forward velocity command used with --sim-autostart-parkour")
        ("sim-command-y", po::value<float>(&sim_command_y)->default_value(0.0f),
            "simulation-only lateral velocity command used with --sim-autostart-parkour")
        ("sim-command-yaw", po::value<float>(&sim_command_yaw)->default_value(0.0f),
            "simulation-only yaw velocity command used with --sim-autostart-parkour")
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
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    keyboard_control = vm.count("keyboard") > 0;
    if (vm["no-sim-heading-lock"].as<bool>())
    {
        sim_heading_lock = false;
    }
    if (gait_record_every < 1)
    {
        gait_record_every = 1;
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
