#pragma once

#include <iostream>
#include <atomic>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace param
{

inline struct SimulationConfig
{
    std::string robot;
    std::filesystem::path robot_scene;

    int domain_id;
    std::string interface;

    int use_joystick;
    std::string joystick_type;
    std::string joystick_device;
    int joystick_bits;

    int print_scene_information;
    int wait_for_lowcmd_before_physics = 0;
    int headless = 0;
    float headless_seconds = 90.0f;
    float walk_distance_marker = 5.0f;
    float progress_log_interval = 1.0f;
    int realtime_lockstep = 0;
    float lowcmd_kp_scale = 1.0f;
    float lowcmd_kd_scale = 1.0f;
    std::string lowcmd_control_mode = "torque-pd";

    int enable_elastic_band;
    int band_attached_link = 0;
    std::vector<float> initial_base_pos;
    std::vector<float> initial_base_quat;
    std::vector<float> initial_joint_pos;

    int enable_depth_camera = 0;
    std::string depth_camera_name = "parkour_depth_camera";
    std::string depth_pointcloud_topic = "rt/parkour_depth/points";
    int depth_camera_width = 64;
    int depth_camera_height = 36;
    int depth_window_scale = 10;
    float depth_max_distance = 2.5f;
    float depth_camera_min_distance = 0.1f;
    std::string depth_camera_ray_alignment = "base";
    int depth_debug_crop_top = 0;
    int depth_debug_crop_left = 0;
    int depth_debug_crop_width = 64;
    int depth_debug_crop_height = 36;
    int depth_publish_period_ms = 100;

    void load_from_yaml(const std::string &filename)
    {
        auto cfg = YAML::LoadFile(filename);
        try
        {
            robot = cfg["robot"].as<std::string>();
            robot_scene = cfg["robot_scene"].as<std::string>();
            domain_id = cfg["domain_id"].as<int>();
            interface = cfg["interface"].as<std::string>();
            use_joystick = cfg["use_joystick"].as<int>();
            joystick_type = cfg["joystick_type"].as<std::string>();
            joystick_device = cfg["joystick_device"].as<std::string>();
            joystick_bits = cfg["joystick_bits"].as<int>();
            print_scene_information = cfg["print_scene_information"].as<int>();
            if (cfg["wait_for_lowcmd_before_physics"]) {
                wait_for_lowcmd_before_physics = cfg["wait_for_lowcmd_before_physics"].as<int>();
            }
            if (cfg["realtime_lockstep"]) {
                realtime_lockstep = cfg["realtime_lockstep"].as<int>();
            }
            enable_elastic_band = cfg["enable_elastic_band"].as<int>();
            if (cfg["initial_base_pos"]) {
                initial_base_pos = cfg["initial_base_pos"].as<std::vector<float>>();
            }
            if (cfg["initial_base_quat"]) {
                initial_base_quat = cfg["initial_base_quat"].as<std::vector<float>>();
            }
            if (cfg["initial_joint_pos"]) {
                initial_joint_pos = cfg["initial_joint_pos"].as<std::vector<float>>();
            }
            if (cfg["band_attached_link"]) {
                band_attached_link = cfg["band_attached_link"].as<int>();
            }
            if (cfg["enable_depth_camera"]) {
                enable_depth_camera = cfg["enable_depth_camera"].as<int>();
            }
            if (cfg["depth_camera_name"]) {
                depth_camera_name = cfg["depth_camera_name"].as<std::string>();
            }
            if (cfg["depth_pointcloud_topic"]) {
                depth_pointcloud_topic = cfg["depth_pointcloud_topic"].as<std::string>();
            }
            if (cfg["depth_camera_width"]) {
                depth_camera_width = cfg["depth_camera_width"].as<int>();
            }
            if (cfg["depth_camera_height"]) {
                depth_camera_height = cfg["depth_camera_height"].as<int>();
            }
            if (cfg["depth_window_scale"]) {
                depth_window_scale = cfg["depth_window_scale"].as<int>();
            }
            if (cfg["depth_max_distance"]) {
                depth_max_distance = cfg["depth_max_distance"].as<float>();
            }
            if (cfg["depth_camera_min_distance"]) {
                depth_camera_min_distance = cfg["depth_camera_min_distance"].as<float>();
            }
            if (cfg["depth_camera_ray_alignment"]) {
                depth_camera_ray_alignment = cfg["depth_camera_ray_alignment"].as<std::string>();
            }
            if (cfg["depth_debug_crop_top"]) {
                depth_debug_crop_top = cfg["depth_debug_crop_top"].as<int>();
            }
            if (cfg["depth_debug_crop_left"]) {
                depth_debug_crop_left = cfg["depth_debug_crop_left"].as<int>();
            }
            if (cfg["depth_debug_crop_width"]) {
                depth_debug_crop_width = cfg["depth_debug_crop_width"].as<int>();
            }
            if (cfg["depth_debug_crop_height"]) {
                depth_debug_crop_height = cfg["depth_debug_crop_height"].as<int>();
            }
            if (cfg["depth_publish_period_ms"]) {
                depth_publish_period_ms = cfg["depth_publish_period_ms"].as<int>();
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }
} config;

inline std::atomic<bool> lowcmd_connected{false};
inline std::atomic<bool> lowcmd_has_active_control{false};

/* ---------- Command Line Parameters ---------- */
namespace po = boost::program_options;

//※ This function must be called at the beginning of main() function
inline po::variables_map helper(int argc, char** argv)
{
    po::options_description desc("Unitree Mujoco");
    desc.add_options()
        ("help,h", "Show help message")
        ("domain_id,i", po::value<int>(&config.domain_id), "DDS domain ID; -i 0")
        ("network,n", po::value<std::string>(&config.interface), "DDS network interface; -n eth0")
        ("robot,r", po::value<std::string>(&config.robot), "Robot type; -r go2")
        ("scene,s", po::value<std::filesystem::path>(&config.robot_scene), "Robot scene file; -s scene_terrain.xml")
        ("headless", po::bool_switch()->default_value(false), "Run physics/DDS loop without MuJoCo viewer")
        ("headless-seconds", po::value<float>(&config.headless_seconds)->default_value(config.headless_seconds), "Headless run timeout in seconds")
        ("walk-distance-marker", po::value<float>(&config.walk_distance_marker)->default_value(config.walk_distance_marker), "Emit DISTANCE_X marker and stop headless run after this forward distance in meters")
        ("progress-log-interval", po::value<float>(&config.progress_log_interval)->default_value(config.progress_log_interval), "Emit PARKOUR_PROGRESS every N simulation seconds")
        ("realtime-lockstep", po::value<int>(&config.realtime_lockstep)->default_value(config.realtime_lockstep), "Run viewer physics at one MuJoCo step per wall-clock timestep instead of batching catch-up steps")
        ("lowcmd-kp-scale", po::value<float>(&config.lowcmd_kp_scale)->default_value(config.lowcmd_kp_scale), "Diagnostic scale for Unitree LowCmd position gain in the MuJoCo bridge")
        ("lowcmd-kd-scale", po::value<float>(&config.lowcmd_kd_scale)->default_value(config.lowcmd_kd_scale), "Diagnostic scale for Unitree LowCmd velocity gain in the MuJoCo bridge")
        ("lowcmd-control-mode", po::value<std::string>(&config.lowcmd_control_mode)->default_value(config.lowcmd_control_mode), "Diagnostic LowCmd bridge mode: torque-pd or position-target")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(0);
    }
    if (vm["headless"].as<bool>()) {
        config.headless = 1;
    }
    if (config.lowcmd_control_mode != "torque-pd" && config.lowcmd_control_mode != "position-target") {
        std::cerr << "--lowcmd-control-mode must be torque-pd or position-target" << std::endl;
        exit(EXIT_FAILURE);
    }

    return vm;
}

}
