#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <unitree/dds_wrapper/common/Subscription.h>
#include <unitree/idl/ros2/PointCloud2_.hpp>

namespace isaaclab {
class Articulation;
}

class ParkourDepthProvider
{
public:
    explicit ParkourDepthProvider(const YAML::Node& cfg);

    void initialize(isaaclab::Articulation* robot);
    void reset(isaaclab::Articulation* robot);
    void update(isaaclab::Articulation* robot);

    [[nodiscard]] bool enabled() const { return enabled_; }
    [[nodiscard]] bool has_valid_frame() const { return has_valid_frame_; }
    [[nodiscard]] const std::string& sensor_name() const { return sensor_name_; }
    [[nodiscard]] int expected_size() const { return expected_size_; }

private:
    using PointCloudMsg = sensor_msgs::msg::dds_::PointCloud2_;
    using PointCloudSub = unitree::robot::SubscriptionBase<PointCloudMsg>;

    bool fill_frame_from_pointcloud(std::vector<float>& frame) const;
    void append_frame(std::vector<float> frame);
    std::vector<float> compose_history_stack() const;
    void apply_gaussian_blur(std::vector<float>& frame) const;
    static int find_float32_field_offset(const PointCloudMsg& msg, const std::string& field_name);
    static float read_float32(const std::vector<uint8_t>& data, size_t offset, bool is_bigendian);

    bool enabled_ = false;
    bool warned_missing_feed_ = false;
    bool retain_last_valid_frame_ = true;
    bool organized_pointcloud_ = true;
    bool has_valid_frame_ = false;
    std::string sensor_name_ = "depth_image";
    std::string topic_name_;
    std::string pointcloud_mode_ = "z_depth";
    std::unordered_map<std::string, std::string> pointcloud_field_names_{{"x", "x"}, {"y", "y"}, {"z", "z"}};
    int raw_width_ = 64;
    int raw_height_ = 36;
    int crop_top_ = 0;
    int crop_bottom_ = 0;
    int crop_left_ = 0;
    int crop_right_ = 0;
    int output_width_ = 64;
    int output_height_ = 36;
    int history_source_length_ = 37;
    int history_skip_frames_ = 5;
    int num_output_frames_ = 8;
    int expected_size_ = 0;
    int timeout_ms_ = 500;
    int gaussian_kernel_size_ = 3;
    float gaussian_sigma_ = 1.0f;
    float depth_min_ = 0.0f;
    float depth_max_ = 2.5f;
    float output_min_ = 0.0f;
    float output_max_ = 1.0f;
    std::shared_ptr<PointCloudSub> pointcloud_sub_;
    std::deque<std::vector<float>> history_frames_;
};
