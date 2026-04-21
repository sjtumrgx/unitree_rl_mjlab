#include "SupportGeometryProvider.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <spdlog/spdlog.h>
#include <unitree/idl/ros2/PointField_.hpp>

#include "isaaclab/assets/articulation/articulation.h"

SupportGeometryProvider::SupportGeometryProvider(const YAML::Node& cfg)
{
    const auto support_geometry = cfg["support_geometry_interface"];
    if (!support_geometry || !support_geometry["depth_camera"]) {
        return;
    }

    enabled_ = true;
    const auto depth_camera = support_geometry["depth_camera"];
    sensor_name_ = depth_camera["sensor_name"].as<std::string>(sensor_name_);
    topic_name_ = depth_camera["topic_name"].as<std::string>("");
    pointcloud_mode_ = depth_camera["pointcloud_mode"].as<std::string>(pointcloud_mode_);
    cutoff_distance_ = depth_camera["cutoff_distance"].as<float>(cutoff_distance_);

    const auto camera_group = cfg["observations"]["camera"];
    if (camera_group && camera_group[sensor_name_] && camera_group[sensor_name_]["params"]) {
        expected_size_ = camera_group[sensor_name_]["params"]["expected_size"].as<int>(0);
    }

    if (!topic_name_.empty()) {
        pointcloud_sub_ = std::make_shared<PointCloudSub>(topic_name_);
    }
}

void SupportGeometryProvider::initialize(isaaclab::Articulation* robot)
{
    if (!enabled_ || robot == nullptr || expected_size_ <= 0) {
        return;
    }

    auto& buffer = robot->data.named_observations[sensor_name_];
    buffer.assign(expected_size_, 0.0f);
}

void SupportGeometryProvider::update(isaaclab::Articulation* robot)
{
    if (!enabled_ || robot == nullptr || expected_size_ <= 0) {
        return;
    }

    auto& buffer = robot->data.named_observations[sensor_name_];
    if (static_cast<int>(buffer.size()) != expected_size_) {
        buffer.assign(expected_size_, 0.0f);
    }

    if (pointcloud_sub_ && !pointcloud_sub_->isTimeout()) {
        fill_from_pointcloud(buffer);
        warned_missing_feed_ = false;
        return;
    }

    std::fill(buffer.begin(), buffer.end(), 0.0f);
    if (!warned_missing_feed_) {
        warned_missing_feed_ = true;
        if (topic_name_.empty()) {
            spdlog::warn(
                "SupportGeometryProvider '{}' has no topic configured; using zero-filled SGI buffer.",
                sensor_name_
            );
        } else {
            spdlog::warn(
                "SupportGeometryProvider '{}' timed out waiting for topic '{}'; using zero-filled SGI buffer.",
                sensor_name_, topic_name_
            );
        }
    }
}

void SupportGeometryProvider::fill_from_pointcloud(std::vector<float>& buffer) const
{
    if (!pointcloud_sub_) {
        return;
    }

    std::lock_guard<std::mutex> lock(pointcloud_sub_->mutex_);
    const auto& msg = pointcloud_sub_->msg_;
    if (msg.point_step() == 0 || msg.data().empty()) {
        std::fill(buffer.begin(), buffer.end(), 0.0f);
        return;
    }

    const int x_offset = find_float32_field_offset(msg, "x");
    const int y_offset = find_float32_field_offset(msg, "y");
    const int z_offset = find_float32_field_offset(msg, "z");
    if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
        std::fill(buffer.begin(), buffer.end(), 0.0f);
        return;
    }

    const auto& data = msg.data();
    const size_t point_step = static_cast<size_t>(msg.point_step());
    const size_t max_points = std::min(buffer.size(), data.size() / point_step);

    for (size_t idx = 0; idx < max_points; ++idx) {
        const size_t base = idx * point_step;
        const float x = read_float32(data, base + static_cast<size_t>(x_offset));
        const float y = read_float32(data, base + static_cast<size_t>(y_offset));
        const float z = read_float32(data, base + static_cast<size_t>(z_offset));

        float value = 0.0f;
        if (pointcloud_mode_ == "z_depth") {
            value = z;
        } else {
            value = std::sqrt(x * x + y * y + z * z);
        }
        value = std::clamp(value, 0.0f, cutoff_distance_);
        buffer[idx] = cutoff_distance_ > 0.0f ? (value / cutoff_distance_) : value;
    }
    for (size_t idx = max_points; idx < buffer.size(); ++idx) {
        buffer[idx] = 0.0f;
    }
}

int SupportGeometryProvider::find_float32_field_offset(const PointCloudMsg& msg, const std::string& field_name)
{
    for (const auto& field : msg.fields()) {
        if (field.name() == field_name
            && field.datatype() == sensor_msgs::msg::dds_::PointField_Constants::FLOAT32_) {
            return static_cast<int>(field.offset());
        }
    }
    return -1;
}

float SupportGeometryProvider::read_float32(const std::vector<uint8_t>& data, size_t offset)
{
    if (offset + sizeof(float) > data.size()) {
        return 0.0f;
    }
    float value = 0.0f;
    std::memcpy(&value, data.data() + offset, sizeof(float));
    return value;
}
