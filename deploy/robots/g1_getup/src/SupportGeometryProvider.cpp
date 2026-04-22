#include "SupportGeometryProvider.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>

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
    timeout_ms_ = depth_camera["timeout_ms"].as<int>(timeout_ms_);
    retain_last_valid_frame_ =
        depth_camera["retain_last_valid_frame"].as<bool>(retain_last_valid_frame_);
    organized_pointcloud_ = depth_camera["organized_pointcloud"].as<bool>(organized_pointcloud_);
    if (const auto field_names = depth_camera["pointcloud_field_names"])
    {
        pointcloud_field_names_["x"] = field_names["x"].as<std::string>(pointcloud_field_names_["x"]);
        pointcloud_field_names_["y"] = field_names["y"].as<std::string>(pointcloud_field_names_["y"]);
        pointcloud_field_names_["z"] = field_names["z"].as<std::string>(pointcloud_field_names_["z"]);
    }

    if (support_geometry["patch_shape"] && support_geometry["patch_shape"].IsSequence()
        && support_geometry["patch_shape"].size() == 2)
    {
        patch_height_ = support_geometry["patch_shape"][0].as<int>(0);
        patch_width_ = support_geometry["patch_shape"][1].as<int>(0);
    }

    const auto camera_group = cfg["observations"]["camera"];
    if (camera_group && camera_group[sensor_name_] && camera_group[sensor_name_]["params"]) {
        expected_size_ = camera_group[sensor_name_]["params"]["expected_size"].as<int>(0);
    }

    if (expected_size_ > 0 && patch_height_ * patch_width_ != expected_size_) {
        patch_height_ = 1;
        patch_width_ = expected_size_;
    }

    if (!topic_name_.empty()) {
        pointcloud_sub_ = std::make_shared<PointCloudSub>(topic_name_);
        pointcloud_sub_->set_timeout_ms(static_cast<uint32_t>(std::max(timeout_ms_, 1)));
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
        if (fill_from_pointcloud(buffer)) {
            warned_missing_feed_ = false;
            has_valid_frame_ = true;
            return;
        }
    }

    if (!warned_missing_feed_) {
        warned_missing_feed_ = true;
        if (topic_name_.empty()) {
            spdlog::warn(
                "SupportGeometryProvider '{}' has no topic configured; using fallback SGI handling.",
                sensor_name_
            );
        } else if (retain_last_valid_frame_ && has_valid_frame_) {
            spdlog::warn(
                "SupportGeometryProvider '{}' timed out waiting for topic '{}'; retaining last valid SGI frame.",
                sensor_name_, topic_name_
            );
        } else {
            spdlog::warn(
                "SupportGeometryProvider '{}' timed out waiting for topic '{}'; using fill-value SGI buffer.",
                sensor_name_, topic_name_
            );
        }
    }

    if (retain_last_valid_frame_ && has_valid_frame_) {
        return;
    }

    std::fill(buffer.begin(), buffer.end(), missing_fill_value_);
}

bool SupportGeometryProvider::fill_from_pointcloud(std::vector<float>& buffer) const
{
    if (!pointcloud_sub_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(pointcloud_sub_->mutex_);
    const auto& msg = pointcloud_sub_->msg_;
    if (msg.point_step() == 0 || msg.data().empty()) {
        return false;
    }

    const int x_offset = find_float32_field_offset(msg, pointcloud_field_names_.at("x"));
    const int y_offset = find_float32_field_offset(msg, pointcloud_field_names_.at("y"));
    const int z_offset = find_float32_field_offset(msg, pointcloud_field_names_.at("z"));
    if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
        return false;
    }

    const auto& data = msg.data();
    const size_t point_step = static_cast<size_t>(msg.point_step());
    const bool is_bigendian = msg.is_bigendian();
    const auto decode_point = [&](size_t base, float& normalized_value) -> bool {
        if (base + point_step > data.size()) {
            return false;
        }
        const float x = read_float32(data, base + static_cast<size_t>(x_offset), is_bigendian);
        const float y = read_float32(data, base + static_cast<size_t>(y_offset), is_bigendian);
        const float z = read_float32(data, base + static_cast<size_t>(z_offset), is_bigendian);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
            return false;
        }

        float value = 0.0f;
        if (pointcloud_mode_ == "z_depth") {
            value = z;
        } else {
            value = std::sqrt(x * x + y * y + z * z);
        }
        if (!std::isfinite(value)) {
            return false;
        }
        value = std::clamp(value, 0.0f, cutoff_distance_);
        normalized_value = cutoff_distance_ > 0.0f ? (value / cutoff_distance_) : value;
        return true;
    };

    size_t valid_points = 0;
    if (organized_pointcloud_ && patch_height_ > 0 && patch_width_ > 0 && msg.height() > 1 && msg.width() > 0) {
        const size_t source_height = static_cast<size_t>(msg.height());
        const size_t source_width = static_cast<size_t>(msg.width());
        const size_t row_step = msg.row_step() > 0 ? static_cast<size_t>(msg.row_step()) : point_step * source_width;
        const size_t target_height = static_cast<size_t>(patch_height_);
        const size_t target_width = static_cast<size_t>(patch_width_);

        for (size_t row = 0; row < target_height; ++row) {
            const size_t source_row = std::min(source_height - 1, (row * source_height) / target_height);
            for (size_t col = 0; col < target_width; ++col) {
                const size_t source_col = std::min(source_width - 1, (col * source_width) / target_width);
                const size_t buffer_index = row * target_width + col;
                const size_t base = source_row * row_step + source_col * point_step;
                float normalized_value = missing_fill_value_;
                if (decode_point(base, normalized_value)) {
                    ++valid_points;
                }
                buffer[buffer_index] = normalized_value;
            }
        }
    } else {
        const size_t max_points = std::min(buffer.size(), data.size() / point_step);
        for (size_t idx = 0; idx < buffer.size(); ++idx) {
            float normalized_value = missing_fill_value_;
            if (idx < max_points && decode_point(idx * point_step, normalized_value)) {
                ++valid_points;
            }
            buffer[idx] = normalized_value;
        }
    }

    return valid_points > 0;
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

float SupportGeometryProvider::read_float32(
    const std::vector<uint8_t>& data, size_t offset, bool is_bigendian
)
{
    if (offset + sizeof(float) > data.size()) {
        return 0.0f;
    }

    std::array<std::uint8_t, sizeof(float)> bytes{};
    std::memcpy(bytes.data(), data.data() + offset, sizeof(float));
    if (is_bigendian) {
        std::reverse(bytes.begin(), bytes.end());
    }

    float value = 0.0f;
    std::memcpy(&value, bytes.data(), sizeof(float));
    return value;
}
