#include "ParkourDepthProvider.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <numeric>

#include <spdlog/spdlog.h>
#include <unitree/idl/ros2/PointField_.hpp>

#include "isaaclab/assets/articulation/articulation.h"

namespace
{

std::optional<float> parse_env_float(const char* name)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return std::nullopt;
    }
    char* parse_end = nullptr;
    const float parsed = std::strtof(raw, &parse_end);
    if (parse_end == raw) {
        spdlog::warn("ParkourDepthProvider ignored invalid {}='{}'.", name, raw);
        return std::nullopt;
    }
    return parsed;
}

} // namespace

ParkourDepthProvider::ParkourDepthProvider(const YAML::Node& cfg)
{
    const auto depth_cfg = cfg["parkour_depth_interface"];
    if (!depth_cfg) {
        return;
    }

    enabled_ = true;
    sensor_name_ = depth_cfg["sensor_name"].as<std::string>(sensor_name_);
    topic_name_ = depth_cfg["topic_name"].as<std::string>(topic_name_);
    pointcloud_mode_ = depth_cfg["pointcloud_mode"].as<std::string>(pointcloud_mode_);
    organized_pointcloud_ = depth_cfg["organized_pointcloud"].as<bool>(organized_pointcloud_);
    retain_last_valid_frame_ = depth_cfg["retain_last_valid_frame"].as<bool>(retain_last_valid_frame_);
    timeout_ms_ = depth_cfg["timeout_ms"].as<int>(timeout_ms_);
    raw_width_ = depth_cfg["raw_resolution"][0].as<int>(raw_width_);
    raw_height_ = depth_cfg["raw_resolution"][1].as<int>(raw_height_);
    crop_top_ = depth_cfg["crop_region"][0].as<int>(crop_top_);
    crop_bottom_ = depth_cfg["crop_region"][1].as<int>(crop_bottom_);
    crop_left_ = depth_cfg["crop_region"][2].as<int>(crop_left_);
    crop_right_ = depth_cfg["crop_region"][3].as<int>(crop_right_);
    output_width_ = depth_cfg["output_resolution"][0].as<int>(output_width_);
    output_height_ = depth_cfg["output_resolution"][1].as<int>(output_height_);
    history_source_length_ = depth_cfg["history_source_length"].as<int>(history_source_length_);
    history_skip_frames_ = depth_cfg["history_skip_frames"].as<int>(history_skip_frames_);
    num_output_frames_ = depth_cfg["num_output_frames"].as<int>(num_output_frames_);
    expected_size_ = depth_cfg["expected_size"].as<int>(expected_size_);
    depth_min_ = depth_cfg["depth_range"][0].as<float>(depth_min_);
    depth_max_ = depth_cfg["depth_range"][1].as<float>(depth_max_);
    output_min_ = depth_cfg["output_range"][0].as<float>(output_min_);
    output_max_ = depth_cfg["output_range"][1].as<float>(output_max_);
    artifact_ceiling_ = depth_cfg["artifact_ceiling"].as<float>(output_max_);
    artifact_ceiling_ = std::clamp(artifact_ceiling_, output_min_, output_max_);
    live_depth_blend_ = std::clamp(depth_cfg["live_depth_blend"].as<float>(live_depth_blend_), 0.0f, 1.0f);
    live_depth_baseline_ = std::clamp(
        depth_cfg["live_depth_baseline"].as<float>(live_depth_baseline_),
        output_min_,
        output_max_
    );
    if (const auto env_blend = parse_env_float("G1_PARKOUR_LIVE_DEPTH_BLEND")) {
        live_depth_blend_ = std::clamp(*env_blend, 0.0f, 1.0f);
        spdlog::warn(
            "ParkourDepthProvider '{}' overriding live_depth_blend from G1_PARKOUR_LIVE_DEPTH_BLEND={}.",
            sensor_name_,
            live_depth_blend_
        );
    }
    if (const auto env_baseline = parse_env_float("G1_PARKOUR_LIVE_DEPTH_BASELINE")) {
        live_depth_baseline_ = std::clamp(*env_baseline, output_min_, output_max_);
        spdlog::warn(
            "ParkourDepthProvider '{}' overriding live_depth_baseline from G1_PARKOUR_LIVE_DEPTH_BASELINE={}.",
            sensor_name_,
            live_depth_baseline_
        );
    }
    gaussian_kernel_size_ = depth_cfg["gaussian_kernel_size"].as<int>(gaussian_kernel_size_);
    gaussian_sigma_ = depth_cfg["gaussian_sigma"].as<float>(gaussian_sigma_);
    if (const auto field_names = depth_cfg["pointcloud_field_names"]) {
        pointcloud_field_names_["x"] = field_names["x"].as<std::string>(pointcloud_field_names_["x"]);
        pointcloud_field_names_["y"] = field_names["y"].as<std::string>(pointcloud_field_names_["y"]);
        pointcloud_field_names_["z"] = field_names["z"].as<std::string>(pointcloud_field_names_["z"]);
    }

    if (topic_name_.empty()) {
        enabled_ = false;
        return;
    }
    spdlog::info(
        "DEPTH_BLEND_CONFIG sensor={} live_depth_blend={} live_depth_baseline={} artifact_ceiling={}",
        sensor_name_,
        live_depth_blend_,
        live_depth_baseline_,
        artifact_ceiling_
    );

    if (const auto constant_depth = parse_env_float("G1_PARKOUR_DEBUG_CONSTANT_DEPTH")) {
        constant_depth_enabled_ = true;
        constant_depth_value_ = std::clamp(*constant_depth, output_min_, output_max_);
        spdlog::warn(
            "ParkourDepthProvider '{}' enabling constant-depth debug mode from G1_PARKOUR_DEBUG_CONSTANT_DEPTH={}.",
            sensor_name_,
            constant_depth_value_
        );
    }

    pointcloud_sub_ = std::make_shared<PointCloudSub>(topic_name_);
    pointcloud_sub_->set_timeout_ms(static_cast<uint32_t>(std::max(timeout_ms_, 1)));
}

void ParkourDepthProvider::initialize(isaaclab::Articulation* robot)
{
    if (!enabled_ || robot == nullptr || expected_size_ <= 0) {
        return;
    }

    warned_missing_feed_ = false;
    has_valid_frame_ = false;
    history_seeded_from_valid_frame_ = false;
    history_frames_.clear();
    robot->data.named_observations[sensor_name_] = std::vector<float>(static_cast<size_t>(expected_size_), output_min_);
}

void ParkourDepthProvider::reset(isaaclab::Articulation* robot)
{
    initialize(robot);
}

void ParkourDepthProvider::update(isaaclab::Articulation* robot)
{
    if (!enabled_ || robot == nullptr || expected_size_ <= 0) {
        return;
    }

    std::vector<float> frame(static_cast<size_t>(output_width_ * output_height_), output_min_);
    bool has_frame = false;
    if (constant_depth_enabled_) {
        has_frame = fill_frame_from_constant_depth(frame);
    } else if (pointcloud_sub_ && !pointcloud_sub_->isTimeout()) {
        has_frame = fill_frame_from_pointcloud(frame);
    }

    if (has_frame) {
        warned_missing_feed_ = false;
        if (!history_seeded_from_valid_frame_) {
            seed_history_from_frame(frame);
        } else {
            append_frame(std::move(frame));
        }
        has_valid_frame_ = true;
    } else {
        if (!warned_missing_feed_) {
            warned_missing_feed_ = true;
            spdlog::warn(
                "ParkourDepthProvider '{}' timed out waiting for topic '{}'; using {} frame.",
                sensor_name_,
                topic_name_,
                (retain_last_valid_frame_ && has_valid_frame_) ? "last valid" : "zero"
            );
        }
        if (retain_last_valid_frame_ && has_valid_frame_ && !history_frames_.empty()) {
            append_frame(history_frames_.back());
        } else if (has_valid_frame_) {
            append_frame(std::move(frame));
        }
    }

    robot->data.named_observations[sensor_name_] = compose_history_stack();
}

bool ParkourDepthProvider::fill_frame_from_pointcloud(std::vector<float>& frame) const
{
    if (!pointcloud_sub_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(pointcloud_sub_->mutex_);
    const auto& msg = pointcloud_sub_->msg_;
    if (msg.point_step() == 0 || msg.data().empty()) {
        return false;
    }

    const int z_offset = find_float32_field_offset(msg, pointcloud_field_names_.at("z"));
    if (z_offset < 0) {
        return false;
    }

    const auto& data = msg.data();
    const size_t point_step = static_cast<size_t>(msg.point_step());
    const size_t source_width = msg.width() > 0 ? static_cast<size_t>(msg.width()) : static_cast<size_t>(raw_width_);
    const size_t source_height = msg.height() > 0 ? static_cast<size_t>(msg.height()) : static_cast<size_t>(raw_height_);
    const size_t row_step = msg.row_step() > 0 ? static_cast<size_t>(msg.row_step()) : point_step * source_width;
    const bool is_bigendian = msg.is_bigendian();

    size_t valid_points = 0;
    for (int row = 0; row < output_height_; ++row) {
        const size_t source_row = std::min(
            source_height - 1,
            static_cast<size_t>(crop_top_ + row)
        );
        for (int col = 0; col < output_width_; ++col) {
            const size_t source_col = std::min(
                source_width - 1,
                static_cast<size_t>(crop_left_ + col)
            );
            const size_t base = source_row * row_step + source_col * point_step;
            float value = depth_max_;
            if (base + point_step <= data.size()) {
                value = read_float32(data, base + static_cast<size_t>(z_offset), is_bigendian);
                if (!std::isfinite(value) || value <= 0.0f) {
                    value = depth_max_;
                } else {
                    ++valid_points;
                }
            }
            value = std::clamp(value, depth_min_, depth_max_);
            const float normalized = depth_max_ > depth_min_
                ? (value - depth_min_) / (depth_max_ - depth_min_)
                : value;
            float output_value = normalized * (output_max_ - output_min_) + output_min_;
            // The C++ OpenGL/DDS route can produce isolated z-far pixels at
            // the top edge of the policy crop on flat ground, while the Python
            // MuJoCo renderer parity path stays below ~0.71 for the same
            // camera/crop.  Those artifact-white pixels strongly perturb the
            // depth encoder and were the remaining cause of live-depth falls.
            // Clamp only the configured normalized ceiling; constant-depth
            // diagnostics and true range normalization remain unchanged.
            output_value = std::min(output_value, artifact_ceiling_);
            output_value = live_depth_baseline_ * (1.0f - live_depth_blend_) + output_value * live_depth_blend_;
            frame[static_cast<size_t>(row * output_width_ + col)] = output_value;
        }
    }

    if (gaussian_kernel_size_ >= 3) {
        apply_gaussian_blur(frame);
    }
    return valid_points > 0;
}

bool ParkourDepthProvider::fill_frame_from_constant_depth(std::vector<float>& frame) const
{
    std::fill(frame.begin(), frame.end(), constant_depth_value_);
    return true;
}

void ParkourDepthProvider::seed_history_from_frame(const std::vector<float>& frame)
{
    history_frames_.clear();
    for (int i = 0; i < history_source_length_; ++i) {
        history_frames_.push_back(frame);
    }
    history_seeded_from_valid_frame_ = true;
    const auto [frame_min, frame_max] = std::minmax_element(frame.begin(), frame.end());
    const float frame_mean = frame.empty()
        ? 0.0f
        : std::accumulate(frame.begin(), frame.end(), 0.0f) / static_cast<float>(frame.size());
    spdlog::info(
        "FIRST_VALID_DEPTH_STACK size={} sensor={} source_history={} frame[min,max,mean]=[{},{},{}]",
        expected_size_,
        sensor_name_,
        history_source_length_,
        frame_min != frame.end() ? *frame_min : 0.0f,
        frame_max != frame.end() ? *frame_max : 0.0f,
        frame_mean
    );
}

void ParkourDepthProvider::append_frame(std::vector<float> frame)
{
    history_frames_.push_back(std::move(frame));
    while (static_cast<int>(history_frames_.size()) > history_source_length_) {
        history_frames_.pop_front();
    }
}

std::vector<float> ParkourDepthProvider::compose_history_stack() const
{
    std::vector<float> stacked;
    stacked.reserve(static_cast<size_t>(expected_size_));
    if (history_frames_.empty()) {
        stacked.resize(static_cast<size_t>(expected_size_), output_min_);
        return stacked;
    }

    const int available = static_cast<int>(history_frames_.size());
    const int start_index = std::max(0, available - 1 - history_skip_frames_ * (num_output_frames_ - 1));
    for (int i = 0; i < num_output_frames_; ++i) {
        const int index = std::min(available - 1, start_index + i * history_skip_frames_);
        const auto& frame = history_frames_[index];
        stacked.insert(stacked.end(), frame.begin(), frame.end());
    }
    if (static_cast<int>(stacked.size()) < expected_size_) {
        stacked.resize(static_cast<size_t>(expected_size_), output_min_);
    }
    return stacked;
}

void ParkourDepthProvider::apply_gaussian_blur(std::vector<float>& frame) const
{
    if (output_width_ <= 2 || output_height_ <= 2) {
        return;
    }
    std::vector<float> copy = frame;
    const float kernel[3][3] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f},
    };
    for (int row = 1; row < output_height_ - 1; ++row) {
        for (int col = 1; col < output_width_ - 1; ++col) {
            float sum = 0.0f;
            for (int kr = -1; kr <= 1; ++kr) {
                for (int kc = -1; kc <= 1; ++kc) {
                    sum += kernel[kr + 1][kc + 1] * copy[static_cast<size_t>((row + kr) * output_width_ + (col + kc))];
                }
            }
            frame[static_cast<size_t>(row * output_width_ + col)] = sum / 16.0f;
        }
    }
}

int ParkourDepthProvider::find_float32_field_offset(const PointCloudMsg& msg, const std::string& field_name)
{
    for (const auto& field : msg.fields()) {
        if (field.name() == field_name
            && field.datatype() == sensor_msgs::msg::dds_::PointField_Constants::FLOAT32_) {
            return static_cast<int>(field.offset());
        }
    }
    return -1;
}

float ParkourDepthProvider::read_float32(const std::vector<uint8_t>& data, size_t offset, bool is_bigendian)
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
