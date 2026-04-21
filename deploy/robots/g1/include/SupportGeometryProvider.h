#pragma once

#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <unitree/dds_wrapper/common/Subscription.h>
#include <unitree/idl/ros2/PointCloud2_.hpp>

namespace isaaclab {
class Articulation;
}

class SupportGeometryProvider
{
public:
    explicit SupportGeometryProvider(const YAML::Node& cfg);

    void initialize(isaaclab::Articulation* robot);
    void update(isaaclab::Articulation* robot);

    [[nodiscard]] bool enabled() const { return enabled_; }
    [[nodiscard]] const std::string& sensor_name() const { return sensor_name_; }
    [[nodiscard]] int expected_size() const { return expected_size_; }

private:
    using PointCloudMsg = sensor_msgs::msg::dds_::PointCloud2_;
    using PointCloudSub = unitree::robot::SubscriptionBase<PointCloudMsg>;

    void fill_from_pointcloud(std::vector<float>& buffer) const;
    static int find_float32_field_offset(const PointCloudMsg& msg, const std::string& field_name);
    static float read_float32(const std::vector<uint8_t>& data, size_t offset);

    bool enabled_ = false;
    bool warned_missing_feed_ = false;
    std::string sensor_name_ = "support_depth";
    std::string topic_name_;
    std::string pointcloud_mode_ = "euclidean_norm";
    int expected_size_ = 0;
    float cutoff_distance_ = 1.0f;
    std::shared_ptr<PointCloudSub> pointcloud_sub_;
};
