#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

namespace isaaclab {
class Articulation;
}

class SupportGeometryProvider
{
public:
    explicit SupportGeometryProvider(const YAML::Node&) {}

    void initialize(isaaclab::Articulation*) {}
    void update(isaaclab::Articulation*) {}

    [[nodiscard]] bool enabled() const { return false; }
    [[nodiscard]] const std::string& sensor_name() const { return sensor_name_; }
    [[nodiscard]] int expected_size() const { return 0; }

private:
    std::string sensor_name_ = "support_depth";
};
