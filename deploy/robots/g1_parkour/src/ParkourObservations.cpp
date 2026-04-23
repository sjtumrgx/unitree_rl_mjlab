#include "ParkourObservations.h"
#include "isaaclab/envs/mdp/observations/observations.h"

namespace isaaclab
{
namespace mdp
{

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
    (void)isaaclab::observations_map();
}
