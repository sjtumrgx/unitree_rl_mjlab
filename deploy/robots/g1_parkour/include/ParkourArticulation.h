#pragma once

#include <eigen3/Eigen/Dense>
#include <mutex>

#include "isaaclab/assets/articulation/articulation.h"

namespace unitree
{

inline Eigen::Vector3f align_policy_body_vector(const Eigen::Vector3f& sim_body_vector)
{
    static const Eigen::Matrix3f kPolicyFromSimImu =
        Eigen::AngleAxisf(-1.57079632679f, Eigen::Vector3f::UnitY()).toRotationMatrix();
    return kPolicyFromSimImu * sim_body_vector;
}

template <typename LowStatePtr>
class ParkourArticulation : public isaaclab::Articulation
{
public:
    explicit ParkourArticulation(LowStatePtr lowstate_, bool use_mjlab_body_frame = false)
    : lowstate(lowstate_), use_mjlab_body_frame_(use_mjlab_body_frame)
    {
        data.joystick = &lowstate->joystick;
    }

    void update() override
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);

        Eigen::Vector3f raw_root_ang_vel_b;
        for (int i = 0; i < 3; ++i) {
            raw_root_ang_vel_b[i] = lowstate->msg_.imu_state().gyroscope()[i];
        }
        if (use_mjlab_body_frame_) {
            data.root_ang_vel_b = raw_root_ang_vel_b;
        } else {
            data.root_ang_vel_b = align_policy_body_vector(raw_root_ang_vel_b);
        }

        data.root_quat_w = Eigen::Quaternionf(
            lowstate->msg_.imu_state().quaternion()[0],
            lowstate->msg_.imu_state().quaternion()[1],
            lowstate->msg_.imu_state().quaternion()[2],
            lowstate->msg_.imu_state().quaternion()[3]
        );

        const Eigen::Vector3f raw_projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        if (use_mjlab_body_frame_) {
            data.projected_gravity_b = raw_projected_gravity_b;
        } else {
            data.projected_gravity_b = align_policy_body_vector(raw_projected_gravity_b);
        }

        for (int i = 0; i < data.joint_ids_map.size(); ++i) {
            data.joint_pos[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].q();
            data.joint_vel[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].dq();
        }
    }

    LowStatePtr lowstate;

private:
    bool use_mjlab_body_frame_ = false;
};

} // namespace unitree
