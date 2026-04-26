#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

#include "param.h"
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

        std::vector<float> current_joint_pos(data.joint_ids_map.size(), 0.0f);
        last_sensor_joint_vel_.assign(data.joint_ids_map.size(), 0.0f);
        for (int i = 0; i < data.joint_ids_map.size(); ++i) {
            current_joint_pos[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].q();
            data.joint_pos[i] = current_joint_pos[i];
            last_sensor_joint_vel_[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].dq();
        }

        const auto tick = lowstate->msg_.tick();
        last_finite_diff_joint_vel_ = last_sensor_joint_vel_;
        if ((param::joint_vel_source == "finite-diff-policy" || param::joint_vel_source == "finite-diff-lowstate")
            && finite_diff_initialized_) {
            float dt = 0.02f;
            if (param::joint_vel_source == "finite-diff-lowstate" && tick > last_lowstate_tick_) {
                dt = static_cast<float>(tick - last_lowstate_tick_) * 1.0e-3f;
            }
            if (dt > 1.0e-6f && last_joint_pos_.size() == current_joint_pos.size()) {
                for (size_t i = 0; i < current_joint_pos.size(); ++i) {
                    last_finite_diff_joint_vel_[i] = (current_joint_pos[i] - last_joint_pos_[i]) / dt;
                    data.joint_vel[i] = last_finite_diff_joint_vel_[i];
                }
            }
        } else {
            for (int i = 0; i < data.joint_ids_map.size(); ++i) {
                data.joint_vel[i] = last_sensor_joint_vel_[i];
            }
        }

        last_joint_pos_ = std::move(current_joint_pos);
        last_lowstate_tick_ = tick;
        finite_diff_initialized_ = true;
    }

    std::string joint_vel_source() const
    {
        return param::joint_vel_source;
    }

    std::vector<float> last_sensor_joint_vel() const
    {
        return last_sensor_joint_vel_;
    }

    std::vector<float> last_finite_diff_joint_vel() const
    {
        return last_finite_diff_joint_vel_;
    }

    LowStatePtr lowstate;

private:
    bool use_mjlab_body_frame_ = false;
    bool finite_diff_initialized_ = false;
    uint32_t last_lowstate_tick_ = 0;
    std::vector<float> last_joint_pos_;
    std::vector<float> last_sensor_joint_vel_;
    std::vector<float> last_finite_diff_joint_vel_;
};

} // namespace unitree
