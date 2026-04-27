#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <unitree/dds_wrapper/common/Publisher.h>
#include <unitree/idl/ros2/PointCloud2_.hpp>
#include <unitree/idl/ros2/PointField_.hpp>

#include "simulate.h"

class ParkourDepthBridge
{
public:
  using PointCloudPublisher = unitree::robot::RealTimePublisher<sensor_msgs::msg::dds_::PointCloud2_>;

  ParkourDepthBridge(
    mujoco::Simulate* sim,
    GLFWwindow* shared_window,
    mjModel** model_ptr,
    mjData** data_ptr,
    std::atomic<bool>* data_ready,
    std::atomic<bool>* dds_ready);
  ~ParkourDepthBridge();

  bool start();
  void stop();

private:
  void run();
  bool create_render_window(int width, int height);
  bool ensure_render_resources();
  void apply_ray_alignment_override(const mjModel* model, const mjData* data);
  void repair_policy_crop_bottom_artifact_band(std::vector<float>& linear_depth, int width, int height) const;
  void publish_pointcloud(const std::vector<float>& linear_depth, int width, int height, int camera_id);
  void draw_depth_window(const std::vector<float>& linear_depth, int width, int height);
  static float depth_buffer_to_meters(float depth_buffer_value, float znear, float zfar);

  mujoco::Simulate* sim_ = nullptr;
  GLFWwindow* shared_window_ = nullptr;
  mjModel** model_ptr_ = nullptr;
  mjData** data_ptr_ = nullptr;
  std::atomic<bool>* data_ready_ = nullptr;
  std::atomic<bool>* dds_ready_ = nullptr;
  GLFWwindow* window_ = nullptr;
  std::unique_ptr<PointCloudPublisher> pointcloud_publisher_;
  std::thread thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};

  mjvScene scene_{};
  mjvCamera camera_{};
  mjvOption option_{};
  mjvPerturb perturb_{};
  mjrContext context_{};
  mjData* render_data_ = nullptr;
  const mjModel* render_model_ = nullptr;
  bool scene_ready_ = false;
  bool debug_window_visible_ = false;
  int camera_id_ = -1;
  int camera_body_id_ = -1;
};
