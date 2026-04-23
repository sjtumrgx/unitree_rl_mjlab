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

  ParkourDepthBridge(mujoco::Simulate* sim, GLFWwindow* shared_window, mjModel** model_ptr, mjData** data_ptr);
  ~ParkourDepthBridge();

  bool start();
  void stop();

private:
  void run();
  bool ensure_render_resources();
  void apply_ray_alignment_override(const mjModel* model, const mjData* data);
  void publish_pointcloud(const std::vector<float>& linear_depth, int width, int height, int camera_id);
  void draw_depth_window(const std::vector<float>& linear_depth, int width, int height);
  static float depth_buffer_to_meters(float depth_buffer_value, float znear, float zfar);

  mujoco::Simulate* sim_ = nullptr;
  GLFWwindow* shared_window_ = nullptr;
  mjModel** model_ptr_ = nullptr;
  mjData** data_ptr_ = nullptr;
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
  bool scene_ready_ = false;
  int camera_id_ = -1;
  int camera_body_id_ = -1;
};
