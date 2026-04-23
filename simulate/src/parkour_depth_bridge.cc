#include "parkour_depth_bridge.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

#include <spdlog/spdlog.h>

#include "param.h"

namespace
{
constexpr int kPointStep = 3 * sizeof(float);

using Vec3 = std::array<mjtNum, 3>;
using Mat3 = std::array<mjtNum, 9>;

Vec3 matvec(const Mat3& matrix, const Vec3& vector)
{
  return {
    matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
    matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
    matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
  };
}

Mat3 matmul(const Mat3& lhs, const Mat3& rhs)
{
  Mat3 out{};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      out[row * 3 + col] =
        lhs[row * 3 + 0] * rhs[0 * 3 + col] +
        lhs[row * 3 + 1] * rhs[1 * 3 + col] +
        lhs[row * 3 + 2] * rhs[2 * 3 + col];
    }
  }
  return out;
}

Mat3 quat_to_mat(const mjtNum* quat)
{
  Mat3 matrix{};
  mju_quat2Mat(matrix.data(), quat);
  return matrix;
}

Mat3 body_xmat(const mjData* data, int body_id)
{
  Mat3 matrix{};
  std::copy_n(data->xmat + 9 * body_id, 9, matrix.begin());
  return matrix;
}

Vec3 body_xpos(const mjData* data, int body_id)
{
  return {
    data->xpos[3 * body_id + 0],
    data->xpos[3 * body_id + 1],
    data->xpos[3 * body_id + 2],
  };
}

Vec3 mat_column(const Mat3& matrix, int col)
{
  return {
    matrix[col],
    matrix[3 + col],
    matrix[6 + col],
  };
}

Vec3 normalize_xy(Vec3 vector)
{
  const mjtNum norm = std::sqrt(vector[0] * vector[0] + vector[1] * vector[1]);
  if (norm < 1e-6) {
    return {1.0, 0.0, 0.0};
  }
  return {vector[0] / norm, vector[1] / norm, 0.0};
}

Mat3 extract_yaw_rotation(const Mat3& rotation)
{
  Vec3 x_projection = mat_column(rotation, 0);
  x_projection[2] = 0.0;
  const mjtNum x_norm = std::sqrt(x_projection[0] * x_projection[0] + x_projection[1] * x_projection[1]);
  if (x_norm < 0.1) {
    Vec3 y_projection = mat_column(rotation, 1);
    y_projection[2] = 0.0;
    y_projection = normalize_xy(y_projection);
    x_projection = {y_projection[1], -y_projection[0], 0.0};
  } else {
    x_projection = normalize_xy(x_projection);
  }

  return {
    x_projection[0], -x_projection[1], 0.0,
    x_projection[1],  x_projection[0], 0.0,
    0.0,              0.0,             1.0,
  };
}

void write_gl_camera(mjvGLCamera& gl_camera, const Vec3& position, const Mat3& world_rotation)
{
  const Vec3 up = mat_column(world_rotation, 1);
  const Vec3 forward_axis = mat_column(world_rotation, 2);
  gl_camera.pos[0] = static_cast<float>(position[0]);
  gl_camera.pos[1] = static_cast<float>(position[1]);
  gl_camera.pos[2] = static_cast<float>(position[2]);
  gl_camera.forward[0] = static_cast<float>(-forward_axis[0]);
  gl_camera.forward[1] = static_cast<float>(-forward_axis[1]);
  gl_camera.forward[2] = static_cast<float>(-forward_axis[2]);
  gl_camera.up[0] = static_cast<float>(up[0]);
  gl_camera.up[1] = static_cast<float>(up[1]);
  gl_camera.up[2] = static_cast<float>(up[2]);
}
}

ParkourDepthBridge::ParkourDepthBridge(
    mujoco::Simulate* sim,
    GLFWwindow* shared_window,
    mjModel** model_ptr,
    mjData** data_ptr)
    : sim_(sim), shared_window_(shared_window), model_ptr_(model_ptr), data_ptr_(data_ptr)
{
  mjv_defaultScene(&scene_);
  mjv_defaultCamera(&camera_);
  mjv_defaultOption(&option_);
  option_.geomgroup[3] = 0;
  mjv_defaultPerturb(&perturb_);
  mjr_defaultContext(&context_);
}

ParkourDepthBridge::~ParkourDepthBridge()
{
  stop();
}

bool ParkourDepthBridge::start()
{
  if (running_ || stop_requested_) {
    return false;
  }

  const int width = std::max(param::config.depth_camera_width * param::config.depth_window_scale, 1);
  const int height = std::max(param::config.depth_camera_height * param::config.depth_window_scale, 1);
  window_ = glfwCreateWindow(width, height, "Parkour Depth", nullptr, shared_window_);
  if (!window_) {
    spdlog::warn("Failed to create parkour depth window; disabling depth bridge window output.");
    return false;
  }

  pointcloud_publisher_ = std::make_unique<PointCloudPublisher>(param::config.depth_pointcloud_topic);
  stop_requested_ = false;
  running_ = true;
  thread_ = std::thread(&ParkourDepthBridge::run, this);
  return true;
}

void ParkourDepthBridge::stop()
{
  stop_requested_ = true;
  if (thread_.joinable()) {
    thread_.join();
  }
  running_ = false;

  if (scene_ready_) {
    glfwMakeContextCurrent(window_);
    mjr_freeContext(&context_);
    mjv_freeScene(&scene_);
    glfwMakeContextCurrent(nullptr);
    scene_ready_ = false;
  }
  if (window_) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
  }
  pointcloud_publisher_.reset();
}

void ParkourDepthBridge::run()
{
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);

  while (!stop_requested_) {
    if (!ensure_render_resources()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      continue;
    }

    if (glfwWindowShouldClose(window_)) {
      break;
    }

    const int raw_width = std::max(param::config.depth_camera_width, 1);
    const int raw_height = std::max(param::config.depth_camera_height, 1);
    const mjrRect viewport{0, 0, raw_width, raw_height};

    std::vector<float> linear_depth;
    linear_depth.reserve(static_cast<size_t>(param::config.depth_camera_width * param::config.depth_camera_height));

    mjr_setBuffer(mjFB_OFFSCREEN, &context_);
    {
      const std::unique_lock<std::recursive_mutex> lock(sim_->mtx);
      mjModel* model = model_ptr_ ? *model_ptr_ : nullptr;
      mjData* data = data_ptr_ ? *data_ptr_ : nullptr;
      if (!model || !data) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        continue;
      }
      mj_forward(model, data);
      mjv_updateScene(model, data, &option_, &perturb_, &camera_, mjCAT_ALL, &scene_);
      apply_ray_alignment_override(model, data);
      mjr_render(viewport, &scene_, &context_);

      std::vector<float> depth_buffer(static_cast<size_t>(raw_width * raw_height), 1.0f);
      mjr_readPixels(nullptr, depth_buffer.data(), viewport, &context_);
      const float znear = static_cast<float>(model->vis.map.znear * model->stat.extent);
      const float zfar = static_cast<float>(model->vis.map.zfar * model->stat.extent);
      linear_depth.resize(depth_buffer.size());
      for (size_t i = 0; i < depth_buffer.size(); ++i) {
        float value = std::clamp(
          depth_buffer_to_meters(depth_buffer[i], znear, zfar),
          0.0f,
          param::config.depth_max_distance
        );
        if (value < param::config.depth_camera_min_distance) {
          value = param::config.depth_max_distance;
        }
        linear_depth[i] = value;
      }
    }

    mjr_setBuffer(mjFB_WINDOW, &context_);
    publish_pointcloud(linear_depth, raw_width, raw_height, camera_id_);
    draw_depth_window(linear_depth, raw_width, raw_height);
    glfwSwapBuffers(window_);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  glfwMakeContextCurrent(nullptr);
}

bool ParkourDepthBridge::ensure_render_resources()
{
  if (scene_ready_ && camera_id_ >= 0) {
    return true;
  }
  mjModel* model = model_ptr_ ? *model_ptr_ : nullptr;
  if (!model) {
    return false;
  }

  camera_id_ = mj_name2id(model, mjOBJ_CAMERA, param::config.depth_camera_name.c_str());
  if (camera_id_ < 0) {
    spdlog::warn("Depth camera '{}' was not found in the loaded MuJoCo model.", param::config.depth_camera_name);
    return false;
  }
  camera_body_id_ = model->cam_bodyid[camera_id_];

  mjv_makeScene(model, &scene_, 2000);
  mjr_makeContext(model, &context_, mjFONTSCALE_100);
  camera_.type = mjCAMERA_FIXED;
  camera_.fixedcamid = camera_id_;
  scene_ready_ = true;
  return true;
}

void ParkourDepthBridge::apply_ray_alignment_override(const mjModel* model, const mjData* data)
{
  if (!model || !data || camera_id_ < 0 || camera_body_id_ < 0) {
    return;
  }
  if (param::config.depth_camera_ray_alignment == "base") {
    return;
  }
  if (param::config.depth_camera_ray_alignment == "yaw") {
    const Mat3 body_rotation = body_xmat(data, camera_body_id_);
    const Mat3 camera_local_rotation = quat_to_mat(model->cam_quat + 4 * camera_id_);
    const Mat3 yaw_only_rotation = matmul(extract_yaw_rotation(body_rotation), camera_local_rotation);
    const Vec3 camera_local_pos = {
      model->cam_pos[3 * camera_id_ + 0],
      model->cam_pos[3 * camera_id_ + 1],
      model->cam_pos[3 * camera_id_ + 2],
    };
    const Vec3 camera_position = body_xpos(data, camera_body_id_);
    const Vec3 world_camera_position_offset = matvec(body_rotation, camera_local_pos);
    const Vec3 world_camera_position = {
      camera_position[0] + world_camera_position_offset[0],
      camera_position[1] + world_camera_position_offset[1],
      camera_position[2] + world_camera_position_offset[2],
    };
    write_gl_camera(scene_.camera[0], world_camera_position, yaw_only_rotation);
    write_gl_camera(scene_.camera[1], world_camera_position, yaw_only_rotation);
  }
}

void ParkourDepthBridge::publish_pointcloud(const std::vector<float>& linear_depth, int width, int height, int camera_id)
{
  if (!pointcloud_publisher_) {
    return;
  }

  const auto* model = model_ptr_ ? *model_ptr_ : nullptr;
  const float fovy_deg = (model && camera_id >= 0 && model->cam_fovy[camera_id] > 0.0)
      ? static_cast<float>(model->cam_fovy[camera_id])
      : 58.29f;
  const float fovy = fovy_deg * static_cast<float>(M_PI / 180.0);
  const float fy = static_cast<float>(height) / (2.0f * std::tan(fovy / 2.0f));
  const float fx = fy * (static_cast<float>(width) / static_cast<float>(height));
  const float cx = (static_cast<float>(width) - 1.0f) * 0.5f;
  const float cy = (static_cast<float>(height) - 1.0f) * 0.5f;

  pointcloud_publisher_->lock();
  auto& msg = pointcloud_publisher_->msg_;
  msg.height(static_cast<uint32_t>(height));
  msg.width(static_cast<uint32_t>(width));
  msg.is_bigendian(false);
  msg.point_step(kPointStep);
  msg.row_step(static_cast<uint32_t>(width * kPointStep));
  msg.is_dense(false);
  msg.fields() = {
    sensor_msgs::msg::dds_::PointField_("x", 0, sensor_msgs::msg::dds_::PointField_Constants::FLOAT32_, 1),
    sensor_msgs::msg::dds_::PointField_("y", 4, sensor_msgs::msg::dds_::PointField_Constants::FLOAT32_, 1),
    sensor_msgs::msg::dds_::PointField_("z", 8, sensor_msgs::msg::dds_::PointField_Constants::FLOAT32_, 1),
  };
  msg.data().resize(static_cast<size_t>(height * width * kPointStep));
  msg.header().frame_id(param::config.depth_camera_name);

  for (int row = 0; row < height; ++row) {
    const int source_row = height - 1 - row;
    for (int col = 0; col < width; ++col) {
      const size_t source_index = static_cast<size_t>(source_row * width + col);
      const float z = linear_depth[source_index];
      const float x = (static_cast<float>(col) - cx) * z / fx;
      const float y = (static_cast<float>(row) - cy) * z / fy;
      const size_t base = static_cast<size_t>(row * width + col) * kPointStep;
      std::memcpy(msg.data().data() + base + 0, &x, sizeof(float));
      std::memcpy(msg.data().data() + base + 4, &y, sizeof(float));
      std::memcpy(msg.data().data() + base + 8, &z, sizeof(float));
    }
  }

  pointcloud_publisher_->unlockAndPublish();
}

void ParkourDepthBridge::draw_depth_window(const std::vector<float>& linear_depth, int width, int height)
{
  int display_width = 0;
  int display_height = 0;
  glfwGetFramebufferSize(window_, &display_width, &display_height);
  if (display_width <= 0 || display_height <= 0) {
    return;
  }

  std::vector<unsigned char> rgb(static_cast<size_t>(display_width * display_height * 3), 0);
  const float max_distance = std::max(param::config.depth_max_distance, 1e-6f);
  const int crop_top = std::clamp(param::config.depth_debug_crop_top, 0, std::max(height - 1, 0));
  const int crop_left = std::clamp(param::config.depth_debug_crop_left, 0, std::max(width - 1, 0));
  const int crop_width = std::clamp(param::config.depth_debug_crop_width, 1, width - crop_left);
  const int crop_height = std::clamp(param::config.depth_debug_crop_height, 1, height - crop_top);
  for (int display_row = 0; display_row < display_height; ++display_row) {
    const int cropped_row = crop_top + std::min(crop_height - 1, display_row * crop_height / display_height);
    const int source_row = height - 1 - std::min(height - 1, cropped_row);
    for (int display_col = 0; display_col < display_width; ++display_col) {
      const int col = crop_left + std::min(crop_width - 1, display_col * crop_width / display_width);
      const size_t source_index = static_cast<size_t>(source_row * width + col);
      const float normalized = 1.0f - std::clamp(linear_depth[source_index] / max_distance, 0.0f, 1.0f);
      const auto value = static_cast<unsigned char>(normalized * 255.0f);
      const size_t display_index = static_cast<size_t>((display_row * display_width + display_col) * 3);
      rgb[display_index + 0] = value;
      rgb[display_index + 1] = value;
      rgb[display_index + 2] = value;
    }
  }

  if (context_.auxWidth[0] != display_width || context_.auxHeight[0] != display_height) {
    mjr_addAux(0, display_width, display_height, 0, &context_);
  }

  const mjrRect viewport{0, 0, display_width, display_height};
  mjr_setBuffer(mjFB_WINDOW, &context_);
  mjr_setAux(0, &context_);
  mjr_drawPixels(rgb.data(), nullptr, viewport, &context_);
  mjr_restoreBuffer(&context_);
  mjr_blitAux(0, viewport, 0, 0, &context_);
}

float ParkourDepthBridge::depth_buffer_to_meters(float depth_buffer_value, float znear, float zfar)
{
  const float z_ndc = 2.0f * depth_buffer_value - 1.0f;
  const float denominator = zfar + znear - z_ndc * (zfar - znear);
  if (std::abs(denominator) < 1e-6f) {
    return zfar;
  }
  return (2.0f * znear * zfar) / denominator;
}
