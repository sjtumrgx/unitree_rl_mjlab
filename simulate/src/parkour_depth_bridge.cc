#include "parkour_depth_bridge.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
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
    mjData** data_ptr,
    std::atomic<bool>* dds_ready)
    : sim_(sim),
      shared_window_(shared_window),
      model_ptr_(model_ptr),
      data_ptr_(data_ptr),
      dds_ready_(dds_ready)
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
  if (!create_render_window(width, height)) {
    spdlog::error(
      "ParkourDepthBridge publisher is ready, but no GLFW render context could be created; depth frames cannot be rendered."
    );
    return false;
  }

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

  if (scene_ready_ && window_) {
    glfwMakeContextCurrent(window_);
    mjr_freeContext(&context_);
    mjv_freeScene(&scene_);
    glfwMakeContextCurrent(nullptr);
    scene_ready_ = false;
  }
  if (render_data_) {
    mj_deleteData(render_data_);
    render_data_ = nullptr;
    render_model_ = nullptr;
  }
  if (window_) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
  }
  pointcloud_publisher_.reset();
}

void ParkourDepthBridge::run()
{
  if (!window_) {
    return;
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);

  while (!stop_requested_ && dds_ready_ && !dds_ready_->load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(std::max(param::config.depth_publish_period_ms, 1)));
  }
  if (stop_requested_) {
    glfwMakeContextCurrent(nullptr);
    return;
  }
  if (!pointcloud_publisher_) {
    pointcloud_publisher_ = std::make_unique<PointCloudPublisher>(param::config.depth_pointcloud_topic);
    spdlog::info(
      "PUBLISHER_READY topic={} camera={} raw={}x{} max_distance={}",
      param::config.depth_pointcloud_topic,
      param::config.depth_camera_name,
      param::config.depth_camera_width,
      param::config.depth_camera_height,
      param::config.depth_max_distance
    );
  }

  while (!stop_requested_) {
    if (!ensure_render_resources()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      continue;
    }

    if (debug_window_visible_ && glfwWindowShouldClose(window_)) {
      break;
    }

    const int raw_width = std::max(param::config.depth_camera_width, 1);
    const int raw_height = std::max(param::config.depth_camera_height, 1);
    const mjrRect viewport{0, 0, raw_width, raw_height};

    std::vector<float> linear_depth;
    linear_depth.reserve(static_cast<size_t>(param::config.depth_camera_width * param::config.depth_camera_height));
    const mjModel* render_model = nullptr;
    mjData* render_data = nullptr;

    {
      const std::unique_lock<std::recursive_mutex> lock(sim_->mtx);
      mjModel* model = model_ptr_ ? *model_ptr_ : nullptr;
      mjData* data = data_ptr_ ? *data_ptr_ : nullptr;
      if (!model || !data || !render_data_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        continue;
      }
      // Copy the live MuJoCo state while holding the Simulate mutex, then
      // release the UI/physics lock before doing OpenGL rendering.  Keeping the
      // mutex across mjr_render/mjr_readPixels made the native viewer appear
      // frozen once the DDS controller started physics.
      mj_copyData(render_data_, model, data);
      render_model = model;
      render_data = render_data_;
    }

    mjr_setBuffer(mjFB_OFFSCREEN, &context_);
    mj_forward(render_model, render_data);
    mjv_updateScene(render_model, render_data, &option_, &perturb_, &camera_, mjCAT_ALL, &scene_);
    apply_ray_alignment_override(render_model, render_data);
    mjr_render(viewport, &scene_, &context_);

    std::vector<float> depth_buffer(static_cast<size_t>(raw_width * raw_height), 1.0f);
    mjr_readPixels(nullptr, depth_buffer.data(), viewport, &context_);
    const float znear = static_cast<float>(render_model->vis.map.znear * render_model->stat.extent);
    const float zfar = static_cast<float>(render_model->vis.map.zfar * render_model->stat.extent);
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

    mjr_setBuffer(mjFB_WINDOW, &context_);
    publish_pointcloud(linear_depth, raw_width, raw_height, camera_id_);
    if (debug_window_visible_) {
      draw_depth_window(linear_depth, raw_width, raw_height);
      glfwSwapBuffers(window_);
    }
    // Respect the configured publish period.  The previous fixed 20 ms sleep
    // forced the hidden GL renderer to run at ~50 Hz even when
    // depth_publish_period_ms was set lower (for example 100 ms).  That extra
    // render/copy load perturbs the asynchronous DDS controller enough to
    // destabilize an otherwise constant-depth walk.
    std::this_thread::sleep_for(
      std::chrono::milliseconds(std::max(param::config.depth_publish_period_ms, 1))
    );
  }

  glfwMakeContextCurrent(nullptr);
}

bool ParkourDepthBridge::create_render_window(int width, int height)
{
  const char* debug_window_env = std::getenv("G1_PARKOUR_DEPTH_DEBUG_WINDOW");
  const bool want_visible_window = !(debug_window_env && std::string(debug_window_env) == "0");

  if (want_visible_window) {
    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    window_ = glfwCreateWindow(width, height, "Parkour Depth", nullptr, shared_window_);
    if (window_) {
      debug_window_visible_ = true;
      return true;
    }
    spdlog::warn("Failed to create visible parkour depth debug window; retrying with a hidden render context.");
  }

  glfwDefaultWindowHints();
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  window_ = glfwCreateWindow(width, height, "Parkour Depth Render Context", nullptr, shared_window_);
  glfwDefaultWindowHints();
  if (!window_) {
    debug_window_visible_ = false;
    return false;
  }
  debug_window_visible_ = false;
  spdlog::info("ParkourDepthBridge using a hidden render context; DDS depth publishing remains enabled.");
  return true;
}

bool ParkourDepthBridge::ensure_render_resources()
{
  const mjModel* model = model_ptr_ ? *model_ptr_ : nullptr;
  if (scene_ready_ && camera_id_ >= 0 && render_data_ && render_model_ == model) {
    return true;
  }
  if (!model) {
    return false;
  }
  if (scene_ready_) {
    mjr_freeContext(&context_);
    mjv_freeScene(&scene_);
    scene_ready_ = false;
  }
  if (render_data_) {
    mj_deleteData(render_data_);
    render_data_ = nullptr;
    render_model_ = nullptr;
  }

  camera_id_ = mj_name2id(model, mjOBJ_CAMERA, param::config.depth_camera_name.c_str());
  if (camera_id_ < 0) {
    spdlog::warn("Depth camera '{}' was not found in the loaded MuJoCo model.", param::config.depth_camera_name);
    return false;
  }
  camera_body_id_ = model->cam_bodyid[camera_id_];

  mjv_makeScene(model, &scene_, 2000);
  mjr_makeContext(model, &context_, mjFONTSCALE_100);
  render_data_ = mj_makeData(model);
  if (!render_data_) {
    spdlog::warn("Failed to allocate parkour depth render-data snapshot.");
    mjr_freeContext(&context_);
    mjv_freeScene(&scene_);
    return false;
  }
  render_model_ = model;
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
      float z = linear_depth[source_index];
      if (!std::isfinite(z)) {
        z = param::config.depth_max_distance;
      }
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
    // `mjr_readPixels` returns OpenGL framebuffer rows with the first row at
    // the bottom.  The policy/debug crop is expressed in camera/top-origin
    // coordinates, but `mjr_drawPixels` also consumes bottom-origin pixel rows.
    // Therefore the DDS point cloud flips rows for top-origin policy data,
    // while this debug window flips the *display row* so the visible image is
    // not upside down.
    const int top_origin_row = crop_top + std::min(
      crop_height - 1,
      (display_height - 1 - display_row) * crop_height / display_height
    );
    const int source_row = height - 1 - std::min(height - 1, top_origin_row);
    for (int display_col = 0; display_col < display_width; ++display_col) {
      const int col = crop_left + std::min(crop_width - 1, display_col * crop_width / display_width);
      const size_t source_index = static_cast<size_t>(source_row * width + col);
      const float normalized = std::clamp(linear_depth[source_index] / max_distance, 0.0f, 1.0f);
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
