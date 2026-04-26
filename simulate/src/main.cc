// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// !!! hack code: make glfw_adapter.window_ public
#define private public
#include "glfw_adapter.h"
#undef private

#include <chrono>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <atomic>
#include <sstream>
#include <string>
#include <thread>

#include <mujoco/mujoco.h>
#include "simulate.h"
#include "array_safety.h"
#include "unitree_sdk2_bridge.h"
#include "param.h"
#include "parkour_depth_bridge.h"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"
#define NUM_MOTOR_IDL_GO 20

extern "C"
{
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif
}

class ElasticBand
{
public:
  ElasticBand(){};
  void Advance(std::vector<double> x, std::vector<double> dx)
  {
    std::vector<double> delta_x = {0.0, 0.0, 0.0};
    delta_x[0] = point_[0] - x[0];
    delta_x[1] = point_[1] - x[1];
    delta_x[2] = point_[2] - x[2];
    double distance = sqrt(delta_x[0] * delta_x[0] + delta_x[1] * delta_x[1] + delta_x[2] * delta_x[2]);

    std::vector<double> direction = {0.0, 0.0, 0.0};
    direction[0] = delta_x[0] / distance;
    direction[1] = delta_x[1] / distance;
    direction[2] = delta_x[2] / distance;

    double v = dx[0] * direction[0] + dx[1] * direction[1] + dx[2] * direction[2];

    f_[0] = (stiffness_ * (distance - length_) - damping_ * v) * direction[0];
    f_[1] = (stiffness_ * (distance - length_) - damping_ * v) * direction[1];
    f_[2] = (stiffness_ * (distance - length_) - damping_ * v) * direction[2];
  }


  double stiffness_ = 200;
  double damping_ = 100;
  std::vector<double> point_ = {0, 0, 3};
  double length_ = 0.0;
  bool enable_ = true;
  std::vector<double> f_ = {0, 0, 0};
};
inline ElasticBand elastic_band;


namespace
{
  namespace mj = ::mujoco;
  namespace mju = ::mujoco::sample_util;

  std::filesystem::path config_path_for_executable(const char* argv0)
  {
    const std::string executable_name = std::filesystem::path(argv0).filename().string();
    if (executable_name.find("parkour") != std::string::npos) {
      return "config_parkour.yaml";
    }
    return "config.yaml";
  }

  bool env_flag_is_false(const char* value)
  {
    if (!value) {
      return false;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return normalized == "0" || normalized == "false" || normalized == "off" || normalized == "no";
  }

  struct ParkourProgressTracker
  {
    bool initialized = false;
    double initial_root_x = 0.0;
    double next_progress_time = 0.0;
    bool distance_marker_logged = false;
    bool fall_marker_logged = false;
  };

  double configured_walk_distance_marker()
  {
    return std::max(0.0f, param::config.walk_distance_marker);
  }

  double configured_progress_log_interval()
  {
    return std::max(0.05f, param::config.progress_log_interval);
  }

  std::string format_distance_marker(double distance)
  {
    std::ostringstream os;
    os << std::fixed << std::setprecision(1) << distance;
    return os.str();
  }

  void quat_to_roll_pitch_yaw(const mjtNum* quat, double& roll, double& pitch, double& yaw)
  {
    const double w = quat[0];
    const double x = quat[1];
    const double y = quat[2];
    const double z = quat[3];
    roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
    const double sin_pitch = std::clamp(2.0 * (w * y - z * x), -1.0, 1.0);
    pitch = std::asin(sin_pitch);
    yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  }

  double distance_x_from_tracker(ParkourProgressTracker& tracker, const mjData* data)
  {
    if (!tracker.initialized)
    {
      tracker.initial_root_x = data->qpos[0];
      tracker.next_progress_time = data->time;
      tracker.initialized = true;
    }
    return data->qpos[0] - tracker.initial_root_x;
  }

  void reset_progress_tracker(ParkourProgressTracker& tracker)
  {
    tracker = ParkourProgressTracker{};
  }

  void emit_progress_if_due(ParkourProgressTracker& tracker, const mjData* data, const char* source)
  {
    const double distance_x = distance_x_from_tracker(tracker, data);
    if (data->time + 1e-9 < tracker.next_progress_time)
    {
      return;
    }

    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    quat_to_roll_pitch_yaw(&data->qpos[3], roll, pitch, yaw);
    std::cout << "PARKOUR_PROGRESS source=" << source
              << " sim_time=" << data->time
              << " distance_x=" << distance_x
              << " root_x=" << data->qpos[0]
              << " base_height=" << data->qpos[2]
              << " roll=" << roll
              << " pitch=" << pitch
              << " yaw=" << yaw
              << " lowcmd_connected=" << param::lowcmd_connected.load()
              << " lowcmd_active=" << param::lowcmd_has_active_control.load()
              << std::endl;
    tracker.next_progress_time += configured_progress_log_interval();
  }

  bool emit_distance_marker_if_reached(ParkourProgressTracker& tracker, const mjData* data)
  {
    const double target_distance = configured_walk_distance_marker();
    const double distance_x = distance_x_from_tracker(tracker, data);
    if (!tracker.distance_marker_logged && distance_x >= target_distance)
    {
      std::cout << "DISTANCE_X>=" << format_distance_marker(target_distance)
                << " distance_x=" << distance_x
                << " root_x=" << data->qpos[0]
                << " sim_time=" << data->time << std::endl;
      tracker.distance_marker_logged = true;
      return true;
    }
    return tracker.distance_marker_logged;
  }

  bool emit_fall_marker_if_needed(ParkourProgressTracker& tracker, const mjData* data)
  {
    if (!tracker.fall_marker_logged && data->qpos[2] < 0.35)
    {
      double roll = 0.0;
      double pitch = 0.0;
      double yaw = 0.0;
      quat_to_roll_pitch_yaw(&data->qpos[3], roll, pitch, yaw);
      std::cout << "FALL_RESET_DETECTED base_height=" << data->qpos[2]
                << " distance_x=" << distance_x_from_tracker(tracker, data)
                << " roll=" << roll
                << " pitch=" << pitch
                << " yaw=" << yaw
                << " sim_time=" << data->time << std::endl;
      tracker.fall_marker_logged = true;
      return true;
    }
    return tracker.fall_marker_logged;
  }

  // constants
  const double syncMisalign = 0.1;       // maximum mis-alignment before re-sync (simulation seconds)
  const double simRefreshFraction = 0.7; // fraction of refresh available for simulation
  const int kErrorLength = 1024;         // load error string length

  // model and data
  mjModel *m = nullptr;
  mjData *d = nullptr;
  std::atomic<bool> mujoco_data_initialized{false};
  std::atomic<bool> unitree_channel_ready{false};

  // control noise variables
  mjtNum *ctrlnoise = nullptr;

  using Seconds = std::chrono::duration<double>;

  void apply_configured_pose_to_qpos(const mjModel* model, mjtNum* qpos)
  {
    if (!model || !qpos) {
      return;
    }
    if (!param::config.initial_base_pos.empty() && param::config.initial_base_pos.size() == 3 && model->nq >= 3) {
      qpos[0] = param::config.initial_base_pos[0];
      qpos[1] = param::config.initial_base_pos[1];
      qpos[2] = param::config.initial_base_pos[2];
    }
    if (!param::config.initial_base_quat.empty() && param::config.initial_base_quat.size() == 4 && model->nq >= 7) {
      qpos[3] = param::config.initial_base_quat[0];
      qpos[4] = param::config.initial_base_quat[1];
      qpos[5] = param::config.initial_base_quat[2];
      qpos[6] = param::config.initial_base_quat[3];
    }
    if (param::config.initial_joint_pos.empty()) {
      return;
    }
    if (param::config.initial_joint_pos.size() == static_cast<size_t>(model->nu)) {
      // Parkour initial_joint_pos is interpreted in actuator/motor order, which
      // matches lowcmd/lowstate sensors but differs from the XML joint order.
      for (int actuator_id = 0; actuator_id < model->nu; ++actuator_id) {
        const int joint_id = model->actuator_trnid[2 * actuator_id];
        if (joint_id < 0 || model->jnt_type[joint_id] == mjJNT_FREE) {
          continue;
        }
        qpos[model->jnt_qposadr[joint_id]] = param::config.initial_joint_pos[actuator_id];
      }
      return;
    }
    size_t joint_index = 0;
    for (int joint_id = 0; joint_id < model->njnt && joint_index < param::config.initial_joint_pos.size(); ++joint_id)
    {
      if (model->jnt_type[joint_id] == mjJNT_FREE) {
        continue;
      }
      qpos[model->jnt_qposadr[joint_id]] = param::config.initial_joint_pos[joint_index++];
    }
  }

  //---------------------------------------- plugin handling -----------------------------------------

  // return the path to the directory containing the current executable
  // used to determine the location of auto-loaded plugin libraries
  std::string getExecutableDir()
  {
#if defined(_WIN32) || defined(__CYGWIN__)
    constexpr char kPathSep = '\\';
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      DWORD buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
        if (written < buf_size)
        {
          success = true;
        }
        else if (written == buf_size)
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
        else
        {
          std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
          return "";
        }
      }
      return realpath.get();
    }();
#else
    constexpr char kPathSep = '/';
#if defined(__APPLE__)
    std::unique_ptr<char[]> buf(nullptr);
    {
      std::uint32_t buf_size = 0;
      _NSGetExecutablePath(nullptr, &buf_size);
      buf.reset(new char[buf_size]);
      if (!buf)
      {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }
      if (_NSGetExecutablePath(buf.get(), &buf_size))
      {
        std::cerr << "unexpected error from _NSGetExecutablePath\n";
      }
    }
    const char *path = buf.get();
#else
    const char *path = "/proc/self/exe";
#endif
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      std::uint32_t buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        std::size_t written = readlink(path, realpath.get(), buf_size);
        if (written < buf_size)
        {
          realpath.get()[written] = '\0';
          success = true;
        }
        else if (written == -1)
        {
          if (errno == EINVAL)
          {
            // path is already not a symlink, just use it
            return path;
          }

          std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
          return "";
        }
        else
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
      }
      return realpath.get();
    }();
#endif

    if (realpath.empty())
    {
      return "";
    }

    for (std::size_t i = realpath.size() - 1; i > 0; --i)
    {
      if (realpath.c_str()[i] == kPathSep)
      {
        return realpath.substr(0, i);
      }
    }

    // don't scan through the entire file system's root
    return "";
  }

  // scan for libraries in the plugin directory to load additional plugins
  void scanPluginLibraries()
  {
    // check and print plugins that are linked directly into the executable
    int nplugin = mjp_pluginCount();
    if (nplugin)
    {
      std::printf("Built-in plugins:\n");
      for (int i = 0; i < nplugin; ++i)
      {
        std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
      }
    }

    // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
    const std::string sep = "\\";
#else
    const std::string sep = "/";
#endif

    // try to open the ${EXECDIR}/plugin directory
    // ${EXECDIR} is the directory containing the simulate binary itself
    const std::string executable_dir = getExecutableDir();
    if (executable_dir.empty())
    {
      return;
    }

    const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
    mj_loadAllPluginLibraries(
        plugin_dir.c_str(), +[](const char *filename, int first, int count)
                            {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        } });
  }

  //------------------------------------------- simulation -------------------------------------------

  mjModel *LoadModel(const char *file, mj::Simulate &sim)
  {
    // this copy is needed so that the mju::strlen call below compiles
    char filename[mj::Simulate::kMaxFilenameLength];
    mju::strcpy_arr(filename, file);

    // make sure filename is not empty
    if (!filename[0])
    {
      return nullptr;
    }

    // load and compile
    char loadError[kErrorLength] = "";
    mjModel *mnew = 0;
    if (mju::strlen_arr(filename) > 4 &&
        !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                      mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4))
    {
      mnew = mj_loadModel(filename, nullptr);
      if (!mnew)
      {
        mju::strcpy_arr(loadError, "could not load binary model");
      }
    }
    else
    {
      mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);
      // remove trailing newline character from loadError
      if (loadError[0])
      {
        int error_length = mju::strlen_arr(loadError);
        if (loadError[error_length - 1] == '\n')
        {
          loadError[error_length - 1] = '\0';
        }
      }
    }

    mju::strcpy_arr(sim.load_error, loadError);

    if (!mnew)
    {
      std::printf("%s\n", loadError);
      return nullptr;
    }

    // compiler warning: print and pause
    if (loadError[0])
    {
      // mj_forward() below will print the warning message
      std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
      sim.run = 0;
    }

    return mnew;
  }

  // simulate in background thread (while rendering in main thread)
  void apply_initial_joint_pose()
  {
    if (!m || !d) {
      return;
    }
    apply_configured_pose_to_qpos(m, d->qpos);
    mju_zero(d->qvel, m->nv);
    mju_zero(d->qacc, m->nv);
    mju_zero(d->qacc_warmstart, m->nv);
    mju_zero(d->ctrl, m->nu);
  }

  void PhysicsLoop(mj::Simulate &sim)
  {
    // cpu-sim syncronization point
    std::chrono::time_point<mj::Simulate::Clock> syncCPU;
    std::chrono::time_point<mj::Simulate::Clock> nextLockstepCPU;
    mjtNum syncSim = 0;
    ParkourProgressTracker progress_tracker;
    double last_observed_sim_time = 0.0;
    bool observed_sim_time_initialized = false;

    // ChannelFactory::Instance()->Init(0);
    // UnitreeDds ud(d);

    // run until asked to exit
    while (!sim.exitrequest.load())
    {
      if (sim.droploadrequest.load())
      {
        sim.LoadMessage(sim.dropfilename);
        mjModel *mnew = LoadModel(sim.dropfilename, sim);
        sim.droploadrequest.store(false);

        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          mujoco_data_initialized.store(false);
          sim.Load(mnew, dnew, sim.dropfilename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          apply_initial_joint_pose();
          mj_forward(m, d);
          mujoco_data_initialized.store(true);
          reset_progress_tracker(progress_tracker);
          observed_sim_time_initialized = false;

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = (mjtNum *)malloc(sizeof(mjtNum) * m->nu);
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      if (sim.uiloadrequest.load())
      {
        sim.uiloadrequest.fetch_sub(1);
        sim.LoadMessage(sim.filename);
        mjModel *mnew = LoadModel(sim.filename, sim);
        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          mujoco_data_initialized.store(false);
          sim.Load(mnew, dnew, sim.filename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          apply_initial_joint_pose();
          mj_forward(m, d);
          mujoco_data_initialized.store(true);
          reset_progress_tracker(progress_tracker);
          observed_sim_time_initialized = false;

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      // sleep for 1 ms or yield, to let main thread run
      //  yield results in busy wait - which has better timing but kills battery life
      if (sim.run && sim.busywait)
      {
        std::this_thread::yield();
      }
      else
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      {
        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        // run only if model is present
        if (m)
        {
          if (
            observed_sim_time_initialized &&
            d &&
            d->time + 1.0e-9 < last_observed_sim_time
          )
          {
            // MuJoCo's native UI reset path calls mj_resetData() internally.
            // Do not mutate the model default pose to intercept that path:
            // changing it after XML compilation changes simulator dynamics
            // enough to destabilize the parkour policy.  Instead, detect the
            // time rewind before the next physics step and reapply the
            // configured parkour pose directly to mjData.
            apply_initial_joint_pose();
            mj_forward(m, d);
            reset_progress_tracker(progress_tracker);
            syncCPU = mj::Simulate::Clock::time_point{};
            nextLockstepCPU = mj::Simulate::Clock::time_point{};
            syncSim = d->time;
            sim.speed_changed = true;
            std::cout << "PARKOUR_SIM_RESET_REAPPLIED_CONFIGURED_POSE time="
                      << d->time << std::endl;
          }
          if (d)
          {
            observed_sim_time_initialized = true;
            last_observed_sim_time = d->time;
          }

          // running
          if (sim.run)
          {
            if (
              param::config.wait_for_lowcmd_before_physics == 1 &&
              (!param::lowcmd_connected.load() || !param::lowcmd_has_active_control.load())
            )
            {
              mj_forward(m, d);
              sim.speed_changed = true;
              nextLockstepCPU = mj::Simulate::Clock::now();
              continue;
            }
            bool stepped = false;

            // record cpu time at start of iteration
            const auto startCPU = mj::Simulate::Clock::now();

            // inject noise
            if (sim.ctrl_noise_std)
            {
              // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
              mjtNum rate = mju_exp(-m->opt.timestep / mju_max(sim.ctrl_noise_rate, mjMINVAL));
              mjtNum scale = sim.ctrl_noise_std * mju_sqrt(1 - rate * rate);

              for (int i = 0; i < m->nu; i++)
              {
                // update noise
                ctrlnoise[i] = rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

                // apply noise
                d->ctrl[i] = ctrlnoise[i];
              }
            }

            if (param::config.realtime_lockstep == 1)
            {
              const auto step_duration = std::chrono::duration_cast<mj::Simulate::Clock::duration>(
                Seconds(std::max<mjtNum>(m->opt.timestep, 0.001))
              );
              if (
                nextLockstepCPU.time_since_epoch().count() == 0 ||
                sim.speed_changed ||
                startCPU - nextLockstepCPU > Seconds(syncMisalign)
              )
              {
                nextLockstepCPU = startCPU;
                sim.speed_changed = false;
              }

              if (startCPU + std::chrono::microseconds(100) >= nextLockstepCPU)
              {
                // elastic band on base link
                if (param::config.enable_elastic_band == 1)
                {
                  if (elastic_band.enable_)
                  {
                    std::vector<double> x = {d->qpos[0], d->qpos[1], d->qpos[2]};
                    std::vector<double> dx = {d->qvel[0], d->qvel[1], d->qvel[2]};

                    elastic_band.Advance(x, dx);

                    d->xfrc_applied[param::config.band_attached_link] = elastic_band.f_[0];
                    d->xfrc_applied[param::config.band_attached_link + 1] = elastic_band.f_[1];
                    d->xfrc_applied[param::config.band_attached_link + 2] = elastic_band.f_[2];
                  }
                }

                // call mj_step
                mj_step(m, d);
                stepped = true;
                nextLockstepCPU += step_duration;
              }
            }
            else
            {
              // elapsed CPU and simulation time since last sync
              const auto elapsedCPU = startCPU - syncCPU;
              double elapsedSim = d->time - syncSim;

              // requested slow-down factor
              double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

              // misalignment condition: distance from target sim time is bigger than syncmisalign
              bool misaligned =
                  mju_abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

              // out-of-sync (for any reason): reset sync times, step
              if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
                  misaligned || sim.speed_changed)
              {
                // re-sync
                syncCPU = startCPU;
                syncSim = d->time;
                sim.speed_changed = false;

                // run single step, let next iteration deal with timing
                mj_step(m, d);
                stepped = true;
              }

              // in-sync: step until ahead of cpu
              else
              {
                bool measured = false;
                mjtNum prevSim = d->time;

                double refreshTime = simRefreshFraction / sim.refresh_rate;

                // step while sim lags behind cpu and within refreshTime
                while (Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                       mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime))
                {
                  // measure slowdown before first step
                  if (!measured && elapsedSim)
                  {
                    sim.measured_slowdown =
                        std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                    measured = true;
                  }

                  // elastic band on base link
                  if (param::config.enable_elastic_band == 1)
                  {
                    if (elastic_band.enable_)
                    {
                      std::vector<double> x = {d->qpos[0], d->qpos[1], d->qpos[2]};
                      std::vector<double> dx = {d->qvel[0], d->qvel[1], d->qvel[2]};

                      elastic_band.Advance(x, dx);

                      d->xfrc_applied[param::config.band_attached_link] = elastic_band.f_[0];
                      d->xfrc_applied[param::config.band_attached_link + 1] = elastic_band.f_[1];
                      d->xfrc_applied[param::config.band_attached_link + 2] = elastic_band.f_[2];
                    }
                  }

                  // call mj_step
                  mj_step(m, d);
                  stepped = true;

                  // break if reset
                  if (d->time < prevSim)
                  {
                    break;
                  }
                }
              }
            }

            // save current state to history buffer
            if (stepped)
            {
              sim.AddToHistory();
              emit_progress_if_due(progress_tracker, d, "viewer");
              emit_distance_marker_if_reached(progress_tracker, d);
              emit_fall_marker_if_needed(progress_tracker, d);
            }
          }

          // paused
          else
          {
            // run mj_forward, to update rendering and joint sliders
            mj_forward(m, d);
            sim.speed_changed = true;
          }
        }
      } // release std::lock_guard<std::mutex>
    }
  }
} // namespace

//-------------------------------------- physics_thread --------------------------------------------

void PhysicsThread(mj::Simulate *sim, const char *filename)
{
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr)
  {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
    if (m)
      d = mj_makeData(m);
    if (d)
    {
      sim->Load(m, d, filename);
      apply_initial_joint_pose();
      mj_forward(m, d);
      mujoco_data_initialized.store(true);

      // allocate ctrlnoise
      free(ctrlnoise);
      ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
      mju_zero(ctrlnoise, m->nu);
    }
    else
    {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  free(ctrlnoise);
  mj_deleteData(d);
  mj_deleteModel(m);

  exit(0);
}

void *UnitreeSdk2BridgeThread(void *arg)
{
  // Wait for mujoco data
  while (true)
  {
    if (m && d && mujoco_data_initialized.load())
    {
      std::cout << "Mujoco data is prepared" << std::endl;
      break;
    }
    usleep(500000);
  }

  unitree::robot::ChannelFactory::Instance()->Init(param::config.domain_id, param::config.interface);
  unitree_channel_ready.store(true);


  int body_id = mj_name2id(m, mjOBJ_BODY, "torso_link");
  if (body_id < 0) {
    body_id = mj_name2id(m, mjOBJ_BODY, "base_link");
  }
  param::config.band_attached_link = 6 * body_id;
  
  std::unique_ptr<UnitreeSDK2BridgeBase> interface = nullptr;
  if (m->nu > NUM_MOTOR_IDL_GO) {
    interface = std::make_unique<G1Bridge>(m, d);
  } else {
    interface = std::make_unique<Go2Bridge>(m, d);
  }
  interface->start();
  
  while (true)
  {
    sleep(1);
  }
}

mjModel* LoadHeadlessModel(const char* file)
{
  char loadError[kErrorLength] = "";
  mjModel* model = nullptr;
  const std::string filename = file ? file : "";
  if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".mjb")
  {
    model = mj_loadModel(filename.c_str(), nullptr);
    if (!model)
    {
      std::snprintf(loadError, sizeof(loadError), "could not load binary model");
    }
  }
  else
  {
    model = mj_loadXML(filename.c_str(), nullptr, loadError, kErrorLength);
  }
  if (!model)
  {
    std::cerr << loadError << std::endl;
  }
  else if (loadError[0])
  {
    std::cout << "Model compiled with warning: " << loadError << std::endl;
  }
  return model;
}

int RunHeadless()
{
  m = LoadHeadlessModel(param::config.robot_scene.c_str());
  if (!m)
  {
    return 1;
  }
  d = mj_makeData(m);
  if (!d)
  {
    mj_deleteModel(m);
    m = nullptr;
    return 1;
  }
  apply_initial_joint_pose();
  mj_forward(m, d);
  mujoco_data_initialized.store(true);

  std::thread unitree_thread(UnitreeSdk2BridgeThread, nullptr);
  unitree_thread.detach();

  const auto wall_start = std::chrono::steady_clock::now();
  ParkourProgressTracker progress_tracker;
  bool distance_logged = false;
  bool fall_logged = false;
  std::cout << "HEADLESS_READY seconds=" << param::config.headless_seconds
            << " walk_distance_marker=" << configured_walk_distance_marker()
            << " progress_log_interval=" << configured_progress_log_interval()
            << " wait_for_lowcmd=" << param::config.wait_for_lowcmd_before_physics
            << " depth_bridge=disabled" << std::endl;

  while (true)
  {
    const double elapsed_wall = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall_start
    ).count();
    if (elapsed_wall >= param::config.headless_seconds)
    {
      break;
    }

    if (
      param::config.wait_for_lowcmd_before_physics == 1 &&
      (!param::lowcmd_connected.load() || !param::lowcmd_has_active_control.load())
    )
    {
      mj_forward(m, d);
    }
    else
    {
      mj_step(m, d);
    }

    emit_progress_if_due(progress_tracker, d, "headless");
    if (!distance_logged && emit_distance_marker_if_reached(progress_tracker, d))
    {
      distance_logged = true;
      break;
    }
    if (!fall_logged && emit_fall_marker_if_needed(progress_tracker, d))
    {
      fall_logged = true;
      break;
    }

    std::this_thread::sleep_for(std::chrono::duration<double>(std::max<mjtNum>(m->opt.timestep, 0.001)));
  }

  if (!fall_logged)
  {
    std::cout << "NO_FALL_RESET" << std::endl;
  }
  mj_deleteData(d);
  mj_deleteModel(m);
  d = nullptr;
  m = nullptr;
  return distance_logged && !fall_logged ? 0 : 2;
}
//------------------------------------------ main --------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char *title, const char *msg);
static const char *rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char *msg)
{
  rosetta_error_msg = msg;
}
#endif

// user keyboard callback
void user_key_cb(GLFWwindow* window, int key, int scancode, int act, int mods) {
  if (act==GLFW_PRESS)
  {
    if(param::config.enable_elastic_band == 1) {
      if (key==GLFW_KEY_9) {
        elastic_band.enable_ = !elastic_band.enable_;
      } else if (key==GLFW_KEY_7 || key==GLFW_KEY_UP) {
        elastic_band.length_ -= 0.1;
      } else if (key==GLFW_KEY_8 || key==GLFW_KEY_DOWN) {
        elastic_band.length_ += 0.1;
      }
    }
    if(key==GLFW_KEY_BACKSPACE) {
      mj_resetData(m, d);
      apply_initial_joint_pose();
      mj_forward(m, d);
    }
  }
}

// run event loop
int main(int argc, char **argv)
{

  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg)
  {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version())
  {
    mju_error("Headers and library have different versions");
  }

  // scan for libraries in the plugin directory to load additional plugins
  scanPluginLibraries();

  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // Load simulation configuration
  std::filesystem::path proj_dir = std::filesystem::path(getExecutableDir()).parent_path();
  param::config.load_from_yaml((proj_dir / config_path_for_executable(argv[0])).string());
  param::helper(argc, argv);
  param::lowcmd_connected.store(false);
  param::lowcmd_has_active_control.store(false);
  mujoco_data_initialized.store(false);
  if(param::config.robot_scene.is_relative()) {
    param::config.robot_scene = proj_dir.parent_path() / param::config.robot_scene;
  }
  if (param::config.headless == 1)
  {
    return RunHeadless();
  }

  // simulate object encapsulates the UI
  auto sim = std::make_unique<mj::Simulate>(
    std::make_unique<mj::GlfwAdapter>(),
    &cam, &opt, &pert, /* is_passive = */ false);

  std::thread unitree_thread(UnitreeSdk2BridgeThread, nullptr);

  // start physics thread
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), param::config.robot_scene.c_str());
  std::unique_ptr<ParkourDepthBridge> depth_bridge;
  const bool depth_bridge_enabled = param::config.enable_depth_camera == 1
    && !env_flag_is_false(std::getenv("G1_PARKOUR_DEPTH_BRIDGE"));
  if (depth_bridge_enabled) {
    depth_bridge = std::make_unique<ParkourDepthBridge>(
      sim.get(),
      static_cast<mj::GlfwAdapter*>(sim->platform_ui.get())->window_,
      &m,
      &d,
      &unitree_channel_ready
    );
    const bool depth_bridge_started = depth_bridge->start();
    std::cout << "ParkourDepthBridge start status: "
              << (depth_bridge_started ? "started" : "failed")
              << std::endl;
  } else if (param::config.enable_depth_camera == 1) {
    std::cout << "ParkourDepthBridge disabled by G1_PARKOUR_DEPTH_BRIDGE=0; "
              << "use controller G1_PARKOUR_DEBUG_CONSTANT_DEPTH for control-only diagnostics."
              << std::endl;
  }
  // start simulation UI loop (blocking call)
  glfwSetKeyCallback(static_cast<mj::GlfwAdapter*>(sim->platform_ui.get())->window_,user_key_cb);
  sim->RenderLoop();
  if (depth_bridge) {
    depth_bridge->stop();
  }
  physicsthreadhandle.join();

  pthread_exit(NULL);
  return 0;
}
