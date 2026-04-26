#include "ParkourObservations.h"
#include "ParkourArticulation.h"
#include "FSM/State_RLBase.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    reset_on_tick_rewind_ = cfg["reset_on_tick_rewind"].as<bool>(false);
    ensureParkourObservationRegistration();
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    const auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::ParkourArticulation<LowState_t::SharedPtr>>(
            FSMState::lowstate,
            param::sim_autostart_parkour
        )
    );
    depth_provider_ = std::make_unique<ParkourDepthProvider>(deploy_cfg);
    if (depth_provider_->enabled()) {
        depth_provider_->initialize(env->robot.get());
    }
    env->alg = std::make_unique<isaaclab::ParkourOrtRunner>(
        (policy_dir / "exported" / "0-depth_encoder.onnx").string(),
        (policy_dir / "exported" / "actor.onnx").string()
    );

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

namespace
{

std::string json_vector(const std::vector<float>& values)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << std::setprecision(9) << values[i];
    }
    os << "]";
    return os.str();
}

std::string json_vector3(const Eigen::Vector3f& values)
{
    std::ostringstream os;
    os << "["
       << std::setprecision(9) << values[0] << ","
       << std::setprecision(9) << values[1] << ","
       << std::setprecision(9) << values[2] << "]";
    return os.str();
}

std::string json_stats(const isaaclab::ParkourOrtRunner::VectorStats& stats)
{
    std::ostringstream os;
    os << "{\"min\":" << std::setprecision(9) << stats.min
       << ",\"max\":" << std::setprecision(9) << stats.max
       << ",\"mean\":" << std::setprecision(9) << stats.mean
       << "}";
    return os.str();
}

uint32_t lowstate_tick(const LowState_t::SharedPtr& lowstate)
{
    std::lock_guard<std::mutex> lowstate_lock(lowstate->mutex_);
    return lowstate->msg_.tick();
}

std::optional<int> extract_json_int(const std::string& line, const std::string& key)
{
    const auto key_pos = line.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto colon_pos = line.find(':', key_pos);
    if (colon_pos == std::string::npos) {
        return std::nullopt;
    }
    const char* begin = line.c_str() + colon_pos + 1;
    char* end = nullptr;
    const long value = std::strtol(begin, &end, 10);
    if (end == begin) {
        return std::nullopt;
    }
    return static_cast<int>(value);
}

std::optional<std::vector<float>> extract_json_float_array(const std::string& line, const std::string& key)
{
    const auto key_pos = line.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto array_begin = line.find('[', key_pos);
    if (array_begin == std::string::npos) {
        return std::nullopt;
    }
    const auto array_end = line.find(']', array_begin);
    if (array_end == std::string::npos || array_end <= array_begin) {
        return std::nullopt;
    }

    std::vector<float> values;
    const char* cursor = line.c_str() + array_begin + 1;
    const char* end = line.c_str() + array_end;
    while (cursor < end) {
        while (cursor < end && (std::isspace(static_cast<unsigned char>(*cursor)) || *cursor == ',')) {
            ++cursor;
        }
        if (cursor >= end) {
            break;
        }
        char* next = nullptr;
        const float value = std::strtof(cursor, &next);
        if (next == cursor) {
            return std::nullopt;
        }
        values.push_back(value);
        cursor = next;
    }
    return values;
}

std::string json_uint_vector(const std::deque<uint32_t>& values)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << values[i];
    }
    os << "]";
    return os.str();
}

std::string json_int_vector(const std::vector<int64_t>& values)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << values[i];
    }
    os << "]";
    return os.str();
}

std::string json_bool(bool value)
{
    return value ? "true" : "false";
}

} // namespace

void State_RLBase::open_gait_record_if_requested()
{
    close_gait_record();
    if (param::gait_record_jsonl.empty()) {
        return;
    }
    std::filesystem::path output_path = param::gait_record_jsonl;
    if (output_path.is_relative()) {
        output_path = param::proj_dir / output_path;
    }
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    gait_record_stream_.open(output_path, std::ios::out | std::ios::trunc);
    if (!gait_record_stream_) {
        spdlog::error("Failed to open parkour gait record JSONL: {}", output_path.string());
        return;
    }
    spdlog::info(
        "Recording parkour gait parity JSONL to {} every {} policy step(s).",
        output_path.string(),
        param::gait_record_every
    );
}

void State_RLBase::close_gait_record()
{
    if (gait_record_stream_.is_open()) {
        gait_record_stream_.close();
    }
}


void State_RLBase::load_gait_replay_if_requested()
{
    gait_replay_samples_.clear();
    gait_replay_loaded_ = false;
    last_replay_mode_ = "off";
    last_replay_source_step_ = -1;
    last_replay_applied_ = false;

    if (param::gait_replay_jsonl.empty() || param::gait_replay_mode == "off") {
        return;
    }
    std::filesystem::path input_path = param::gait_replay_jsonl;
    if (input_path.is_relative()) {
        input_path = param::proj_dir / input_path;
    }
    std::ifstream input(input_path);
    if (!input) {
        throw std::runtime_error("Failed to open --gait-replay-jsonl: " + input_path.string());
    }

    std::string line;
    int row_index = 0;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        if (row_index++ < param::gait_replay_start_step) {
            continue;
        }
        if (param::gait_replay_max_steps > 0
            && static_cast<int>(gait_replay_samples_.size()) >= param::gait_replay_max_steps) {
            break;
        }

        GaitReplaySample sample;
        sample.source_step = extract_json_int(line, "step").value_or(row_index - 1);
        if (const auto values = extract_json_float_array(line, "raw_action_policy_order")) {
            sample.raw_action_policy_order = *values;
        }
        if (const auto values = extract_json_float_array(line, "target_q_deploy_order")) {
            sample.target_q_deploy_order = *values;
        }
        const bool has_requested_mode =
            (param::gait_replay_mode == "raw-action-policy" && !sample.raw_action_policy_order.empty())
            || (param::gait_replay_mode == "target-q-deploy" && !sample.target_q_deploy_order.empty());
        if (has_requested_mode) {
            gait_replay_samples_.push_back(std::move(sample));
        }
    }

    gait_replay_loaded_ = !gait_replay_samples_.empty();
    if (!gait_replay_loaded_) {
        throw std::runtime_error(
            "--gait-replay-jsonl did not contain requested replay field for mode " + param::gait_replay_mode);
    }
    spdlog::info(
        "Loaded {} G1 parkour gait replay samples from {} mode={} start={} max={}",
        gait_replay_samples_.size(),
        input_path.string(),
        param::gait_replay_mode,
        param::gait_replay_start_step,
        param::gait_replay_max_steps
    );
}

const State_RLBase::GaitReplaySample* State_RLBase::current_gait_replay_sample() const
{
    if (!gait_replay_loaded_ || param::gait_replay_mode == "off") {
        return nullptr;
    }
    const size_t sample_index = gait_record_step_;
    if (sample_index >= gait_replay_samples_.size()) {
        return nullptr;
    }
    return &gait_replay_samples_[sample_index];
}

std::vector<float> State_RLBase::target_q_to_raw_action(const std::vector<float>& target_q) const
{
    const auto action_cfg = env->cfg["actions"]["JointPositionAction"];
    const auto scale = action_cfg["scale"].as<std::vector<float>>();
    const auto offset = action_cfg["offset"].as<std::vector<float>>();
    if (target_q.size() != scale.size() || target_q.size() != offset.size()) {
        throw std::runtime_error("target-q replay vector size does not match deploy action scale/offset size.");
    }
    std::vector<float> raw_action(target_q.size(), 0.0f);
    for (size_t i = 0; i < target_q.size(); ++i) {
        if (std::abs(scale[i]) < 1.0e-8f) {
            throw std::runtime_error("target-q replay cannot invert zero action scale.");
        }
        raw_action[i] = (target_q[i] - offset[i]) / scale[i];
    }
    return raw_action;
}

bool State_RLBase::apply_gait_replay_raw_action()
{
    if (param::gait_replay_mode != "raw-action-policy") {
        return false;
    }
    const auto* sample = current_gait_replay_sample();
    if (sample == nullptr) {
        last_replay_applied_ = false;
        last_replay_mode_ = param::gait_replay_mode;
        last_replay_source_step_ = -1;
        return false;
    }
    auto deploy_action = isaaclab::ParkourOrtRunner::policy_order_action_to_deploy_order(sample->raw_action_policy_order);
    env->action_manager->process_action(deploy_action);
    last_replay_applied_ = true;
    last_replay_mode_ = param::gait_replay_mode;
    last_replay_source_step_ = sample->source_step;
    return true;
}

bool State_RLBase::apply_gait_replay_target_q()
{
    if (param::gait_replay_mode != "target-q-deploy") {
        return false;
    }
    const auto* sample = current_gait_replay_sample();
    if (sample == nullptr) {
        last_replay_applied_ = false;
        last_replay_mode_ = param::gait_replay_mode;
        last_replay_source_step_ = -1;
        return false;
    }
    env->action_manager->process_action(target_q_to_raw_action(sample->target_q_deploy_order));
    last_replay_applied_ = true;
    last_replay_mode_ = param::gait_replay_mode;
    last_replay_source_step_ = sample->source_step;
    return true;
}

void State_RLBase::record_gait_sample(float blend_alpha)
{
    const auto tick = lowstate_tick(lowstate);
    lowstate_tick_history_.push_back(tick);
    while (lowstate_tick_history_.size() > 8) {
        lowstate_tick_history_.pop_front();
    }

    std::vector<int64_t> tick_deltas;
    int repeated_frame_count = 0;
    int skipped_tick_count = 0;
    const int64_t expected_tick_delta = static_cast<int64_t>(std::llround(env->step_dt / 1.0e-3f));
    for (size_t i = 1; i < lowstate_tick_history_.size(); ++i) {
        const auto prev = static_cast<int64_t>(lowstate_tick_history_[i - 1]);
        const auto curr = static_cast<int64_t>(lowstate_tick_history_[i]);
        const int64_t delta = curr - prev;
        tick_deltas.push_back(delta);
        if (delta == 0) {
            ++repeated_frame_count;
        }
        if (expected_tick_delta > 0 && delta > expected_tick_delta + 2) {
            ++skipped_tick_count;
        }
    }

    const int every = std::max(param::gait_record_every, 1);
    const bool should_write = gait_record_stream_.is_open()
        && gait_record_step_ % static_cast<size_t>(every) == 0;
    if (!should_write) {
        ++gait_record_step_;
        return;
    }

    auto* runner = dynamic_cast<isaaclab::ParkourOrtRunner*>(env->alg.get());
    const auto raw_action_deploy = env->action_manager->action();
    const auto processed_action_deploy = env->action_manager->processed_actions();
    const auto* replay_sample = current_gait_replay_sample();
    std::vector<float> raw_action_policy = runner ? runner->last_policy_order_action() : std::vector<float>{};
    if (last_replay_applied_ && last_replay_mode_ == "raw-action-policy" && replay_sample != nullptr) {
        raw_action_policy = replay_sample->raw_action_policy_order;
    }
    const auto depth_stats = runner ? runner->last_depth_stats() : isaaclab::ParkourOrtRunner::VectorStats{};
    const auto proprio_stats = runner ? runner->last_proprio_stats() : isaaclab::ParkourOrtRunner::VectorStats{};
    const auto policy_wall_time = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - policy_wall_start_time_).count();

    const std::vector<float> joint_pos(
        env->robot->data.joint_pos.data(),
        env->robot->data.joint_pos.data() + env->robot->data.joint_pos.size()
    );
    const std::vector<float> joint_vel(
        env->robot->data.joint_vel.data(),
        env->robot->data.joint_vel.data() + env->robot->data.joint_vel.size()
    );
    const auto* parkour_robot = dynamic_cast<unitree::ParkourArticulation<LowState_t::SharedPtr>*>(env->robot.get());
    const auto joint_vel_sensor = parkour_robot ? parkour_robot->last_sensor_joint_vel() : std::vector<float>{};
    const auto joint_vel_fdiff = parkour_robot ? parkour_robot->last_finite_diff_joint_vel() : std::vector<float>{};
    const std::string joint_vel_source = parkour_robot ? parkour_robot->joint_vel_source() : param::joint_vel_source;
    const std::vector<float> command{
        param::sim_observed_command_x.load(),
        param::sim_observed_command_y.load(),
        param::sim_observed_command_yaw.load(),
    };

    gait_record_stream_
        << "{\"source\":\"cpp_dds_ctrl\""
        << ",\"step\":" << gait_record_step_
        << ",\"policy_step\":" << gait_record_step_
        << ",\"elapsed_seconds\":" << std::setprecision(9) << (env->episode_length * env->step_dt)
        << ",\"sim_time\":" << std::setprecision(9) << (static_cast<double>(tick) * 1.0e-3)
        << ",\"policy_wall_time\":" << std::setprecision(9) << policy_wall_time
        << ",\"policy_tick_sync\":" << json_bool(param::policy_tick_sync)
        << ",\"lowstate_tick\":" << tick
        << ",\"startup_blend_alpha\":" << std::setprecision(9) << blend_alpha
        << ",\"command\":" << json_vector(command)
        << ",\"base_ang_vel\":" << json_vector3(env->robot->data.root_ang_vel_b)
        << ",\"projected_gravity\":" << json_vector3(env->robot->data.projected_gravity_b)
        << ",\"joint_pos_deploy_order\":" << json_vector(joint_pos)
        << ",\"joint_vel_deploy_order\":" << json_vector(joint_vel)
        << ",\"joint_vel_source\":\"" << joint_vel_source << "\""
        << ",\"joint_vel_sensor_deploy_order\":" << json_vector(joint_vel_sensor)
        << ",\"joint_vel_fdiff_deploy_order\":" << json_vector(joint_vel_fdiff)
        << ",\"raw_action_policy_order\":" << json_vector(raw_action_policy)
        << ",\"raw_action_deploy_order\":" << json_vector(raw_action_deploy)
        << ",\"applied_action_deploy_order\":" << json_vector(raw_action_deploy)
        << ",\"target_q_deploy_order\":" << json_vector(processed_action_deploy)
        << ",\"replay_mode\":\"" << last_replay_mode_ << "\""
        << ",\"replay_source_step\":" << last_replay_source_step_
        << ",\"replay_applied\":" << json_bool(last_replay_applied_)
        << ",\"history_freshness\":{"
        << "\"lowstate_ticks\":" << json_uint_vector(lowstate_tick_history_)
        << ",\"tick_deltas\":" << json_int_vector(tick_deltas)
        << ",\"expected_tick_delta\":" << expected_tick_delta
        << ",\"repeated_frame_count\":" << repeated_frame_count
        << ",\"skipped_tick_count\":" << skipped_tick_count
        << ",\"last_action_age_steps\":0"
        << ",\"last_action_reused\":false"
        << ",\"reset_epoch\":" << reset_epoch_
        << "}"
        << ",\"depth_stats\":" << json_stats(depth_stats)
        << ",\"proprio_stats\":" << json_stats(proprio_stats)
        << "}\n";
    gait_record_stream_.flush();
    ++gait_record_step_;
}

void State_RLBase::run()
{
    std::vector<float> action;
    {
        std::lock_guard<std::mutex> env_lock(env_mutex_);
        action = env->action_manager->processed_actions();
    }
    static std::atomic<bool> logged_once{false};
    if (!logged_once.exchange(true)) {
        const auto minmax = std::minmax_element(action.begin(), action.end());
        std::cout << "[parkour debug] processed_action[min,max]=["
                  << (minmax.first != action.end() ? *minmax.first : 0.0f) << ","
                  << (minmax.second != action.end() ? *minmax.second : 0.0f) << "] head=";
        for (size_t i = 0; i < std::min<size_t>(10, action.size()); ++i) {
            std::cout << action[i] << (i + 1 < std::min<size_t>(10, action.size()) ? "," : "");
        }
        std::vector<float> motor_order(action.size(), 0.0f);
        for (size_t i = 0; i < action.size(); ++i) {
            const auto motor_index = static_cast<size_t>(env->robot->data.joint_ids_map[i]);
            if (motor_index < motor_order.size()) {
                motor_order[motor_index] = action[i];
            }
        }
        std::cout << " motor_order=";
        for (size_t i = 0; i < motor_order.size(); ++i) {
            std::cout << motor_order[i] << (i + 1 < motor_order.size() ? "," : "");
        }
        std::cout << std::endl;
    }
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
