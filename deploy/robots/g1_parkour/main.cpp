#include "FSM/CtrlFSM.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_Passive.h"
#include "FSM/State_RLBase.h"
#include <unistd.h>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

namespace
{

void print_control_help()
{
    if (param::keyboard_control)
    {
        std::cout << "Keyboard mode enabled. Keep this terminal focused.\n";
        if (param::sim_loopback_interactive)
        {
            std::cout << "  Loopback default: starts in Parkour idle-hold; press w/up to walk.\n";
        }
        std::cout << "  f : enter FixStand from Passive/FSM diagnostics\n";
        std::cout << "  k : enter Parkour control (when in FixStand)\n";
        std::cout << "  p : return to Passive mode\n";
        std::cout << "  w/up : set forward speed to " << param::sim_command_x << " m/s\n";
        std::cout << "  +/= / - : adjust forward speed by the policy keyboard step\n";
        std::cout << "  a/left/q : turn left\n";
        std::cout << "  d/right/e : turn right\n";
        std::cout << "  c : stop yaw turn\n";
        std::cout << "  s/down/x/space : return to idle-hold command\n";
        return;
    }

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R2 + X] to start parkour control.\n";
    if (param::sim_autostart_parkour)
    {
        std::cout << "Simulation autostart enabled: entering Parkour on loopback with command ["
                  << param::sim_command_x << ", "
                  << param::sim_command_y << ", "
                  << param::sim_command_yaw << "].\n";
    }
}

}

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    auto vm = param::helper(argc, argv);
    const std::string network = vm["network"].as<std::string>();
    if (param::sim_autostart_parkour)
    {
        if (network != "lo")
        {
            spdlog::critical("--sim-autostart-parkour requires --network=lo for loopback-only simulation safety.");
            return -1;
        }
        param::config["FSM"]["start_state"] = "Parkour";
        spdlog::info(
            "Simulation autostart enabled: start_state=Parkour command=[{}, {}, {}]",
            param::sim_command_x,
            param::sim_command_y,
            param::sim_command_yaw
        );
    }
    else if (param::sim_loopback_interactive)
    {
        param::config["FSM"]["start_state"] = "Parkour";
        spdlog::info(
            "Loopback interactive simulation enabled: start_state=Parkour, keyboard cruise speed={} m/s, idle command x={} m/s",
            param::sim_command_x,
            param::sim_keyboard_idle_command_x
        );
    }
    if (param::keyboard_control)
    {
        if (!isatty(STDIN_FILENO))
        {
            spdlog::critical("--keyboard requires an interactive terminal on stdin.");
            return -1;
        }
        FSMState::keyboard = std::make_shared<Keyboard>();
        param::active_keyboard = FSMState::keyboard;
    }

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Parkour Controller \n";

    unitree::robot::ChannelFactory::Instance()->Init(0, network);
    print_control_help();

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5;
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }

    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    while (true)
    {
        sleep(1);
    }

    return 0;
}
