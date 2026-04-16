#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include <atomic>
#include <csignal>
#include <unistd.h>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

namespace
{

std::atomic<bool> shutdown_requested = false;

void request_shutdown(int)
{
    shutdown_requested = true;
}

void print_control_help()
{
    if (param::keyboard_control)
    {
        std::cout << "Keyboard mode enabled. Keep this terminal focused.\n";
        std::cout << "  f : enter FixStand\n";
        std::cout << "  v : enter Velocity control\n";
        std::cout << "  p : return to Passive mode\n";
        std::cout << "  w/s : move forward/backward\n";
        std::cout << "  a/d : strafe left/right\n";
        std::cout << "  q/e : turn left/right\n";
        std::cout << "Release movement keys to stop.\n";
        return;
    }

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R2 + A] to start controlling the robot.\n";
}

}

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);
    std::signal(SIGINT, request_shutdown);
    std::signal(SIGTERM, request_shutdown);
    if (param::keyboard_control)
    {
        if (!isatty(STDIN_FILENO))
        {
            spdlog::critical("--keyboard requires an interactive terminal on stdin.");
            return -1;
        }
    }

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());
    print_control_help();

    init_fsm_state();
    if (param::keyboard_control)
    {
        FSMState::keyboard = std::make_shared<Keyboard>();
        param::active_keyboard = FSMState::keyboard;
    }

    {
        // Initialize FSM
        auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
        fsm->start();

        while (!shutdown_requested)
        {
            sleep(1);
        }
    }

    param::active_keyboard.reset();
    FSMState::keyboard.reset();
    
    return 0;
}
