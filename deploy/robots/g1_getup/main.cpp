#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
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
        std::cout << "  f : enter FixStand\n";
        std::cout << "  g : enter GetUp control\n";
        std::cout << "  p : return to Passive mode\n";
        return;
    }

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R2 + Y] to start get-up control.\n";
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
    std::cout << "     G1-29dof Get-Up Controller \n";

    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());
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
