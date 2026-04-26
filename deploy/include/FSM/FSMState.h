#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"
#include <algorithm>
#include <cctype>

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                registered_checks.emplace_back(
                    std::make_pair(
                        [func]()->bool{ return func(FSMState::lowstate->joystick); },
                        fsm_id
                    )
                );
            }
        }

        if (param::keyboard_control)
        {
            auto keyboard_transitions = param::config["FSM"][state_string]["keyboard_transitions"];
            if (keyboard_transitions)
            {
                auto transition_map = keyboard_transitions.as<std::map<std::string, std::string>>();
                for (auto it = transition_map.begin(); it != transition_map.end(); ++it)
                {
                    const std::string target_fsm = it->first;
                    if (!FSMStringMap.right.count(target_fsm))
                    {
                        spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                        continue;
                    }

                    const int fsm_id = FSMStringMap.right.at(target_fsm);
                    const std::string condition = it->second;
                    registered_checks.emplace_back(
                        std::make_pair(
                            [condition]()->bool { return keyboard_condition(condition); },
                            fsm_id
                        )
                    );
                }
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        if(keyboard) keyboard->update();
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;

private:
    static std::string normalize_condition(std::string value)
    {
        value.erase(
            std::remove_if(
                value.begin(),
                value.end(),
                [](unsigned char ch) { return std::isspace(ch) != 0; }
            ),
            value.end()
        );
        std::transform(
            value.begin(),
            value.end(),
            value.begin(),
            [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); }
        );
        return value;
    }

    static bool keyboard_condition(const std::string& raw_condition)
    {
        if (!keyboard)
        {
            return false;
        }

        const auto or_pos = raw_condition.find("||");
        if (or_pos != std::string::npos)
        {
            return keyboard_condition(raw_condition.substr(0, or_pos))
                || keyboard_condition(raw_condition.substr(or_pos + 2));
        }

        std::string condition = normalize_condition(raw_condition);
        bool require_on_pressed = false;
        bool require_on_released = false;

        static const std::string on_pressed_suffix = ".on_pressed";
        static const std::string on_released_suffix = ".on_released";

        if (condition.size() >= on_pressed_suffix.size() &&
            condition.compare(condition.size() - on_pressed_suffix.size(), on_pressed_suffix.size(), on_pressed_suffix) == 0)
        {
            require_on_pressed = true;
            condition.erase(condition.size() - on_pressed_suffix.size());
        }
        else if (condition.size() >= on_released_suffix.size() &&
                 condition.compare(condition.size() - on_released_suffix.size(), on_released_suffix.size(), on_released_suffix) == 0)
        {
            require_on_released = true;
            condition.erase(condition.size() - on_released_suffix.size());
        }

        if (require_on_pressed && !keyboard->on_pressed)
        {
            return false;
        }
        if (require_on_released && !keyboard->on_released)
        {
            return false;
        }

        return normalize_condition(keyboard->key()) == condition;
    }
};
