#!/usr/bin/env python
import sys, gym, time

#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#

env = gym.make("LunarLander-v2" if len(sys.argv) < 2 else sys.argv[1])

if not hasattr(env.action_space, "n"):
    raise Exception("Keyboard agent only supports discrete action spaces")

ACTIONS = env.action_space.n
SKIP_CONTROL = 0  # Use previous control decision SKIP_CONTROL times, that's how you
# can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key == 0xFF0D:
        human_wants_restart = True

    if key == 32:
        human_sets_pause = not human_sets_pause

    a = int(key - ord("0"))
    if a <= 0 or a >= ACTIONS:
        return

    human_agent_action = a


def key_release(key, mod):
    global human_agent_action

    a = int(key - ord("0"))
    if a <= 0 or a >= ACTIONS:
        return

    if human_agent_action == a:
        human_agent_action = 0


env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    done = False
    while not human_wants_restart and not done:
        if not skip:
            # print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        if r != 0:
            print("reward %0.3f" % r)

        total_reward += r

        while True:
            if not env.render():
                return False
            time.sleep(0.05)

            if not human_sets_pause:
                break

    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
    return True


print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while rollout(env):
    pass
