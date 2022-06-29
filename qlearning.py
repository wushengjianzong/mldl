"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible


N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['L', 'R']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 10   # maximum episodes
FRESH_TIME = 0.1    # fresh time for one move


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action = np.random.choice(ACTIONS)
    else:   # act greedy
        action = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action


def feedback(state, action):
    # This is how agent will interact with the environment
    if action == 'R':    # move right
        if state == N_STATES - 2:   # terminate
            _state = -1
            reward = 1
        else:
            _state = state + 1
            reward = 0
    else:   # move left
        reward = 0
        if state == 0:
            _state = state  # reach the wall
        else:
            _state = state - 1
    return _state, reward


def print_environment(state, episode, step):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if state == -1:
        print(f'Episode {episode}: total_steps = {step}')
        time.sleep(1)
    else:
        env_list[state] = 'o'
        print(''.join(env_list))
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = pd.DataFrame(
        np.zeros((N_STATES, len(ACTIONS))),     # q_table initial values
        columns=ACTIONS,    # actions's name
    )
    for episode in range(MAX_EPISODES):
        step = 0
        state = 0
        is_terminated = False
        print_environment(state, episode, step)
        while not is_terminated:
            action = choose_action(state, q_table)
            _state, reward = feedback(state, action)  # take action & get next state and reward
            q_predict = q_table.loc[state, action]
            if _state != -1:
                q_target = reward + GAMMA * q_table.iloc[_state, :].max()   # next state is not terminal
            else:
                q_target = reward     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[state, action] += ALPHA * (q_target - q_predict)  # update
            state = _state  # move to next state

            print_environment(state, episode, step+1)
            step += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('Q-table:')
    print(q_table)