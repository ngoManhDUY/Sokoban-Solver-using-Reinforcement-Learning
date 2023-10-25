import Sokoban_env
from Sokoban_env import Sokoban_v2
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import pandas as pd
import pygame 
import os

 #Q-learning algorithm
def Q_learning(env,learning_rate = 0.1, discount_factor = 0.9, epsilon = 0.6, num_episodes = 5000):
    # Setting Q
    state_space = env.num_states.n
    action_space = env.action_space.n

    q_table = np.zeros((state_space, action_space))

    eva = []

    for episode in range(num_episodes):
        print(f'---- Episode: {episode}')
        state = env.reset()
        done = False
        Return = 0

        while not done:
            # Choose an action using epsilon-greedy policy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[np.argmax(q_table[state, :])])  # Exploit     

            next_state, reward, done, _ = env.step(action)

            # Update Q-table using the Q-learning formula
            q_table[state, action] = (1 - learning_rate) * q_table[state, action] + \
                                    learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]))

            state = next_state
            Return += reward
        eva.append(Return)
        display.clear_output()
    return q_table, eva


# SARSA
def sarsa(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, num_episodes = 1000):
    state_space = env.num_states.n
    action_space = env.action_space.n

    # Initialize the Q-table 
    q_table = np.zeros((state_space, action_space))
    eva = []
    Return = 0

    # SARSA algorithm
    for episode in range(num_episodes):
        print(f'---- Episode: {episode}')
        state = env.reset()
        done = False

        # Choose an action using epsilon-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state, :])  # Exploit

        while not done:
            Return = 0
            next_state, reward, done, _ = env.step(action)

            # Choose the next action using epsilon-greedy policy
            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()  # Explore
            else:
                next_action = np.argmax(q_table[next_state, :])  # Exploit

            # Update Q-table using SARSA formula
            q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + discount_factor * q_table[next_state, next_action] - q_table[state, action]
            )

            state = next_state
            action = next_action
            Return += reward
        eva.append(Return)
        display.clear_output()
    return q_table, eva

def first_visit_monte_carlo(env, num_episodes, epsilon=0.1):
    state_space = env.num_states.n
    action_space = env.action_space.n

    q_table = np.zeros((state_space, action_space))
    returns = np.zeros((state_space, action_space))
    counts = np.zeros((state_space, action_space))
    
    eva = []

    def epsilon_greedy(state):
        if np.random.uniform(0, 1) < epsilon:
            return env.action_space.sample()  # Explore
        else:
            return np.argmax(q_table[state, :])  # Exploit

    for episode in range(num_episodes):
        print(f'---- Episode: {episode}')
        episode_data = []
        state = env.reset()
        done = False

        while not done:
            action = epsilon_greedy(state)
            next_state, reward, done, _ = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state

        # Update Q-values using Monte Carlo method
        G = 0
        for t in reversed(range(len(episode_data))):
            state, action, reward = episode_data[t]
            G = reward + G
            if (state, action) not in episode_data[:t]:
                counts[state, action] += 1
                returns[state, action] += G
                q_table[state, action] = returns[state, action] / counts[state, action]
        eva.append(G)
    return q_table, eva


def running_alg(env, q_table):
    #Evaluate the trained Q-table
    
    os.environ['SDL_AUDIODRIVER'] = 'alsa'
    pygame.mixer.init()
    pygame.mixer.music.load('/home/duy/Documents/Reinforcement Learning/Sokoban RL/Code/surface/troll.mp3')
    
    total_rewards = 0
    num_episodes_eval = 1

    display.clear_output(wait=True)
    env.reset()
    state = env.reset()
    done = False

    while not done:
        plt.imshow(env.render())
        action = np.argmax(q_table[state, :])
        next_state, reward, done, _ = env.step(action)

        display.display(plt.gcf())
        display.clear_output(wait=True)
        total_rewards += reward
        state = next_state
    plt.imshow(env.render())
    display.display(plt.gcf())
    display.clear_output(wait=True)
    if done:
        pygame.mixer.music.play()

    average_reward = total_rewards / num_episodes_eval
    print(f"Reward over {num_episodes_eval} episodes: {average_reward:}")
    

    
def eva_graph(eva):
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    
    plt.plot(eva)
    