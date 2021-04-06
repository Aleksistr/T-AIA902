import numpy as np
import gym
import random
from os import system, name
from time import sleep

# Define function to clear console window.
def clear():
    # Clear on Windows.
    if name == 'nt':
        _ = system('cls')

    # Clear on Mac and Linux. (os.name is 'posix')
    else:
        _ = system('clear')


env = gym.make("Taxi-v3")

# Get Game action and state size
action_size = env.action_space.n
state_size = env.observation_space.n

# Build the qtable
qtable = np.zeros((state_size, action_size))

# Define hyperparameters
total_episodes = 50000 #Total "game" episodes
total_test_episodes = 100 #total test episodes
max_steps = 99 #Max step per episodes for an agent (a taxi)

learning_rate = 0.8 #Learning rate
gamma = 0.618 #Discount rate

# exploration parameters

max_epsilon = 1.0 # Explo rate probability at start
epsilon = max_epsilon
min_epsilon = 0.01 # Min exploration rate probability
decay_rate = 0.01 #Exponencial decay rate for exploration prob

# Implement qLearning algorithm

for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False

    # Set total penalities and reward
    penalties, reward = 0, 0

    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma *
                                                                         np.max(qtable[new_state, :]) - qtable[
                                                                             state, action])

        if reward == -10:
            penalties += 1
        # Our new state is state
        state = new_state

        # If done : finish episode
        if done == True:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

env.reset()
rewards = []

total_epochs, total_penalties = 0, 0

for episode in range(total_test_episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    step = 0
    done = False
    # print("****************************************************")
    # print("EPISODE ", episode)

    for step in range(max_steps):
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state, :])
        new_state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1
        # Clear terminal output
        clear()
        # display result
        env.render()
        print(f"Step n° {epochs}")
        print(f"State: {state}")
        print(f"Action done: {action}")
        print(f"Reward: {reward}")
        # Wait little bit to able user to read result
        sleep(0.5)



        if done:
            total_epochs += epochs
            total_penalties += penalties
            # print ("Score", total_rewards)
            break
        state = new_state
env.close()

print(f"Result after {total_episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / total_episodes}")
print(f"Average penalties per episode: {total_penalties / total_episodes}")