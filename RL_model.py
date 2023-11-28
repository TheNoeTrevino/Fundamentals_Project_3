import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = np.zeros((state_size, action_size))

    def select_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)  # Explore
        else:
            return np.argmax(self.q_table[state, :])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        self.q_table[state, action] += self.learning_rate * (
            reward + self.discount_factor * self.q_table[next_state, best_next_action] - self.q_table[state, action]
        )

class TaxiEnvironment(gym.Env):
    def __init__(self):
        super(TaxiEnvironment, self).__init__()

        self.grid_size = 5
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = gym.spaces.Discrete(self.grid_size * self.grid_size)  # 5x5 grid

        # Set hazard area to top middle
        self.hazardous_area = (0, 2)

        # Initialize customers
        self.customer_positions = []
        self.customer_premium = []

        # Other parameters...
        self.discount_factor = 0.9
        self.max_customers = 2
        self.current_customers = 0

        # Initialize the state
        self.state = None
        self.reset()

    def reset(self):
        # Reset the environment for a new episode
        self.state = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        self.generate_customers()
        self.current_customers = 0
        return self.state

    def generate_customers(self):
        # Define specific customer positions: Top row - second and fourth space, Bottom row - first and last space
        top_row_positions = [(0, 1), (0, 3)]
        bottom_row_positions = [(4, 0), (4, 4)]

        # Randomly choose one position from each row
        self.customer_positions = [np.random.choice(top_row_positions), np.random.choice(bottom_row_positions)]
        self.customer_premium = [np.random.choice([True, False]), np.random.choice([True, False])]

    def step(self, action):
        # Take a step in the environment based on the given action
        x, y = self.state

        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.grid_size - 1, y + 1)

        # Check if the new position is outside the grid, if so, stay in the current position
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            x, y = self.state

        self.state = (x, y)

        # Define rewards based on the new state
        reward, done = self.calculate_reward_and_check_done()

        return self.state, reward, done, {}

    def calculate_reward_and_check_done(self):
        x, y = self.state

        # Define rewards and penalties based on the current state
        reward = -0.5  # Live-in reward as an incentive to encourage shorter paths

        # Check if the taxi entered the space where a customer is
        for i, customer_pos in enumerate(self.customer_positions):
            if (x, y) == customer_pos:
                if not self.customer_premium[i]:
                    reward += 20  # Regular customer pickup reward
                else:
                    reward += 30  # Premium customer pickup reward
                self.current_customers += 1
                self.customer_positions[i] = (-1, -1)  # Mark the customer as picked up

        # Check if all customers are picked up
        done = self.current_customers == self.max_customers

        # Check for hazardous area
        if (x, y) == self.hazardous_area:
            reward -= 10  # Negative reward for hazardous area
            done = True  # End episode if the taxi enters the hazardous area

        return reward, done

    def render(self, agent):
        # Render the current state of the environment with arrows indicating the path
        q_table = agent.q_table  # Get the Q-table from the agent
        for i in range(self.grid_size):
            row = ''
            for j in range(self.grid_size):
                if (i, j) == self.state:
                    row += 'T '  # Taxi position
                elif (i, j) == self.hazardous_area:
                    row += 'H '  # Hazardous area
                elif (i, j) in self.customer_positions:
                    row += 'C '  # Customer position
                else:
                    action_arrow = self.get_action_arrow((i, j), q_table)
                    row += action_arrow + ' '  # Arrow indicating the action
            print(row)
        print()

    def get_action_arrow(self, state, q_table):
        # Get the action arrow based on the highest Q-value for the given state
        action = np.argmax(q_table[state, :])

        if action == 0:  # Up
            return '^'
        elif action == 1:  # Down
            return 'v'
        elif action == 2:  # Left
            return '<'
        elif action == 3:  # Right
            return '>'

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration-exploitation trade-off
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    env = TaxiEnvironment()
    state_size = env.observation_space.n
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    batch_size = 32
    num_episodes = 1000

    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time in range(500):  # Adjust the maximum number of steps as needed
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10  # Adjust the reward for completing the task
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print("Episode: {}/{}, Total Reward: {}, Epsilon: {:.2}".format(
                    episode + 1, num_episodes, total_reward, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

if __name__ == "__main__":
    main()