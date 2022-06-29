from typing import Tuple

import gym
import ipdb
import torch
import numpy as np
from torch import nn
from torch import Tensor
from torch.optim import Adam
from torch.nn.functional import relu
 
BatchSize = 32
LearningRate = 0.01
Epsilon = 0.9
Gamma = 0.9
ReplaceFrequency = 100
MemoryCapacity = 2000
 
env = gym.make('CartPole-v0').unwrapped
N_actions: int = env.action_space.n
N_states: int = env.observation_space.shape[0]
 
class Net(nn.Module):
 
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_actions)
        self.out.weight.data.normal_(0, 0.1)
   
    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = relu(self.fc1(x))
        return self.out(x)
   
 
class DQN:
 
    def __init__(self):
        self.eval_net = Net()
        self.target_net = Net()
        self.learning_count = 0
        self.memory_count = 0
        self.memory = np.zeros((MemoryCapacity, N_states * 2 + 2))
        self.optimizer = Adam(self.eval_net.parameters(), lr=LearningRate)
        self.loss_function = nn.MSELoss()
 
    def action(self, state: np.ndarray) -> np.int64:
        state: Tensor = torch.unsqueeze(torch.FloatTensor(state), dim=0)
        if np.random.uniform() >= Epsilon:
            action: np.int64 = np.random.randint(0, N_actions)
        else:
            max_index: Tensor = torch.max(self.eval_net.forward(state), dim=1)[1]
            action: np.int64 = max_index.numpy()[0]
        return action
   
    def store(self, prev: np.ndarray, action: np.int64, reward: float, curr: np.ndarray):
        transition: np.ndarray = np.hstack((prev, [action, reward], curr))
        index = self.memory_count % MemoryCapacity
        self.memory[index] = transition
        self.memory_count += 1
   
    def learn(self):
        if self.learning_count % ReplaceFrequency == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learning_count += 1

        sample_indices: np.ndarray = np.random.choice(MemoryCapacity, BatchSize)
        sample_memory: np.ndarray = self.memory[sample_indices, :]
        
        batch_prev: Tensor = torch.FloatTensor(sample_memory[:, :N_states])
        batch_action: Tensor = torch.LongTensor(sample_memory[:, N_states:N_states+1].astype(int))
        batch_raward: Tensor = torch.FloatTensor(sample_memory[:, N_states+1:N_states+2])
        batch_next: Tensor = torch.FloatTensor(sample_memory[:, -N_states:])
        
        # ipdb.set_trace()
        q_eval: Tensor = self.eval_net(batch_prev).gather(1, batch_action)
        q_next: Tensor = self.target_net(batch_next).detach()
        q_target: Tensor = batch_raward + Gamma * q_next.max(dim=1)[0].view(BatchSize, 1)
        
        loss: Tensor = self.loss_function(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

 
if __name__ == '__main__':
    N_iteration = 10
    dqn = DQN()
 
    for episode in range(500):
        prev_state: np.ndarray = env.reset()
        sum_reward = 0
 
        while True:
            env.render()

            action: np.int64 = dqn.action(prev_state)
            step_result: Tuple[np.ndarray, float, bool, dict] = env.step(action)
            next_state: np.ndarray = step_result[0]
            done: bool = step_result[2]
            
            x, _, t, _ = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(t)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            
            dqn.store(prev_state, action, reward, next_state)
            sum_reward += reward
            prev_state = next_state

            if dqn.memory_count > MemoryCapacity:
                dqn.learn()
            if done:
                break
        
        print(f'episode: {episode}\tsum_reward: {sum_reward}')
