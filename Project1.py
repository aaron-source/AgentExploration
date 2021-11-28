


"""
Framework must consist of n x n matrix

that can take any k number of agents

each robot would have a copy of the framework
=============================================
Reward Function:
    which cells have been visited and which cells have not been visited
    + 1 for a robot to stay put
    + 10 for visiting a new environment
    + 2 for visiting an environment that has already been visited
    - 100 if two agents visit the same environment

Action:
    - 8 neighbors
    - staying put

"""
import numpy as np
import logging
import time

from collections import namedtuple
from itertools import count

"""
How the state is represented
    - three matrixes
        -1) all the obstacles in the environment
        1) the current location of the robot
        2) all the cells that have been covered so far

"""

#creating neural network
import torch.nn as nn
import torch.nn.functional as F
import torch

logging.basicConfig(filename="logging.log", filemode="w", level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 5
MEMORY = 10000

class Model(nn.Module):
    def __init__(self, height, width):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=1).to(device=device)
        self.norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1).to(device=device)
        self.norm2 = nn.BatchNorm2d(32)
        #self.conv3 = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=1)
        #self.norm3 = nn.BatchNorm2d(32)


        def reduce_size(size, kernel_size, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        width = reduce_size(reduce_size(width, kernel_size=2), kernel_size=2)
        height = reduce_size(reduce_size(height, kernel_size=2), kernel_size=2)
        
        self.Linearlayer = nn.Linear(width * height * 32, 9)
        
    def forward(self, x):
        try:
            logging.info("Information passing through first convolution layer")
            x = F.relu(self.norm1(self.conv1(x)))
            logging.info("Information passing through second convolution layer")
            x = F.relu(self.norm2(self.conv2(x)))
            logging.info("Infomation is passing through third convolutional layer")
            logging.info("Information paassing through linear layer")
            x = self.Linearlayer(x.view(x.size(0), -1))
        except Exception as e:
            logging.error(f"The following error occured: {e}")
        return x    



from collections import deque

class Epsilon():
    def __init__(self, start, end, decay):
        self.start = start #1
        self.end = end #0.01
        self.decay = decay #0.999
        
    def choose_action(self, time_step):
        epsilon = self.end + (self.start - self.end) \
          * math.exp(-1. * time_step / self.decay)
        return epsilon

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [[]]
        self.Transitions = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.steps = []
        
    def push(self, *args):
        self.steps.append(self.Transitions(*args))

    def add_to_memory(self, episode):
        if episode > len(self.memory) - 1:
            self.memory.pop(0)
        self.memory.append(self.steps)
        self.steps = []

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
            

    def get_sample(self, batch_size):
        return len(self.memory) >= batch_size


# # Robot Class

# In[28]:


import math
import random
import torch
class Environment():
    def __init__(self,robot, N):
        self.N = N
        self.environment = torch.zeros((3, N, N))
        self.action_count = 9
        self.states = N * N * 3
        self.agent = robot
        self.done = False
        self.p_completion = 0.0

    
    """
    This will reset the environment and set the agent to position (1, 1) in the grid
    
    """
    def reset(self):
        self.agent.model_count = 0
        self.agent.random_count = 0
        self.environment = torch.zeros((3, self.N, self.N))
        self.p_completion = 0.0
        self.agent.steps = 0
        self.agent.x_coordinate = 0
        self.agent.y_coordinate = 0
        self.environment[1][self.agent.x_coordinate][self.agent.y_coordinate] = 1
    
    #returns the state of which the robot is in
    def get_state(self):
        return self.environment 

    def step(self, action):
        done = False
        reward = 0
        if len(torch.nonzero(self.environment[2])) == self.N * self.N:
            done = True
        old_x, old_y = self.move_robot(action)
        if self.environment[0][self.agent.x_coordinate][self.agent.y_coordinate] == -1:
            reward = -100
        elif self.environment[2][self.agent.x_coordinate][self.agent.y_coordinate] == 2: #2 = visited stated
            reward = 0
        elif self.environment[2][self.agent.x_coordinate][self.agent.y_coordinate] == 0: #0 = unvisited state
            reward = 10
        logging.info(f" The robot got a negative reward of : {reward}")
        """
        each state is represented as [0, 0, 0]cla

        for each index in the state will be described below:
            position [0] is keeping track of of the obstacle locations
            position [1] is keeping track of the robots location
            position [2] is keeping track of all the previous positions the robot has visited
        Line 157: is setting the previous state back to 0
        Line 158: is setting the current position of the robot
        Line 159: is setting the value of the third index to two to keep track of the robots steps
        """
        self.environment[1][old_x][old_y] = 0
        self.environment[1][self.agent.x_coordinate][self.agent.y_coordinate]= 1 #1 = current state
        self.environment[2][old_x][old_y] = 2
        self.p_completion = len(torch.nonzero(self.environment[2])) / (self.N * self.N)
        return old_x, reward, done, old_y

    def move_robot(self, action):
            old_x, old_y = self.agent.x_coordinate, self.agent.y_coordinate
            actions = {
                0: (self.agent.x_coordinate, self.agent.y_coordinate + 1),
                1: (self.agent.x_coordinate, self.agent.y_coordinate - 1),
                2: (self.agent.x_coordinate + 1, self.agent.y_coordinate),
                3: (self.agent.x_coordinate - 1, self.agent.y_coordinate),
                4: (self.agent.x_coordinate + 1, self.agent.y_coordinate - 1),
                5: (self.agent.x_coordinate - 1, self.agent.y_coordinate - 1),
                6: (self.agent.x_coordinate + 1, self.agent.y_coordinate + 1),
                7: (self.agent.x_coordinate - 1, self.agent.y_coordinate + 1),
                8: (self.agent.x_coordinate, self.agent.y_coordinate)
            }
            """
            This is checking to make sure that the robot is not going out of bounds
            
            """
            new_state = actions[action]
            self.agent.x_coordinate = new_state[0]
            self.agent.y_coordinate = new_state[1]
            
            if self.agent.x_coordinate > self.N - 1: 
                self.agent.x_coordinate = old_x
            elif self.agent.x_coordinate < 0:
                self.agent.x_coordinate = old_x
            if self.agent.y_coordinate > self.N - 1:
                self.agent.y_coordinate = old_y
            elif self.agent.y_coordinate < 0:
                self.agent.y_coordinate = old_y
            return old_x, old_y

class Robot():
    def __init__(self, matrix_shape):
        self.x_coordinate = 0
        self.y_coordinate = 0
        self.steps = 0
        self.action_space=9
        self.decay = 10000
        self.model_count = 0
        self.random_count = 0 
        #change it to exponential decay, right now it is at a fixed rate
    
    def take_action(self, state, policy_network):

        rate = Epsilon(1, 0.01, self.decay)


        epsilon = rate.choose_action(self.steps)
        
        self.decay = self.decay * .999
    
        self.steps += 1
    
        if epsilon > random.random():
            action = random.randrange(self.action_space)
            self.random_count += 1
            return torch.tensor([[action]])
        else:
            with torch.no_grad():
                self.model_count += 1
                return policy_network(state).max(1)[1].view(1, 1)

import torch.optim as optim
policy_net = Model(N, N).to(device=device)
target_network = Model(N, N).to(device=device)
target_network.load_state_dict(policy_net.state_dict())
target_network.eval()
adam = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)
from itertools import chain
def extract_tensors(batch_set):
    sample = memory.sample(BATCH_SIZE)
    sample = list(chain.from_iterable(sample))
    batch = memory.Transitions(*zip(*sample))
    states = torch.cat(batch.state).to(device)
    rewards = torch.cat(batch.reward).to(device)
    actions = torch.cat(batch.action).to(device)
    new_states = torch.cat(batch.next_state).to(device)
    return states, actions, rewards, new_states


BATCH_SIZE = 32
def optimize_model(episode):
        transitions = memory.sample(BATCH_SIZE)

        states, actions, rewards, new_states = extract_tensors(transitions)

        Q_values = policy_net(states).gather(1, actions.view(-1, 1))

        final_values = target_network(new_states).max(1)[0].detach()

        expected = (final_values * 0.999) + rewards

        loss_function = nn.MSELoss()

        loss_value = loss_function(Q_values, expected.unsqueeze(-1)).to(device=device)

        logging.info(f"THE LOSS FOR EPISODE: {episode} IS {loss_value}")

        adam.zero_grad()

        loss_value.backward()
      
        adam.step()







## 3 matrices
"""
1) matrix is going to keep track of the position of where the robot is at
2) path of which the robot has traveled
3) all the obstacles are located
    - save the plots as jpeg
"""
"""
display function
"""
def display_path():
    pass

"""
showing the percentage, episode in the x_axis and the percentage completed in the y_axis

"""

def plot_percentage(percent_complete):
    pass
# In[32]

time_focus = []
area_completion_chart = []
percentage_covered = []
import time
robot = Robot(N)
area = Environment(robot, N)
policy_net = Model(N, N)
start_2 = time.time()
agent_route = []
current_step = 0;
for episode in range(100):
    steps = []
    start = time.time()
    area.reset()
    state = area.get_state()
    print(f"Episode: {episode} has begun")
    logging.info(f"Episode: {episode} has begun")
    total = 0
    durations = 0
    for i in count():
        action = area.agent.take_action(state.view(1, 3, N, N), policy_net)
        state = state.view(1, 3, N, N)
        logging.info(f"Performed action: {action}")
        
        steps.append((area.agent.x_coordinate, area.agent.y_coordinate))
        
        old_x, reward, done, old_y = area.step(action.item())

        next_state = area.get_state()

        next_state = next_state.view(1, 3, N, N)

        reward = torch.tensor([reward])

        memory.push(state, action, next_state, reward)

        state = next_state
        percentage_covered.append(area.p_completion * 100)
        if memory.get_sample(BATCH_SIZE):
            optimize_model(episode)
        if done:
            logging.info(f"The robot explored the entire environment in episode: {episode}")
            logging.info(f"To explore the entire environment took: {(time.time() - start_2) / 60}")
            break
        if i >= N * N * 3:
            logging.info(f"Maximum time step reached")
            break
        current_step += 1
        
        durations += 1
    memory.add_to_memory(episode)
    end = time.time() - start
    agent_route.append((episode, steps))
    area_completion_chart.append((episode, area.p_completion))
    print(f"The total time to finish episode {episode}: {end / 60} minutes, completion %: {area.p_completion * 100} ")
    print(f"Number of Random Actions: {area.agent.random_count}, Number of Model Actions: {area.agent.model_count}, Number of Steps: {area.agent.steps}")
    time_focus.append((episode, end / 60))
    if episode % 10 == 0:
        logging.info("uploading the policy network weights to the target network")
        print("uploading the policy network weights to the target network")
        target_network.load_state_dict(policy_net.state_dict())

import pandas as pd
frame = pd.DataFrame(area_completion_chart, columns=["Episode", "percentage covered"])
time_frame = pd.DataFrame(time_focus, columns=["Episode", "Time"])
route_taken = pd.DataFrame(agent_route, columns=["Episode", "Robot Route"])
route_taken.to_csv("agent_route5.csv", index=False)
frame.to_csv("robot5.csv", index=False)
time_frame.to_csv("timeframe5.csv", index=False)
