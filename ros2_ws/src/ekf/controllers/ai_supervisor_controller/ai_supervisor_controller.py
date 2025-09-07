from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range
from LSTM_agent import LSTMAgent, Transition

from gym.spaces import Box, Discrete
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
experiment_name = 'EKF_experiment_straight'
experiment_number = '1'

writer = SummaryWriter(f'runs/{experiment_name}_{experiment_number}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

GUI = False

if GUI:
    from network_visualizer import NetworkVisualizer
    import tkinter as tk

    layer_structure = [6, 32, 64, 6]

    layer_name_map = {
        'input_current': 0,
        'current_mlp_relu': 1,
        'common_mlp_relu': 2,
        'output_mu': 3
    }
    root = tk.Tk()
    root.title("Нейросеть в действии")
    visualizer = NetworkVisualizer(root, layer_structure, layer_name_map)


class EKFRobot(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        # Define agent's observation space using Gym's Box, setting the lowest and highest possible values
        self.buffer_size = 30
        
        self.buffer = [[0, 0, 0, 0, 0, 0]]*self.buffer_size
        self.pred_buffer =[[0, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        
        mVector6 = [-np.inf, -np.inf, -np.inf,
                   -np.inf, -np.inf, -np.inf]
        mVector9 = [-np.inf, -np.inf, -np.inf,
                    -np.inf, -np.inf, -np.inf,
                   -np.inf, -np.inf, -np.inf]
        pVector6 = [np.inf, np.inf, np.inf,
                   np.inf, np.inf, np.inf]
        pVector9 = [np.inf, np.inf, np.inf,
                   np.inf, np.inf, np.inf,
                   np.inf, np.inf, np.inf]
        self.observation_space = Box(low=np.array(mVector6 + mVector6*self.buffer_size),
                                     high=np.array(pVector6 + pVector6*self.buffer_size),
                                     dtype=np.float64)
        # Define agent's action space using Gym's Discrete
        self.action_space = Box(low=np.array([-np.inf, -np.inf, -np.inf,
                                              -np.inf, -np.inf, -np.inf]),
                                     high=np.array([np.inf, np.inf, np.inf,
                                                    np.inf, np.inf, np.inf]),
                                     dtype=np.float64)

        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        ### DEFINE NODES BEGIN ###
        self.car = self.getFromDef('CAR')
        self.gps = self.getDevice('gps') # GPS to get real speed data

        # input data for NN
        self.accel = self.getDevice('accelerometer')
        self.gyro = self.getDevice('gyro')

        ### DEFINE NODES END ###
        
        ### ENABLE SENSORS BEGIN ###
        self.gps.enable(self.timestep)
        self.accel.enable(self.timestep)
        self.gyro.enable(self.timestep)
        ### ENABLE SENSORS END ###  

        self.steps_per_episode = 2000  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        
        
        self.mean_deltas = []

    # In ai_supervisor_controller.py

    def get_observations(self):
        accelerometer_data = self.accel.getValues()
        gyro_data = self.gyro.getValues()
        self.gps_data = self.gps.getSpeedVector()
        self.xyz_data = self.car.getPosition()
        self.angles_data = self.car.getField('rotation').getSFVec3f()

        # Current observation is a list of 6 features
        current_observation = [accelerometer_data[0], accelerometer_data[1], accelerometer_data[2],
                               gyro_data[0], gyro_data[1], gyro_data[2]]

        # Update the history buffer
        self.buffer.insert(0, current_observation)
        self.buffer = self.buffer[:-1]

        # History is a numpy array of shape (30, 6)
        history_observation = np.array(self.buffer, dtype=np.float32)

        # Return both parts separately
        return current_observation, history_observation

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_reward(self, action=None):
        
        last = self.pred_buffer[0]
        
        xyz_delta = 1.0
        angles_delta = 1.0
        
        for i in range(3):
            xyz_delta += abs(self.xyz_data[i]-last[i])
        for i in range(3):
            angles_delta += abs(self.angles_data[i]-last[i+3])
            
        deltas = [xyz_delta, angles_delta]    
        if len(self.mean_deltas)>0:
            for i in range(2):
                self.mean_deltas[i] = (self.mean_deltas[i]+deltas[i])/2
        else:
            self.mean_deltas = [xyz_delta, angles_delta]
            
            
        
        return (1)/(xyz_delta**2)# + (1)/(angles_delta**2)     

    def is_done(self):
        if self.episode_score > 15000.0 or self.xyz_data[2] < 0:
            return True
   
        # if abs(pole_angle) > 0.261799388:  # more than 15 degrees off vertical (defined in radians)
        #     return True

        return False

    def solved(self):
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > 6000.0:  # Last 100 episodes' scores average value
                return True
        return False

    # In ai_supervisor_controller.py, inside the EKFRobot class

    def get_info(self):
        return self.mean_deltas

    # ADD THIS METHOD
    def reset(self):
        """
        Overrides the base reset method to return the two-part observation.
        """
        # Call the parent's reset method to handle simulation reset
        super().reset()
        
        # Reset the buffers to a known state (optional but recommended)
        self.buffer = [[0, 0, 0, 0, 0, 0]] * self.buffer_size
        self.pred_buffer = [[0, 0, 0, 0, 0, 0, 0, 0, 0]]

        # Return the initial two-part observation
        return self.get_observations()

    def render(self, mode='human'):
        pass

    def apply_action(self, action):
        self.pred_buffer.insert(0,action)
        self.pred_buffer = self.pred_buffer[:-1]
        if GUI:
            activations['output_mu'] = torch.tensor(action)
            visualizer.update_visuals(activations)
            root.update()
        
        
        
        


env = EKFRobot()

# NEW CODE
# Define the dimensions clearly
sensor_dim = 6  # 3 accelerometer + 3 gyro values
action_dim = env.action_space.shape[0]

# Initialize the agent with the new, specific parameters
agent = LSTMAgent(
    current_input_dim=sensor_dim,
    history_feature_dim=sensor_dim,
    action_dim=action_dim,
    device=device, 
    clip_param=0.2,
    actor_lr=0.0003, 
    critic_lr=0.001, 
    max_grad_norm=0.5
)

try:
    agent.load('EKF_model')
    print("Loaded model EKF_model")
except:
    print("No model found, training from scratch")
    
solved = False
episode_count = 0
episode_limit = 20000
activations = {}
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and episode_count < episode_limit:
    env.mean_deltas = []
    current_obs, history_obs = env.reset() # Unpack the tuple 
    env.episode_score = 0

    for step in range(env.steps_per_episode):
        # Pass both parts of the observation to the agent
        selected_action, action_log_prob = agent.work(current_obs, history_obs)
        
        if GUI:
            activations['input_current'] = torch.tensor(current_obs)
            visualizer.update_visuals(activations)
            root.update()
        
        state_tuple = (current_obs, history_obs)

        # Unpack the new state from env.step()
        (new_current_obs, new_history_obs), reward, done, info = env.step(selected_action)

        new_state_tuple = (new_current_obs, new_history_obs)
        
        

        # Store the correct state tuple and log probability
        trans = Transition(state_tuple, selected_action, action_log_prob, reward, new_state_tuple)
        agent.store_transition(trans)

        # Update the observations for the next iteration
        env.episode_score += reward  # Accumulate episode reward
        current_obs, history_obs = new_current_obs, new_history_obs
    
    print("Episode #", episode_count, "score:", env.episode_score)
    writer.add_scalar('Episode score', env.episode_score, episode_count)
    writer.add_scalar('Mean deltas', np.mean(env.get_info()), episode_count)
    agent.save('EKF_model')

    episode_count += 1  # Increment episode counter
    
writer.close()

if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")

root.mainloop()
observation = env.reset()
env.episode_score = 0.0
while True:
    selected_action, action_prob = agent.work(observation, type_="selectActionMax")
    observation, _, done, _ = env.step([selected_action])
    if done:
        observation = env.reset()