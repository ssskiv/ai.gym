from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range
from PPO_agent import PPOAgent, Transition

from gym.spaces import Box, Discrete
import numpy as np

from vehicle import Driver


class EKFRobot(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        # Define agent's observation space using Gym's Box, setting the lowest and highest possible values
        self.buffer_size = 15
        
        self.buffer = [[0, 0, 0, 0, 0, 0, 0, 0, 0]]*self.buffer_size 
        
        
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
        self.observation_space = Box(low=np.array(mVector6 + mVector9*self.buffer_size),
                                     high=np.array(pVector6 + pVector9*self.buffer_size),
                                     dtype=np.float64)
        # Define agent's action space using Gym's Discrete
        self.action_space = Box(low=np.array([-np.inf, -np.inf, -np.inf,
                                              -np.inf, -np.inf, -np.inf,
                                              -np.inf, -np.inf, -np.inf]),
                                     high=np.array([np.inf, np.inf, np.inf,
                                                    np.inf, np.inf, np.inf,
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

        self.steps_per_episode = 20000  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved

    def get_observations(self):
        
        accelerometer_data = self.accel.getValues()
        gyro_data = self.gyro.getValues()
        self.gps_data = self.gps.getSpeedVector()
        self.xyz_data = self.car.getPosition()
        self.angles_data = self.car.getField('rotation').getSFVec3f()
        
        buffer_data = [item for sublist in self.buffer for item in sublist]
        
        observation = [accelerometer_data[0], accelerometer_data[1], accelerometer_data[2],
                gyro_data[0], gyro_data[1], gyro_data[2],
                ] + buffer_data
        return observation

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_reward(self, action=None):
        # Reward is +1 for every step the episode hasn't ended
        # print(self.buffer)
        last = self.buffer[0]
        xyz_delta = 1.0
        vel_delta = 1.0
        angles_delta = 1.0
        for i in range(3):
            xyz_delta += abs(self.xyz_data[i]-last[i])
        for i in range(3):
            vel_delta += abs(self.gps_data[i]-last[i+3])
        for i in range(3):
            angles_delta += abs(self.angles_data[i]-last[i+6])
        
        return 0.1/xyz_delta + 0.1/vel_delta + 0.1/angles_delta     

    def is_done(self):
        if self.episode_score > 15000.0 or self.xyz_data[2] < 0:
            return True
   
        # if abs(pole_angle) > 0.261799388:  # more than 15 degrees off vertical (defined in radians)
        #     return True

        return False

    def solved(self):
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > 195.0:  # Last 100 episodes' scores average value
                return True
        return False

    def get_info(self):
        return None

    def render(self, mode='human'):
        pass

    def apply_action(self, action):
        self.buffer.insert(0,action[0])
        self.buffer = self.buffer[:-1]
        
        
        
        


env = EKFRobot()
agent = PPOAgent(number_of_inputs=env.observation_space.shape[0], number_of_actor_outputs=env.action_space.shape[0])

solved = False
episode_count = 0
episode_limit = 20000
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and episode_count < episode_limit:
    observation = env.reset()  # Reset robot and get starting observation
    env.episode_score = 0

    for step in range(env.steps_per_episode):
        # In training mode the agent samples from the probability distribution, naturally implementing exploration
        selected_action = agent.work(observation, type_="simple")
        # Step the supervisor to get the current selected_action's reward, the new observation and whether we reached
        # the done condition
        new_observation, reward, done, info = env.step([selected_action])

        # Save the current state transition in agent's memory
        trans = Transition(observation, selected_action, len(selected_action), reward, new_observation)
        agent.store_transition(trans)

        if done:
            # Save the episode's score
            env.episode_score_list.append(env.episode_score)
            agent.train_step(batch_size=step + 1)
            solved = env.solved()  # Check whether the task is solved
            break

        env.episode_score += reward  # Accumulate episode reward
        observation = new_observation  # observation for next step is current step's new_observation

    print("Episode #", episode_count, "score:", env.episode_score)
    episode_count += 1  # Increment episode counter

if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")

observation = env.reset()
env.episode_score = 0.0
while True:
    selected_action, action_prob = agent.work(observation, type_="selectActionMax")
    observation, _, done, _ = env.step([selected_action])
    if done:
        observation = env.reset()