import gym
import numpy as np
import torch
import logging
import math
from pytorch_cem import cem
from gym import wrappers, logger as gym_log

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    ENV_NAME = "Pendulum-v0"
    TIMESTEPS = 10  # T
    N_ELITES = 15
    N_SAMPLES = 100  # K
    SAMPLE_ITER = 3  # M
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    d = "cpu"
    dtype = torch.double


    def dynamics(state, perturbed_action):
        # true dynamics from gym
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)

        g = 10
        m = 1
        l = 1
        dt = 0.05

        u = perturbed_action
        u = torch.clamp(u, -2, 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -8, 8)

        state = torch.cat((newth, newthdot), dim=1)
        return state


    def angle_normalize(x):
        return (((x + math.pi) % (2 * math.pi)) - math.pi)


    def running_cost(state, action):
        theta = state[:, 0]
        theta_dt = state[:, 1]
        action = action[:, 0]
        cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt ** 2 + 0.001 * action ** 2
        return cost


    def train(new_data):
        pass


    downward_start = True
    env = gym.make(ENV_NAME).env  # bypass the default TimeLimit wrapper
    env.reset()
    if downward_start:
        env.state = [np.pi, 1]

    env = wrappers.Monitor(env, '/tmp/mppi/', force=True)
    env.reset()
    if downward_start:
        env.env.state = [np.pi, 1]

    nx = 2
    nu = 1
    cem_gym = cem.CEM(dynamics, running_cost, nx, nu, num_samples=N_SAMPLES, num_iterations=SAMPLE_ITER,
                      horizon=TIMESTEPS, device=d, num_elite=N_ELITES, u_max=ACTION_HIGH)
    total_reward, data = cem.run_cem(cem_gym, env, train)
    logger.info("Total reward %f", total_reward)
