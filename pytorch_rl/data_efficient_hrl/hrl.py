from pytorch_rl.data_efficient_hrl.utils import ReplayBuffer
import numpy as np

class HRL() :
    def __init__(self, args, p_low, p_high, env, a_dim, s_dim):
        self.args = args
        self.p_low = p_low
        self.p_high = p_high
        self.env = env
        self.a_dim = a_dim
        self.s_dim = s_dim

        # Variables for training
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.episode_num = 0
        self.done = False


    def train(self):

        # env init
        # step iteration
            # store s_0
            # generate g by high policy
                # output action by low policy
                # accumulate reward for high policy
                # reshape reward
                # store experience
                # train
            # store experience(s_0, s_c, sum(r), g)
            # re_label g_bar
            # train
        pass

    def trainable(self):
        pass

    def train_low(self):
        pass

    def re_labbel(self):
        pass

    def train_high(self):
        pass

    def save_model(self):
        pass