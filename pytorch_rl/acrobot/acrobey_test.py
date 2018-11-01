import torch
import numpy as np
import gym
import argparse
from pytorch_rl.algorithms import DDPG

def main(args) :
    env = gym.make('BipedalWalker-v2')
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    ddpg = DDPG(args, state_space, action_space)

    ddpg.actor.load_state_dict(torch.load('./save/100/model_100'))
    ddpg.epsilon = 0.00001
    for i_episode in range(ddpg.args.num_episode):

        s = env.reset()
        ep_r = 0
        while True:
            env.render()
            a = ddpg.choose_action(s)
            a *= 2.0
            # take action
            s_, r, done, info = env.step(a)

            # modify the reward
            # x, x_dot, theta, theta_dot = s_
            # r1 = (env.env.x_threshold - abs(x)) / env.env.x_threshold - 0.8
            # r2 = (env.env.theta_threshold_radians - abs(theta)) / env.env.theta_threshold_radians - 0.5
            # r = r1 + r2



            ep_r += r
            if done:
                if i_episode % ddpg.args.ep_print_iter == 0:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))
                break
            s = s_



if __name__ == '__main__' :
    parse = argparse.ArgumentParser()

    parse.add_argument('--batch_size', default=256)
    parse.add_argument('--lr', default=0.003)
    parse.add_argument('--epsilon_decay', default=0.9)
    parse.add_argument('--gamma', default=0.90)
    parse.add_argument('--target_replace_iter', default=100)
    parse.add_argument('--memory_capacity', default=500)
    parse.add_argument('--num_episode', default=10000)
    parse.add_argument('--ep_print_iter', default=1)
    parse.add_argument('--model_save_iter', default=10)
    parse.add_argument('--tau', default=0.001)
    args = parse.parse_args()

    main(args)
