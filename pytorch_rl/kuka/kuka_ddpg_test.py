import torch
import pybullet as p
from pybullet_envs.bullet import kukaGymEnv
import argparse
from pytorch_rl.algorithms import DDPG
import os
import numpy as np

def main(args) :
    p.connect(p.SHARED_MEMORY)

    env = kukaGymEnv.KukaGymEnv(renders=True, isEnableSelfCollision=True)

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    print('action_dim : ', action_dim)
    print('state_dim : ', state_dim)
    ddpg = DDPG(args, state_dim, action_dim)

    model_directory = './save_kuka_ddpg/%d/model_%d' % (ddpg.args.saved_iter, ddpg.args.saved_iter)
    ddpg.actor.load_state_dict(torch.load(model_directory))
    ddpg.epsilon = 0

    for i_episode in range(ddpg.args.num_episode):
        s = env.reset()
        ep_r = 0


        while True:

            a = ddpg.choose_action(s)

            # take action
            s_, r, done, info = env.step(a)

            ep_r += r
            if done:
                if i_episode % ddpg.args.ep_print_iter == 0:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))

                break
            s = s_


if __name__ == '__main__' :
    parse = argparse.ArgumentParser()

    parse.add_argument('--batch_size', default=400)
    parse.add_argument('--lr', default=0.003)
    parse.add_argument('--epsilon_decay', default=0.99)
    parse.add_argument('--gamma', default=0.99)
    parse.add_argument('--target_replace_iter', default=100)
    parse.add_argument('--memory_capacity', default=30000)
    parse.add_argument('--num_episode', default=10000)
    parse.add_argument('--episode_len', default=600)
    parse.add_argument('--ep_print_iter', default=1)
    parse.add_argument('--model_save_iter', default=50)
    parse.add_argument('--continue_training', default=True)
    parse.add_argument('--saved_iter', default=1000)
    args = parse.parse_args()

    main(args)