import torch
import gym
import argparse
from pytorch_rl.algorithms import DDPG

def main(args) :
    env = gym.make('MountainCarContinuous-v0')
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    ddpg = DDPG(args, state_space, action_space)

    model_directory = ddpg.args.save_directory+'%d/model_%d' % (ddpg.args.saved_iter, ddpg.args.saved_iter)
    ddpg.actor.load_state_dict(torch.load(model_directory))
    ddpg.epsilon = 0

    print('\nCollecting experience...')
    for i_episode in range(ddpg.args.num_episode):


        s = env.reset()
        ep_r = 0
        while True:
            if ddpg.args.render:
                env.render()
            a = ddpg.choose_action(s)

            # take action
            s_, r, done, info = env.step(a)

            ep_r += r

            if done:
                if i_episode % ddpg.args.ep_print_iter == 0:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2), '| epsilon : ', ddpg.epsilon)
                    break
            s = s_


if __name__ == '__main__' :
    parse = argparse.ArgumentParser()

    parse.add_argument('--batch_size', default=256)
    parse.add_argument('--lr', default=0.003)
    parse.add_argument('--epsilon_decay', default=0.9)
    parse.add_argument('--gamma', default=0.90)
    parse.add_argument('--target_replace_iter', default=100)
    parse.add_argument('--memory_capacity', default=2000)
    parse.add_argument('--tau', default=0.001)

    parse.add_argument('--render', default=True)
    parse.add_argument('--num_episode', default=10000)
    parse.add_argument('--ep_print_iter', default=1, help='print episode_reward at every %d step')
    parse.add_argument('--model_save_iter', default=20, help='save model at every %d step')
    parse.add_argument('--continue_training', default=False,
                       help='Will you continue training using your saved model & memory')
    parse.add_argument('--saved_iter', default=140, help='last saved model iteration number. ')
    parse.add_argument('--save_directory', default='./save/')
    args = parse.parse_args()

    main(args)