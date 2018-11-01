
import gym
import argparse
from pytorch_rl.algorithms import DDPG

def main(args) :
    env = gym.make('BipedalWalker-v2')
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    ddpg = DDPG(args, state_space, action_space)

    ddpg.simple_one_to_one(env)


if __name__ == '__main__' :
    parse = argparse.ArgumentParser()

    parse.add_argument('--batch_size', default=256)
    parse.add_argument('--lr', default=0.003)
    parse.add_argument('--epsilon_decay', default=0.99)
    parse.add_argument('--gamma', default=0.90)
    parse.add_argument('--target_replace_iter', default=100)
    parse.add_argument('--memory_capacity', default=2000)
    parse.add_argument('--tau', default=0.001)

    parse.add_argument('--render', default=True)
    parse.add_argument('--num_episode', default=10000)
    parse.add_argument('--episode_len', default=300)
    parse.add_argument('--ep_print_iter', default=1, help='print episode_reward at every %d step')
    parse.add_argument('--model_save_iter', default=100, help='save model at every %d step')
    parse.add_argument('--continue_training', default=False,
                       help='Will you continue training using your saved model & memory')
    parse.add_argument('--saved_iter', default=5, help='last saved model iteration number. ')
    parse.add_argument('--save_directory', default='./save/')
    args = parse.parse_args()

    main(args)