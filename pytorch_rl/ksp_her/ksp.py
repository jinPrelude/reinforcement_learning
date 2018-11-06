"""
TO use hover_v1, visit https://github.com/jinPrelude/kerbal-rl
"""


from pytorch_rl.ksp_her.hover import hover_v1
import argparse
from pytorch_rl.ksp_her.algorithm import DDPG_HER


def main(args) :

    env = hover_v1(max_altitude=300, max_step=100, epsilon=1)

    action_dim = env.action_space
    state_dim = env.observation_space
    goal_dim = env.goal_space
    max_step = 100        # max episode len
    print('action_dim : ', action_dim)
    print('state_dim : ', state_dim)
    ddpg = DDPG_HER(args, state_dim, action_dim, goal_dim, max_step)
    ddpg.ksp_train_loop(env)


if __name__ == '__main__' :
    parse = argparse.ArgumentParser()

    parse.add_argument('--batch_size', default=200)
    parse.add_argument('--lr', default=0.003)
    parse.add_argument('--epsilon_decay', default=0.9)
    parse.add_argument('--gamma', default=0.99)
    parse.add_argument('--target_replace_iter', default=100)
    parse.add_argument('--tau', default=0.001)
    parse.add_argument('--memory_capacity', default=10000)

    parse.add_argument('--env_epsilon', default=1.)
    parse.add_argument('--num_episode', default=10000)
    parse.add_argument('--episode_len', default=100)
    parse.add_argument('--ep_print_iter', default=1, help='print episode_reward at every %d step')
    parse.add_argument('--model_save_iter', default=100, help='save model at every %d step')
    parse.add_argument('--continue_training', default=False,
                       help='Will you continue training using your saved model & memory')
    parse.add_argument('--saved_iter', default=220, help='last saved model iteration number. ')
    parse.add_argument('--save_directory', default='./save/')

    args = parse.parse_args()

    main(args)