import pybullet as p
from pybullet_envs.bullet import kukaGymEnv
import argparse
from pytorch_rl.algorithms import DDPG


def main(args) :
    p.connect(p.SHARED_MEMORY)

    env = kukaGymEnv.KukaGymEnv(renders=False, isEnableSelfCollision=True)

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    print('action_dim : ', action_dim)
    print('state_dim : ', state_dim)
    ddpg = DDPG(args, state_dim, action_dim)
    ddpg.kuka_train_loop(env)


if __name__ == '__main__' :
    parse = argparse.ArgumentParser()

    parse.add_argument('--batch_size', default=200)
    parse.add_argument('--lr', default=0.003)
    parse.add_argument('--epsilon_decay', default=0.99)
    parse.add_argument('--gamma', default=0.99)
    parse.add_argument('--target_replace_iter', default=100)
    parse.add_argument('--tau', default=0.001)
    parse.add_argument('--memory_capacity', default=10000)


    parse.add_argument('--num_episode', default=10000)
    parse.add_argument('--episode_len', default=600)
    parse.add_argument('--ep_print_iter', default=1, help='print episode_reward at every %d step')
    parse.add_argument('--model_save_iter', default=100, help='save model at every %d step')
    parse.add_argument('--continue_training', default=False,
                       help='Will you continue training using your saved model & memory')
    parse.add_argument('--saved_iter', default=5, help='last saved model iteration number. ')
    parse.add_argument('--save_directory', default='./kuka/save_ddpg/')

    args = parse.parse_args()

    main(args)