"""
This code is for testing the model
"""
import tensorflow as tf
import numpy as np
import gym
import argparse
import pprint as pp


from agent import ActorNetwork, CriticNetwork


def train(sess, env, args, actor, critic):

    saver = tf.train.Saver()

    saver.restore(sess, './results/model_save/model.ckpt')


    for i in range(int(args['max_episodes'])):

        s = env.reset()


        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()


            # uo-process 노이즈 추가
            # add uo-process noise
            a = actor.predict(np.reshape(s, (1, actor.s_dim)))

            s2, r, terminal, info = env.step(a[0])
            s = s2


            if terminal:
                break

def main(args):
    with tf.Session() as sess:

        env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())


        train(sess, env, args, actor, critic)



if __name__ == '__main__':


    # print the parameters on the console
    # and also offer the parametes to the main function
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')
    parser.add_argument('--load-model', default=True)
    parser.set_defaults(render_env=True)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)