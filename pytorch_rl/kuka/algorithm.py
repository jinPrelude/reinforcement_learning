import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from pytorch_rl import ounoise

class Net(nn.Module):
    def __init__(self, state_space, action_space):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_space, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(50, 100)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.fc3 = nn.Linear(100, 40)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(40, action_space)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions_value = self.out(x)
        return actions_value

class DDPG_actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(DDPG_actor, self).__init__()
        self.fc1 = nn.Linear(state_space, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(50, 100)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.fc3 = nn.Linear(100, 40)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(40, action_space)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        actions_value = torch.tanh(self.out(x))
        return actions_value

class DDPG_critic(nn.Module) :
    def __init__(self, state_space, action_space):
        super(DDPG_critic, self).__init__()
        self.fc1 = nn.Linear(state_space+action_space, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(50, 100)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.fc3 = nn.Linear(100, 40)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(40, 1)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.out(x)
        return q_value



class DDPG(object) :
    def __init__(self, args, state_space, action_space):
        self.args = args

        self.epsilon = 1.
        self.state_space = state_space
        self.action_space = action_space

        self.actor, self.target_actor = \
            DDPG_actor(state_space, action_space), DDPG_actor(state_space, action_space)
        self.critic, self.target_critic = \
            DDPG_critic(state_space, action_space), DDPG_critic(state_space, action_space)

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((self.args.memory_capacity, self.state_space * 2 + self.action_space + 1))  # initialize memory
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)
        self.loss_func = nn.MSELoss()
        self.OUNoise = ounoise.OUNoise(action_space, mu=0., sigma=2.0)


    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):

            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        if np.random.uniform() > self.epsilon:  # greedy
            actions_value = self.actor.forward(x)
            action = actions_value.detach().numpy()[0]  # make action from tensor to ndarray
        else:  # random
            #action = np.random.uniform(-1., 1., self.action_space)
            action = self.OUNoise.noise()
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.args.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def sample_batch_memory(self):
        sample_index = np.random.choice(self.args.memory_capacity, self.args.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_space])
        b_a = torch.LongTensor(b_memory[:, self.state_space:self.state_space + self.action_space].astype(int))
        b_r = torch.FloatTensor(
            b_memory[:, self.state_space + self.action_space:self.state_space + self.action_space + 1])
        b_s_ = torch.FloatTensor(b_memory[:, -self.state_space:])
        return b_s, b_a, b_r, b_s_

    def continue_training(self):
        model_directory = self.args.save_directory + '%d/model_%d' % (self.args.saved_iter, self.args.saved_iter)
        memory_directory = self.args.save_directory + '%d/memory_%d.npy' % (self.args.saved_iter, self.args.saved_iter)

        # load trained model
        self.actor.load_state_dict(torch.load(model_directory))
        self.target_actor.load_state_dict(torch.load(model_directory))

        # load experience replay
        self.memory = np.load(memory_directory)
        self.memory_counter = len(self.memory)

        # reset episode start position
        start = self.args.saved_iter

        self.epsilon = 0.001
        return start

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.args.target_replace_iter == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        b_s, b_a, b_r, b_s_ = self.sample_batch_memory()

        # critic update
        b_target_action = self.target_actor(b_s_)
        y_critic_input = torch.cat((b_s_, b_target_action), dim=1)
        y = b_r + self.args.gamma * self.target_critic(y_critic_input)

        q_input = np.concatenate((b_s, b_a), axis=1)
        q_input = torch.unsqueeze(torch.FloatTensor(q_input), 0)
        q = self.critic(q_input)

        q_loss = self.loss_func(y, q)
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # actor update
        actor_q_input = torch.cat((b_s, self.actor(b_s)), dim=1)
        actor_loss = -self.critic(actor_q_input).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # target network soft_update
        self.soft_update(self.target_actor, self.actor, self.args.tau)
        self.soft_update(self.target_critic, self.critic, self.args.tau)


    def kuka_train_loop(self, env):

        start = 0

        if self.args.continue_training :
            self.continue_training()

        print('\nCollecting experience...')
        for i_episode in range(start, self.args.num_episode):
            s = env.reset()
            ep_r = 0

            if self.memory_counter > self.args.memory_capacity: # decay epsilon at every episode
                self.epsilon *= self.args.epsilon_decay

            while True:

                a = self.choose_action(s)

                # take action
                s_, r, done, info = env.step(a)

                self.store_transition(s, a, r, s_)

                ep_r += r
                if self.memory_counter > self.args.memory_capacity:
                    self.learn()
                    if done:
                        if i_episode % self.args.ep_print_iter == 0:
                            print('Ep: ', i_episode,
                                  '| Ep_r: ', round(ep_r, 2), '| epsilon : ', round(self.epsilon, 2))

                            if i_episode % self.args.model_save_iter == 0:
                                directory = self.args.save_directory + '%d/'%i_episode

                                if not os.path.exists(directory):
                                    os.makedirs(directory)

                                torch.save(self.actor.state_dict(), directory + 'model_%d' % (i_episode))
                                np.save(directory + 'memory_%d'%(i_episode), self.memory)

                if done:
                    break
                s = s_

class DDPG_HER(object) :
    def __init__(self, args, state_space, action_space, goal_space, max_step) :
        self.args = args

        self.epsilon = 1.
        self.state_space = state_space
        self.action_space = action_space
        self.goal_space = goal_space
        self.max_step = max_step

        self.actor, self.target_actor = \
            DDPG_actor(state_space+goal_space, action_space), DDPG_actor(state_space+goal_space, action_space)
        self.critic, self.target_critic = \
            DDPG_critic(state_space+goal_space, action_space), DDPG_critic(state_space+goal_space, action_space)

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((self.args.memory_capacity, (self.state_space+self.goal_space) * 2 + self.action_space + self.goal_space + 1))  # initialize memory
        self.episode_memory_counter = 0  # for storing memory
        self.episode_memory = np.zeros((max_step,
                                self.state_space * 2 + self.action_space + self.goal_space + 2 + 1))  # initialize memory
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)
        self.loss_func = nn.MSELoss()
        self.OUNoise = ounoise.OUNoise(action_space, mu=0., sigma=2.0)


    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):

            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        if np.random.uniform() > self.epsilon:  # greedy
            actions_value = self.actor.forward(x)
            action = actions_value.detach().numpy()[0]  # make action from tensor to ndarray
        else:  # random
            #action = np.random.uniform(-1., 1., self.action_space)
            action = self.OUNoise.noise()
        return action

    def store_transition(self, s, a, r, s_, g):
        transition = np.hstack((s, a, r, s_, g))
        # replace the old memory with new memory
        index = self.memory_counter % self.args.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def batch_store_transition(self, b_s, b_a, b_r, b_s_, b_g):
        # transition = np.hstack((s, a, r, s_, g))
        # replace the old memory with new memory
        index = self.memory_counter % self.args.memory_capacity
        for i in range(self.args.batch_size) :
            self.memory[index+i, :(self.state_space+self.goal_space)] = b_s[i]
            self.memory[index+i, (self.state_space+self.goal_space):(self.state_space+self.goal_space) + self.action_space] = b_a[i]
            self.memory[index+i, (self.state_space+self.goal_space) + self.action_space:(self.state_space+self.goal_space) + self.action_space+1] = b_r[i]
            self.memory[index+i, (self.state_space+self.goal_space) + self.action_space+1:(self.state_space+self.goal_space)*2 + self.action_space + 1] = b_s[i]
            self.memory[index+i, -self.goal_space:] = b_g[i]
        """
        b_s = torch.FloatTensor(b_memory[:, :self.state_space])
        b_a = torch.LongTensor(b_memory[:, self.state_space:self.state_space + self.action_space].astype(int))
        b_r = torch.FloatTensor(
            b_memory[:, self.state_space + self.action_space:self.state_space + self.action_space + 1])
        b_s_ = torch.FloatTensor(b_memory[:, self.state_space + self.action_space + 1:self.state_space*2 + self.action_space + 1])
        b_g = torch.FloatTensor(b_memory[:, -self.goal_space:])
        """

        self.memory_counter += self.args.batch_size

    def episode_memory_store_transition(self, s, a, r, s_, g, gripper_pos):
        transition = np.hstack((s, a, r, s_, g, gripper_pos))
        # replace the old memory with new memory
        index = self.episode_memory_counter % self.max_step
        self.episode_memory[index, :] = transition
        self.episode_memory_counter += 1

    def sample_batch_memory(self) :
        sample_index = np.random.choice(self.args.memory_capacity, self.args.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_space])
        b_a = torch.LongTensor(b_memory[:, self.state_space:self.state_space + self.action_space].astype(int))
        b_r = torch.FloatTensor(
            b_memory[:, self.state_space + self.action_space:self.state_space + self.action_space + 1])
        b_s_ = torch.FloatTensor(b_memory[:, self.state_space + self.action_space + 1:self.state_space*2 + self.action_space + 1])
        b_g = torch.FloatTensor(b_memory[:, -self.goal_space:])
        return b_s, b_a, b_r, b_s_, b_g

    def batch_episode_memory(self):
        index = np.arange(self.max_step)
        np.random.shuffle(index)
        b_memory = self.episode_memory[index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_space])
        b_a = torch.LongTensor(b_memory[:, self.state_space:self.state_space + self.action_space].astype(int))
        b_r = torch.FloatTensor(
            b_memory[:, self.state_space + self.action_space:self.state_space + self.action_space + 1])
        b_s_ = torch.FloatTensor(
            b_memory[:, self.state_space + self.action_space + 1:self.state_space * 2 + self.action_space + 1])
        b_g = torch.FloatTensor(b_memory[:, self.state_space * 2 + self.action_space + 1:
                                            self.state_space * 2 + self.action_space + 1 + self.goal_space])
        b_gripper_pos = torch.FloatTensor(b_memory[:, -2:])
        return b_s, b_a, b_r, b_s_, b_g, b_gripper_pos

    def continue_training(self):
        model_directory = self.args.save_directory + '%d/model_%d' % (self.args.saved_iter, self.args.saved_iter)
        memory_directory = self.args.save_directory + '%d/memory_%d.npy' % (self.args.saved_iter, self.args.saved_iter)

        # load trained model
        self.actor.load_state_dict(torch.load(model_directory))
        self.target_actor.load_state_dict(torch.load(model_directory))

        # load experience replay
        self.memory = np.load(memory_directory)
        self.memory_counter = len(self.memory)

        # reset episode start position
        start = self.args.saved_iter

        self.epsilon = 0.001
        return start

    def r(self, gripper_pos, g):

            mean = abs(gripper_pos - g)
            mean = np.sum(mean, axis=-1)
            if mean < 0.1 :
                return 0
            else :
                return -1

    def r_batch(self, gripper_pos, g):

            mean = abs(gripper_pos - g)
            mean = np.sum(mean, axis=-1)
            result = []
            for i in range(mean.size) :
                if mean[i] < 0.1 :
                    result.append(0)
                else :
                    result.append(-1)
            return result

    def learn(self):    # target parameter update
        if self.learn_step_counter % self.args.target_replace_iter == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        b_s, b_a, b_r, b_s_, b_g = self.sample_batch_memory()
        b_sg = torch.cat((b_s, b_g), dim=1)
        b_s_g = torch.cat((b_s_, b_g), dim=1)

        # critic update
        b_target_action = self.target_actor(b_s_g)
        y_critic_input = torch.cat((b_s_g, b_target_action), dim=1)
        y = b_r + self.args.gamma * self.target_critic(y_critic_input)

        q_input = np.concatenate((b_sg, b_a), axis=1)
        q_input = torch.unsqueeze(torch.FloatTensor(q_input), 0)
        q = self.critic(q_input)

        q_loss = self.loss_func(y, q)
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # actor update
        actor_q_input = torch.cat((b_sg, self.actor(b_sg)), dim=1)
        actor_loss = -self.critic(actor_q_input).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # target network soft_update
        self.soft_update(self.target_actor, self.actor, self.args.tau)
        self.soft_update(self.target_critic, self.critic, self.args.tau)

    def kuka_train_loop(self, env):

        start = 0

        if self.args.continue_training :
            self.continue_training()

        print('\nCollecting experience...')
        for i_episode in range(start, self.args.num_episode):
            s, g = env.reset()
            ep_r = 0

            if self.memory_counter > self.args.batch_size: # decay epsilon at every episode
                self.epsilon *= self.args.epsilon_decay

            while True:

                sg = np.concatenate((s, g), axis=0)

                a = self.choose_action(sg)

                # take action
                s_, r, done, gripper_pos, info = env.step(a)

                s_g = np.concatenate((s_, g), axis=0)

                r = self.r(gripper_pos, g)

                self.episode_memory_store_transition(s, a, r, s_, g, gripper_pos)
                # self.store_transition(sg, a, r, s_g, g)


                ep_r += r

                if done:
                    if i_episode % self.args.ep_print_iter == 0:
                        print('Ep: ', i_episode,
                              '| Ep_r: ', round(ep_r, 2), '| epsilon : ', round(self.epsilon, 2))

                        if i_episode % self.args.model_save_iter == 0:
                            directory = self.args.save_directory + '%d/' % i_episode

                            if not os.path.exists(directory):
                                os.makedirs(directory)

                            torch.save(self.actor.state_dict(), directory + 'model_%d' % (i_episode))
                            np.save(directory + 'memory_%d' % (i_episode), self.memory)
                    break

                s = s_


            b_s, b_a, b_r, b_s_, b_g, b_gripper_pos = self.batch_episode_memory()
            b_r = self.r_batch(b_gripper_pos.detach().numpy(), b_g.detach().numpy())

            b_sg = np.concatenate((b_s, b_g), axis=1)
            b_s_g = np.concatenate((b_s_, b_g), axis=1)

            self.batch_store_transition(b_sg, b_a, b_r, b_s_g, b_g)

            _, _, _, _, _, b_gripper_pos_2 = self.batch_episode_memory()
            b_g_ = b_gripper_pos_2         # b_s 대신 b_s_로 함. 어느게 맞는지 아직 잘 모르겠음


            b_r_ = self.r_batch(b_gripper_pos.detach().numpy(), b_g_.detach().numpy())
            b_sg_ = np.concatenate((b_s, b_g_), axis=1)
            b_s_g_ = np.concatenate((b_s_, b_g_), axis=1)
            self.batch_store_transition(b_sg, b_a, b_r_, b_s_g, b_g_)

            for _ in range(100) :
                self.learn()





class DQN(object):
    def __init__(self, args, state_space, action_space):

        self.args = args

        self.epsilon = 1.
        self.state_space = state_space
        self.action_space = action_space

        self.eval_net, self.target_net = \
            Net(state_space, action_space), Net(state_space, action_space)
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((self.args.memory_capacity, self.state_space * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.args.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):

        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() > self.epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]  # if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.action_space)
            action = action     # if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.args.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def sample_batch_memory(self):
        sample_index = np.random.choice(self.args.memory_capacity, self.args.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_space])
        b_a = torch.LongTensor(b_memory[:, self.state_space:self.state_space + self.action_space].astype(int))
        b_r = torch.FloatTensor(
            b_memory[:, self.state_space + self.action_space:self.state_space + self.action_space + 1])
        b_s_ = torch.FloatTensor(b_memory[:, -self.state_space:])
        return b_s, b_a, b_r, b_s_

    def continue_training(self):
        model_directory = self.args.save_directory + '%d/model_%d' % (self.args.saved_iter, self.args.saved_iter)
        memory_directory = self.args.save_directory + '%d/memory_%d.npy' % (self.args.saved_iter, self.args.saved_iter)

        # load trained model
        self.eval_net.load_state_dict(torch.load(model_directory))
        self.target_net.load_state_dict(torch.load(model_directory))

        # load experience replay
        self.memory = np.load(memory_directory)
        self.memory_counter = len(self.memory)

        # reset episode start position
        start = self.args.saved_iter

        self.epsilon = 0.001

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.args.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        b_s, b_a, b_r, b_s_ = self.sample_batch_memory()

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s)
        _, q_eval_argmax = torch.max(q_eval, 1)
        q_eval = q_eval.gather(1, b_a) # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        # q_next = q_next.gather(1, b_a)              #double_dqn
        q_target = b_r + self.args.gamma * q_next.max(1)[0].view(self.args.batch_size, 1)   # shape (batch, 1)
        # q_target = b_r + self.gamma * q_next.view(self.batch_size, 1)  # double_dqn
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def kuka_train_loop(self, env):

        start = 0

        if self.args.continue_training :
            self.continue_training()

        print('\nCollecting experience...')
        for i_episode in range(start, self.args.num_episode):
            s = env.reset()
            ep_r = 0

            if self.memory_counter > self.args.memory_capacity: # decay epsilon at every episode
                self.epsilon *= self.args.epsilon_decay

            while True:

                a = self.choose_action(s)

                # take action
                s_, r, done, info = env.step(a)

                self.store_transition(s, a, r, s_)

                ep_r += r
                if self.memory_counter > self.args.memory_capacity:
                    self.learn()
                    if done:
                        if i_episode % self.args.ep_print_iter == 0:
                            print('Ep: ', i_episode,
                                  '| Ep_r: ', round(ep_r, 2), '| epsilon : ', self.epsilon)
                            if i_episode % self.args.model_save_iter == 0:

                                directory = self.args.save_directory + '%d/' % i_episode

                                if not os.path.exists(directory):
                                    os.makedirs(directory)

                                torch.save(self.eval_net.state_dict(), directory + 'model_%d' % (i_episode))
                                np.save(directory + 'memory_%d' % (i_episode), self.memory)

                if done:
                    break
                s = s_