import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from pytorch_rl import ounoise

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

    def simple_one_to_one(self, env):

        if self.args.continue_training :
            start = self.continue_training()
        else :
            start = 0
        print('\nCollecting experience...')
        for i_episode in range(self.args.num_episode):

            if self.memory_counter > self.args.memory_capacity:  # decay epsilon at every episode
                self.epsilon *= self.args.epsilon_decay
            self.OUNoise.reset()
            s = env.reset()
            ep_r = 0
            iter = 0
            while True:

                if self.args.render:
                    env.render()

                a = self.choose_action(s)
                # take action
                s_, r, done, info = env.step(a)

                self.store_transition(s, a, r, s_)

                ep_r += r
                if self.memory_counter > self.args.memory_capacity:
                    self.learn()

                    if done or iter > self.args.episode_len:
                        if i_episode % self.args.ep_print_iter == 0:
                            print('Ep: ', i_episode,
                                  '| Ep_r: ', round(ep_r, 2), '| epsilon : ', self.epsilon)
                            if i_episode % self.args.model_save_iter == 0:

                                directory = self.args.save_directory + '%d/' % i_episode

                                if not os.path.exists(directory):
                                    os.makedirs(directory)

                                torch.save(self.actor.state_dict(), directory + 'model_%d' % (i_episode))
                                np.save(directory + 'memory_%d' % (i_episode), self.memory)


                        break

                if done or iter > self.args.episode_len :
                    print('iter : ', iter)
                    break
                iter += 1

                s = s_
