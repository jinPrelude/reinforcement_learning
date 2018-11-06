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
                                self.state_space * 2 + self.action_space + self.goal_space*2 + 1))  # initialize memory
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

    # 결과를 스텝마다 메모리에 저장할 때 사용하는 함수입니다.
    def store_transition(self, s, a, r, s_, g):
        transition = np.hstack((s, a, r, s_, g))
        # replace the old memory with new memory
        index = self.memory_counter % self.args.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    # 결과를 에피소드가 끝나고 한번에 미니배치 사이즈로 저장할 때 사용하는 함수입니다.
    def batch_store_transition(self, b_s, b_a, b_r, b_s_, b_g):
        # transition = np.hstack((s, a, r, s_, g))
        # replace the old memory with new memory
        index = self.memory_counter % self.args.memory_capacity
        for i in range(self.episode_memory_counter) :
            self.memory[index+i, :(self.state_space+self.goal_space)] = b_s[i]
            self.memory[index+i, (self.state_space+self.goal_space):(self.state_space+self.goal_space) + self.action_space] = b_a[i]
            self.memory[index+i, (self.state_space+self.goal_space) + self.action_space:(self.state_space+self.goal_space) + self.action_space+1] = b_r[i]
            self.memory[index+i, (self.state_space+self.goal_space) + self.action_space+1:(self.state_space+self.goal_space)*2 + self.action_space + 1] = b_s_[i]
            self.memory[index+i, -self.goal_space:] = b_g[i]


        self.memory_counter += self.episode_memory_counter

    # 임시 메모리에 저장하는 함수입니다.
    def episode_memory_store_transition(self, s, a, r, s_, g, gripper_pos):
        transition = np.hstack((s, a, r, s_, g, gripper_pos))
        # replace the old memory with new memory
        #index = self.episode_memory_counter % self.max_step
        self.episode_memory[self.episode_memory_counter, :] = transition
        self.episode_memory_counter += 1

    # 메모리에서 미니배치 크기만큼 무작위로 가져오는 함수입니다.
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

    # 임시 메모리에서 무작위로 가져오는 함수입니다.
    def batch_episode_memory(self):
        index = np.arange(self.episode_memory_counter)
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
        b_gripper_pos = torch.FloatTensor(b_memory[:, -self.goal_space:])

        return b_s, b_a, b_r, b_s_, b_g, b_gripper_pos

    # 논문에 나오는 r 함수.. 아직 구현을 하지 못했습니다..
    # reward shaping 을 어떻게 해야할까요..
    def r(self, gripper_pos, g):
        gripper_pos = np.float32(gripper_pos)
        g = np.float32(g)  # 2개 타입이 달라서 값이 같아도 false 로 나오는 문제때문에
        result = []
        for i in range(self.episode_memory_counter) :
            test1 = gripper_pos[i]
            test2 = g[i]
            test3 = (test1 == test2)      # 분명 같은데 왜 다르다고 나올까
            if (gripper_pos[i] == g[i]).all() :
                result.append(0)
            else :
                result.append(-1)
        return np.asarray(result)

    def learn(self):    # target parameter update
        if self.learn_step_counter % self.args.target_replace_iter == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
        self.learn_step_counter += 1

        # 랜덤으로 배치를 뽑아서 state 와 goal 을 합쳐줍니다(concatenate).
        b_s, b_a, b_r, b_s_, b_g = self.sample_batch_memory()
        b_sg = torch.cat((b_s, b_g), dim=1)
        b_s_g = torch.cat((b_s_, b_g), dim=1)

        # critic update
        # critic을 업데이트 해줍니다.
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
        # actor 을 업데이트 해줍니다.
        actor_q_input = torch.cat((b_sg, self.actor(b_sg)), dim=1)
        actor_loss = -self.critic(actor_q_input).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # 타겟 네트워크를 업데이트해줍니다!
        self.soft_update(self.target_actor, self.actor, self.args.tau)
        self.soft_update(self.target_critic, self.critic, self.args.tau)

    def test_episode(self, env):
        while True :
            a = np.zeros(self.action_space)
            _, _, done, _, _ = env.step(a)
            if done :
                print('test episode finished')
                break

    def kuka_train_loop(self, env):

        start = 0

        if self.args.continue_training :
            self.continue_training()

        # self.test_episode(env)
        print('\nCollecting experience...')
        for i_episode in range(start, self.args.num_episode):
            s, g = env.reset()
            ep_r = 0
            episode_len = 0
            if self.memory_counter > self.args.batch_size: # decay epsilon at every episode
                self.epsilon *= self.args.epsilon_decay

            while True:

                # action 과 goal 을 합쳐줍니다.
                sg = np.concatenate((s, g), axis=0)

                # epsilon 에 따라서 랜덤액션을 할것인지 모델을 쓸건지 정합니다.
                a = self.choose_action(sg)

                # take action
                # 액션을 취해줍니다.
                s_, r, done, gripper_pos, info = env.step(a)

                if done:

                    # 에피소드 결과 출력해주고 적당한 시점 되면 모델 저장해주고.. 하는 부분입니다ㅎ her 과는 무관합니다!
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

                # 임시 메모리입니다! 에피소드를 끝내고 r 함수로 에피소드의 리워드들을 다시 shaping 해주고 메모리에 저장하는
                # 작업을 위해 잠시 저장해놓는 공간입니다!
                self.episode_memory_store_transition(s, a, r, s_, g, gripper_pos)
                episode_len += 1

                s_T = gripper_pos
                ep_r += r
                s = s_


            # 임시 메모리에서 순서 무작위로 불러왔습니다!
            b_s, b_a, b_r, b_s_, b_g, b_gripper_pos = self.batch_episode_memory()

            # 이부분은 her 논문에 있는 r 함수와 같은 기능을 구현하고자 한 부분인데..
            # 아직 r 함수 구현을 하지 않았습니다.. 어떻게 만들어야 할까요..
            b_r = self.r(b_gripper_pos.detach().numpy(), b_g.detach().numpy())

            # r 함수로 리워드 shaping을 끝내고 정말로 메모리에 저장하기 위해 state와 goal 을 더해줍니다.
            b_sg = np.concatenate((b_s, b_g), axis=1)
            b_s_g = np.concatenate((b_s_, b_g), axis=1)

            # 그리고 저장합니다.
            self.batch_store_transition(b_sg, b_a, b_r, b_s_g, b_g)

            # 논문 pseudo 코드에서 "Sample a set of additional goals for replay G := S(current episode)" 에 해당하는 부분입니다!
            # _, _, _, _, _, b_gripper_pos_2 = self.batch_episode_memory(episode_len)
            b_g_ = np.asarray([s_T for _ in range(self.episode_memory_counter)])      # final strategy

            # her 부분입니다! 바뀐 goal 에 대해서 다시 reward shaping을 해주고
            # state와 goal 을 함쳐준 다음 메모리에 저장합니다.

            # no r function defined
            b_r_ = self.r(b_gripper_pos.detach().numpy(), b_g_)
            b_sg_ = np.concatenate((b_s, b_g_), axis=1)
            b_s_g_ = np.concatenate((b_s_, b_g_), axis=1)

            # NOTE : fix b_r to b_r_ if function r is ready
            self.batch_store_transition(b_sg_, b_a, b_r_, b_s_g_, b_g_)
            self.episode_memory_counter = 0

            for _ in range(100) :

                # 100번 메모리를 통해 DDPG 학습을 시켜줍니다ㅎ
                self.learn()
