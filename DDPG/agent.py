
import tensorflow as tf

class ActorNetwork(object):


    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network를 생성합니다.
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        # It stores the parameters the network has.
        #
        self.network_params = tf.trainable_variables()

        # Target Actor network를 생성합니다.
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        # It stores the parameters the target network has.
        # We should slice the tf.trainable_variables() because unlike
        # network_params target_actor_network is has made above.
        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]


        # Op for periodically updating target network with online network
        # weights
        # update_target_network_params = tau*t theta[i] + (1-tau) * target_theta[i]
        # .assign은 assign() 괄호 안의 내용대로 변수의 값을 변경해주는 함수인듯
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau)) for i in range(len(self.target_network_params))]

        # critic network에게 제공받을 placeholder입니다. action의 gradient입니다.
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        # ys인 scaled_out을 xs인 network_params에 대해서 미분하고, y_grads인 -self.action_gradient를 곱해준다.
        # y_grads 가 ys와 같은 독립변수를 포함하는 함수이면 placeholder에 따라 변경된 y_grads와 scaled_out이 곱해져서
        # 편하다. self.action_gradient 가 음수인 이유는 gradient ascent 가 아닌 gradient descent를 해야 하기 때문이다.
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # 최적화 부분
        # optimization part
        # actor_gradients 의 값을 Adam optimizer 을 이용해서 network_params에 변동사항을 적용하는 부분인듯. 그래서
        # actor_gradients랑 network_params를 zip으로 묶어둔 듯 하다.
        self.optimize = \
            tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        #훈련시킬 네트워크 파라메터가 몇 개인지 저장한다.
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    # Define actor neural network
    def create_actor_network(self):

        inputs = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)
        w1 = tf.Variable(tf.random_normal(shape=[self.s_dim, 10], mean=0., stddev=0.1), name='w1')
        l1 = tf.matmul(inputs, w1)
        l1 = tf.nn.relu(l1)

        w2 = tf.Variable(tf.random_normal(shape=[10, 10], mean=0., stddev=0.1), name='w2')
        l2 = tf.matmul(l1, w2)
        l2 = tf.nn.relu(l2)

        w3 = tf.Variable(tf.random_normal(shape=[10, 6], mean=0., stddev=0.1), name='w3')
        l3 = tf.matmul(l2, w3)
        l3 = tf.nn.relu(l3)

        w4 = tf.Variable(tf.random_normal(shape=[6, self.a_dim], mean=0., stddev=0.1), name='w4')
        l4 = tf.matmul(l3, w4)
        out = tf.nn.tanh(l4)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out


    # action의 gradient 와 inputs(state)를 입력으로 받아 self.optimize를 돌려서 학습합니다.
    # Train by running self.optimize which gets the gradient of the action and inputs(state) as a inputs
    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    # input을 받아 예측한 행동을 반환합니다.
    # Choose and return the action of the actor network
    # by getting input(state) as a input
    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    # target network의 행동 예측값을 반환합니다.
    # Choose and return the action of the target actor network
    # by getting input(state) as a input
    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    # target network를 self.update_target_network_params를 이용해 업데이트합니다.
    # Update the target network by using self.update_target_network_params
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    # 정채불명
    # Unconfirmed
    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # critic network를 생성합니다.
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target critic network를 생성합니다.
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        #target critic network에 y_i 값으로 제공될 placeholder입니다.
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # loss를 정의하고 최적화합니다.
        #self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.out))
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # action에 대해서 신경망의 gradient를 구합니다.
        # 미니배치의 각 critic 출력의 (각 액션에 의해서 구해진)기울기를 합산한다.
        # 모든 출력은 자신이 나눠진 action을 제외한 모든 action에 대해 독립적이다.
        self.action_grads = tf.gradients(self.out, self.action)

    # Critic network를 정의합니다.
    # Define the critic network
    def create_critic_network(self):
        inputs = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)
        action = tf.placeholder(shape=[None, self.a_dim], dtype=tf.float32)

        w1 = tf.Variable(tf.random_uniform(shape=[self.s_dim, 400], maxval=0.3, minval=-0.3), dtype=tf.float32)
        l1 = tf.matmul(inputs, w1)
        l1 = tf.nn.relu(l1)
        w2 = tf.Variable(tf.random_uniform(shape=[400, 300], maxval=0.3, minval=-0.3), dtype=tf.float32)

        # action에 가중치를 곱해서 critic network에 더해준다. 경험적으로 좋은 결과를 이끌어냈다고 함.

        w2_a = tf.Variable(tf.random_uniform(shape=[self.a_dim, 300],  maxval=0.3, minval=-0.3), dtype=tf.float32)
        l2 = tf.nn.relu(tf.matmul(l1, w2) + tf.matmul(action, w2_a))

        w3 = tf.Variable(tf.random_uniform(shape=[300, 1], maxval=0.03, minval=-0.03), dtype=tf.float32)
        out = tf.matmul(l2, w3)


        return inputs, action, out


    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


