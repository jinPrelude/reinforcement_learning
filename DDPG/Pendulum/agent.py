
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


        with tf.variable_scope('actor_network') :
            self.inputs, self.out, self.scaled_out = self.create_actor_network()


        self.network_params = tf.trainable_variables()


        with tf.variable_scope('target_actor_network') :
            self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()


        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]


        # I want green!
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau)) for i in range(len(self.target_network_params))]


        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])


        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        self.optimize = \
            tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):

        inputs = tf.placeholder(name='actor_input', shape=[None, self.s_dim], dtype=tf.float32)
        w1 = tf.get_variable(name='actor_w1', shape=[self.s_dim, 10], initializer=tf.random_uniform_initializer(-0.3, 0.3))
        l1 = tf.matmul(inputs, w1, name='actor_layer1')
        l1 = tf.nn.relu(l1)

        w2 = tf.get_variable(name='actor_w2', shape=[10, 10], initializer=tf.random_uniform_initializer(-0.3, 0.3))
        l2 = tf.matmul(l1, w2)
        l2 = tf.nn.relu(l2)

        w3 = tf.get_variable(name='actor_w3', shape=[10, 6], initializer=tf.random_uniform_initializer(-0.3, 0.3))
        l3 = tf.matmul(l2, w3)
        l3 = tf.nn.relu(l3)

        w4 = tf.get_variable(name='actor_w4', shape=[6, self.a_dim], initializer=tf.random_uniform_initializer(-0.3, 0.3))
        l4 = tf.matmul(l3, w4)
        out = tf.nn.tanh(l4)
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out


    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

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

        with tf.variable_scope('critic_network') :
            self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        with tf.variable_scope('target_critic_network') :
            self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.out))
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)
        action = tf.placeholder(shape=[None, self.a_dim], dtype=tf.float32)


        w1 = tf.get_variable(name='critic_w1', shape=[self.s_dim, 400], initializer=tf.random_uniform_initializer(-0.3, 0.3))
        l1 = tf.matmul(inputs, w1)
        l1 = tf.nn.relu(l1)


        w2 = tf.get_variable(name='criti_w2', shape=[400, 300], initializer=tf.random_uniform_initializer(-0.3, 0.3))


        w2_a  = tf.get_variable(name='criticc_w2_action', shape=[self.a_dim, 300], initializer=tf.random_uniform_initializer(-0.3, 0.3))
        l2 = tf.nn.relu(tf.matmul(l1, w2) + tf.matmul(action, w2_a))


        w3 = tf.get_variable(name='critic_w3', shape=[300, 1], initializer=tf.random_uniform_initializer(-0.3, 0.3))
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


