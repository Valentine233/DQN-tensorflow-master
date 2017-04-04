#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .base_m import BaseModel
from .ops_m import linear, conv2d
from .replay_memory_m import ReplayMemory
from utils_m import get_time, save_pkl, load_pkl

class Agent(BaseModel):
  def __init__(self, config, virtual_user, sess):
    super(Agent, self).__init__(config)
    self.sess = sess
    self.weight_dir = 'weights'

    self.env = virtual_user
    self.memory = ReplayMemory(self.config, self.model_dir)

    # 为了在从检查点恢复的时候，能恢复执行的步数，所以这里建立了一个变量专门用来存放步数数据
    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')        # 存储检查点所处步数的变量
      self.step_input = tf.placeholder('int32', None, name='step_input') # 建立输入变量
      self.step_assign_op = self.step_op.assign(self.step_input)         # 将step_input指派给step_op

    self.build_dqn()

  def train(self):
    """train net.
    This is main loop for train net.
    """
    start_step = self.step_op.eval() # 初始化起始步数；当从检查点加载的时候，能恢复执行步数
    start_time = time.time()
    # @num_retrieval是查询次数；@ep_reward是一次查询中的总得分;这些数据全部是用作统计
    # update_count: the number of step in this test step
    num_retrieval, self.update_count, ep_reward = 0, 0, 0.
    total_reward, self.total_loss, self.total_q = 0., 0., 0.
    max_avg_ep_reward = 0
    ep_rewards, actions = [], [] # @actions 是行动序列

    # initialization
    terminal = True
    while(terminal==True):
        screen, reward, action, terminal = self.env.new_random_start()
    self.memory.update_state(screen, action)

    # This is main loop for train.
    for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
      # when first train, initialize related parameters.
      if self.step == self.learn_start:
        num_retrieval, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        ep_rewards, actions = [], []  # @ep_rewards 用来记录每次查询的得分，用作统计数据

      # 1. predict
      action = self.predict(self.memory.curr_state)
      # 2. act
      screen, reward, terminal = self.env.state_transfer(action, is_training=True)
      # 3. observe
      self.observe(screen, reward, action, terminal) # 记录收到的数据，在一定的时候训练网络和更新网络

      # if retrieval over, then reset it.
      if terminal:
        screen, reward, action, terminal = self.env.new_random_start()
        num_retrieval += 1  # 记录查询次数
        ep_rewards.append(ep_reward) # 将本次查询的最终得分记录下来，下面用来做统计
        ep_reward = 0.
      else:
        ep_reward += reward # 本次查询获得的分数

      actions.append(action)
      total_reward += reward

      # 开始训练之后，每隔test_step步，做一次统计
      if self.step >= self.learn_start:
        if self.step % self.test_step == self.test_step - 1:
          avg_reward = total_reward / self.test_step     # 本论test_step步中的平均奖励；total_reward在下面会清零
          avg_loss = self.total_loss / self.update_count # 本论test_step步中的平均损失；...
          avg_q = self.total_q / self.update_count       # 本论test_step步中的平均Q；

          try:
            max_ep_reward = np.max(ep_rewards)  # 开始查询以来获得的最高分
            min_ep_reward = np.min(ep_rewards)
            avg_ep_reward = np.mean(ep_rewards)
          except:
            max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

          print '\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # retrieval: %d' \
              % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_retrieval)

          # ?
          if max_avg_ep_reward * 0.9 <= avg_ep_reward:
            self.step_assign_op.eval({self.step_input: self.step + 1}) # 将当前执行的步数写入变量self.step_op中,方便存入检查点
            self.save_model(self.step + 1) # save current model data as a checkpoint.

            max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

          if self.step > 180:
            self.inject_summary({
                'average.reward': avg_reward,
                'average.loss': avg_loss,
                'average.q': avg_q,
                'episode.max reward': max_ep_reward,
                'episode.min reward': min_ep_reward,
                'episode.avg reward': avg_ep_reward,
                'episode.num of retrieval': num_retrieval,
                'episode.rewards': ep_rewards,
                'episode.actions': actions,
                'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
              })

          # 完成本论统计之后，重置相关
          num_retrieval = 0
          total_reward = 0.
          self.total_loss = 0.
          self.total_q = 0.
          self.update_count = 0
          ep_reward = 0.
          ep_rewards = []
          actions = []


  def predict(self, s_t, test_ep=None):
    """predict the best action based on input state @s_t.

    args:
       s_t    : screen data, pre-action.
       test_ep: if @test_ep==None,.

    """
    # 当test_ep==None时，随机行动的概率会从ep_start逐渐降到ep_end .
    # 计算公式为 ep_end+(ep_start-ep_end)*f(step)
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))
    # step越大，随机概率越小（探索）

    # 随机-贪婪算法，有一定的概率随机选择下一次行动
    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      # @self.q_action is index of max_a' Q(s',a').
      action = self.q_action.eval({self.s_t: [s_t]})[0]

    return action


  def observe(self, screen, reward, action, terminal):
    """观察收到的数据，在一定的时候，训练自己的网络和更新网络。
    """
    # 限定收到的reward必须在区间 [min_reward,max_reward]中
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.memory.update_state(screen, action)
    self.memory.add(screen, reward, action, terminal) #record this episode to memory.

    # 只有在积累了一定的记忆以后，才开始第一次训练网络
    if self.step > self.learn_start:
      # 限制训练频率
      if self.step % self.train_frequency == 0:
        self.q_learning_mini_batch()
      # 限制target网络的更新频率
      if self.step % self.target_q_update_step == self.target_q_update_step - 1:
        self.update_target_q_network()


  def q_learning_mini_batch(self):
    """train the prediction net.
    There are two method, DQN and Double DQN.
    """
    # ReplayMemory中存储的episode至少要大于1，否则没法训练
    if self.memory.count < 1:
      return
    else:
      # 从ReplayMemory随机的采集a batch of 经验
      s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

    t = time.time()
    if self.double_q:
      # Double Q-learning
      pred_action = self.q_action.eval({self.s_t: s_t_plus_1})

      q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
        self.target_s_t: s_t_plus_1,
        self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
      })
      target_q_t = (1. - terminal) * self.discount * q_t_plus_1_with_pred_action + reward
    else:
      # DQN or ...
      # @s_t_plus_1 is s'.
      # @self.target_q is the target net.
      # @q_t_plus_1 is Q(s',a').
      q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

      terminal = np.array(terminal) + 0.
      # @max_q_t_plus_1 is max_a' Q(s',a')
      max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
      # this is Bellman equation. r+v*max_a' Q(s',a').
      # @(1. - terminal) means terminal episode will get zero reward.This will make the net avoid terminal episode.
      target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

    # 这里的目的是训练 prediction网络，采用的self.optim训练器；其余的self.q，self.loss,self.q_summary都是为了统计
    _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.q_summary], {
      self.target_q_t: target_q_t,
      self.action: action,
      self.s_t: s_t,
      self.learning_rate_step: self.step,
    })

    self.writer.add_summary(summary_str, self.step)
    self.total_loss += loss
    self.total_q += q_t.mean()
    self.update_count += 1


  def build_dqn(self):
    """build DQN model.
    Bellman equation is Q(s,a) = r+v*max_a'(s',a'). Loss function is [ r + v*max_a' Q(s',a') - Q(s,a)].
    ?In this model, prediction net is working as max_a' Q(s',a'). target net is working as Q(s,a).
    """
    self.w = {}   # weight set for prediction net.
    self.t_w = {} # weight set for target net.

    #initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.truncated_normal_initializer(0, 0.02) #?truncated normal distribution
    activation_fn = tf.nn.relu                             #activation function

    # training network
    # prediction net is ,which working for compute max_a' Q(s',a') and Q(s,a).
    # 这个网络是核心网络，用于近似Q函数，下面的target网络用于辅助该网络进行训练
    with tf.variable_scope('prediction'):
      # build input array.
      if self.cnn_format == 'NHWC':
        self.s_t = tf.placeholder('float32',
            [None, self.screen_width, self.screen_height, self.history_length], name='s_t')
      else:
        self.s_t = tf.placeholder('float32',
            [None, self.history_length, self.screen_width, self.screen_height], name='s_t')

      #layer 1-3
      self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1') #output=>[-1,(84-7)/4=19,19,32]
      self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2')
      self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3')

      #flat layer-3 as a 2-D numpy array,[batch_num,height*width*channels]
      shape = self.l3.get_shape().as_list()
      self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        # dueling algorithm use this branch.
        self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')

        self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')

        self.value, self.w['val_w_out'], self.w['val_w_b'] = \
          linear(self.value_hid, 1, name='value_out')

        self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
          linear(self.adv_hid, self.env.action_size, name='adv_out')

        # Average Dueling
        self.q = self.value + (self.advantage -
          tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
      else:
        # DQN algorithm use this branch.
        self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
        self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.env.action_size, name='q')

      self.q_action = tf.argmax(self.q, dimension=1) #get index(action) to max q-value

      q_summary = []
      avg_q = tf.reduce_mean(self.q, 0) #summary
      for idx in xrange(self.env.action_size):
        q_summary.append(tf.histogram_summary('q/%s' % idx, avg_q[idx]))
      self.q_summary = tf.merge_summary(q_summary, 'q_summary')

    # target network
    # 根据DeepMind论文介绍，对Q(s','a)对应的网络采用延迟更新的策略，target网络用于保存延迟的网络；
    # 该网络为训练prediction网络提供Q(s',a')值，每隔一定的步数从prediction网络复制一次，自身并不训练
    with tf.variable_scope('target'):
      if self.cnn_format == 'NHWC':
        self.target_s_t = tf.placeholder('float32',
            [None, self.screen_width, self.screen_height, self.history_length], name='target_s_t')
      else:
        self.target_s_t = tf.placeholder('float32',
            [None, self.history_length, self.screen_width, self.screen_height], name='target_s_t')

      self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t,
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='target_l1')
      self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='target_l2')
      self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l3')

      shape = self.target_l3.get_shape().as_list()
      self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        self.t_value_hid, self.t_w['l4_val_w'], self.t_w['l4_val_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_value_hid')

        self.t_adv_hid, self.t_w['l4_adv_w'], self.t_w['l4_adv_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_adv_hid')

        self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = \
          linear(self.t_value_hid, 1, name='target_value_out')

        self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = \
          linear(self.t_adv_hid, self.env.action_size, name='target_adv_out')

        # Average Dueling
        self.target_q = self.t_value + (self.t_advantage -
          tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
      else:
        self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4')
        self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
            linear(self.target_l4, self.env.action_size, name='target_q')

      # self.target_q是一个二维数组，结构为[ batch_num, Q-action value]. 代表了网络对当前输入状态下，每个
      # 动作的期望评分。target_q_idx代表了所选择的动作编号。
      self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
      self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

    # 给target网络设置指派操作，在之后的update_target_q_network函数中用来修改target网络的权重值
    with tf.variable_scope('pred_to_target'):
      self.t_w_input = {}
      self.t_w_assign_op = {} # 用于存放指派操作的字典，在update_target_q_network函数中会使用

      for name in self.w.keys():
        self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
        self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

    # optimizer
    # 这里是网络的优化部分
    with tf.variable_scope('optimizer'):
      self.target_q_t = tf.placeholder('float32', [None], name='target_q_t') # 输入的数据是 r+discount_factor*max_a' Q(s',a')
      self.action = tf.placeholder('int64', [None], name='action') # 输入的action张量

      action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot') # action的one_hot形式
      #self.q是prediction网络的输出Q；q_acted的格式是[Q(s_1,a_*),Q(s_2,a_*),Q(s_3,a_*),...]，每个元素代表了对应(s,a)的Q值
      q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

      self.delta = self.target_q_t - q_acted  # 计算 delta = r+discount_factor*max_a' Q(s',a') - Q(s,a)
      self.clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta') #裁剪

      self.global_step = tf.Variable(0, trainable=False)

      self.loss = tf.reduce_mean(tf.square(self.clipped_delta), name='loss') # loss function
      self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step') #
      self.learning_rate_op = tf.maximum(self.learning_rate_minimum,   # 建立指数衰减的学习率
          tf.train.exponential_decay(
              self.learning_rate,
              self.learning_rate_step,
              self.learning_rate_decay_step,
              self.learning_rate_decay,
              staircase=True))
      self.optim = tf.train.RMSPropOptimizer(
          self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

    # summary model information
    with tf.variable_scope('summary'):
      scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
          'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of retrieval', 'training.learning_rate']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.scalar_summary("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

      histogram_summary_tags = ['episode.rewards', 'episode.actions']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.histogram_summary(tag, self.summary_placeholders[tag])

      self.writer = tf.train.SummaryWriter('./logs/%s' % self.model_dir, self.sess.graph)

    tf.initialize_all_variables().run()

    # 保存prediction网络中的所有权重变量，以及...
    self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)

    self.load_model() #load checkpoint data, in base.py
    self.update_target_q_network()


  def update_target_q_network(self):
    """将prediction网络中的权重复制到target网络中。
    """
    for name in self.w.keys():
      self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})


  def save_weight_to_pkl(self):
    if not os.path.exists(self.weight_dir):
      os.makedirs(self.weight_dir)

    for name in self.w.keys():
      save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))


  def load_weight_from_pkl(self, cpu_mode=False):
    with tf.variable_scope('load_pred_from_pkl'):
      self.w_input = {}
      self.w_assign_op = {}

      for name in self.w.keys():
        self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
        self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

    for name in self.w.keys():
      self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})

    self.update_target_q_network()

  def inject_summary(self, tag_dict):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)

  def test(self, n_step=10000, n_episode=100, test_ep=None):
    if test_ep == None:
      test_ep = self.ep_end

    test_memory = ReplayMemory(self.config, model_dir=None)

    # if not self.display:
    #   gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
    #   self.env.env.monitor.start(gym_dir)

    best_reward, best_idx = 0, 0
    for idx in xrange(n_episode):
      screen, reward, action, terminal = self.env.new_random_start()
      current_reward = 0
      test_memory.update_state(screen, action)

      for t in tqdm(range(n_step), ncols=70):
        # 1. predict
        action = self.predict(test_memory.curr_state, test_ep)
        # 2. act
        screen, reward, terminal = self.env.state_transfer(action, is_training=False)
        # 3. observe
        test_memory.update_state(screen, action)
        current_reward += reward
        if terminal:
          break

      if current_reward > best_reward:
        best_reward = current_reward
        best_idx = idx

      print "="*30
      print " [%d] Best reward : %d" % (best_idx, best_reward)
      print "="*30

#     if not self.display:
#       self.env.env.monitor.close()
      #gym.upload(gym_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')
