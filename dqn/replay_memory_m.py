#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import os
import random
import logging
import numpy as np

from utils_m import save_npy, load_npy

class ReplayMemory:
  """
  """
  def __init__(self, config, model_dir):
    self.model_dir = model_dir
    self.screen_height, self.screen_width = config.screen_height, config.screen_width
    self.cnn_format = config.cnn_format # 'NHWC' or 'NCHW'
    self.memory_size = config.memory_size # 存储的帧数
	# 根据Bellman等式，一条完整的经验应该包含 (s,a,r,s').其中s和s'包含在
    self.actions = np.empty(self.memory_size, dtype = np.uint8)
    self.rewards = np.empty(self.memory_size, dtype = np.integer) #r
	# 这里的存储单位是帧，不是state
    self.screens = np.empty((self.memory_size, self.screen_height, self.screen_width), dtype = np.float16)
    self.terminals = np.empty(self.memory_size, dtype = np.bool) # retrieval over 标记
    # self.history_length = config.history_length #4帧，s长度
    self.dims = (config.screen_height, config.screen_width)
    self.state_dims = (config.screen_height, config.screen_width+1)
    self.batch_size = config.batch_size # 每次训练提取的经验的条数
    self.count = 0 # 循环队列中的经验条数
    self.current = 0 # 循环队列的尾指针
    # self.curr_screen = np.empty((config.screen_height, config.screen_width), dtype=np.float16)
    self.curr_state = np.empty(self.state_dims, dtype=np.float16)

    # pre-allocate prestates and poststates for minibatch
    # for state, each action is a vector with same values of size screen_height
    # prestates: screens[current-1] + actions[current-1] 左右拼接
    # poststates: screens[current] + actions[current]
    self.prestates = np.empty((self.batch_size, self.state_dims), dtype=np.float16)
    self.poststates = np.empty((self.batch_size, self.state_dims), dtype=np.float16)

  def add(self, screen, reward, action, terminal):
    """add a new record to circular queue.
    """
    assert screen.shape == self.dims
    # NB! screen is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)  # count: replay memory里的state数
    self.current = (self.current + 1) % self.memory_size #这里在控制循环队列

  def update_state(self, screen, action):
    if self.cnn_format == 'NHWC':
        screen = np.transpose(screen, (1, 2, 0))  # ?it must be NCHW,but NCHW and CPU?
    self.curr_state = map(lambda x, y: np.hstack((x, y)), screen, action)

  def getState(self, index):
    """build a experiment from screens.
    get screens
    """
    assert self.count > 0, "replay memory is empty, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # because this is a circular queue, so it is
    action = []
    for _ in range(self.screen_height):
        action += self.actions[index]
    screen = self.screens[index, ...]
    states = map(lambda x, y: np.hstack((x, y)), screen, action)
    return states

  def sample(self):
    """sample a batch of experiment randomly.
    Be careful, every 4 frames form a experience.

    Return:
        (s,a,r,s',t), @t for terminal/retrieval over.

    """
    # memory must include poststate, prestate
    assert self.count > 0
    # sample random indexes
    indexes = []
    while len(indexes) < self.batch_size:
      # find random index
      while True:
        index = random.randint(0, self.count - 1)
        # if wraps over current pointer, then get new one
        # no repetitive index
        if index in indexes or index == self.current + 1:
          continue
        break

      self.prestates[len(indexes), ...] = self.getState(index - 1)
      self.poststates[len(indexes), ...] = self.getState(index)
      indexes.append(index)

    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]

    if self.cnn_format == 'NHWC':
      return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
        rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals
    else:
      return self.prestates, actions, rewards, self.poststates, terminals

  def save(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      save_npy(array, os.path.join(self.model_dir, name))

  def load(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      array = load_npy(os.path.join(self.model_dir, name))

