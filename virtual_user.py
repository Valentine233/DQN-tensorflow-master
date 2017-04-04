import random
import cv2
import tensorflow as tf
import numpy as np


class VirtualUser(object):
    def __init__(self, config):
        self.nb_img = config.nb_img
        self.rel_max = config.rel_max
        self.top_nb = config.top_nb
        self.rad_nb = config.rad_nb
        self.LOW = config.LOW
        self.MIDDLE = config.MIDDLE
        self.HIGH = config.HIGH
        self.action_size = config.action_size
        screen_width, screen_height, self.random_start = \
            config.screen_width, config.screen_height, config.random_start

        self.display = config.display
        self.dims = (screen_width, screen_height)
        self.screen = None
        self.relevancy = None
        self.img_use = dict()  # displayed images
        self.reward = 0
        self.terminal = False
        # store relevancy [key=name, val=relevancy]
        self.rel_store = dict()
        for i in range(self.nb_img):
            self.rel_store["img" + str(i + 1)] = random.randint(0, self.rel_max)
        # store images [key=name, val=image]
        self.img_store = dict()
        """image pre-processing
                RGB->gray, scale in [0,1], modify dimension
                   screen matrix, may be a numpy object.
                """
        for i in range(self.nb_img):
            self.img_store["img" + str(i + 1)] = cv2.resize(cv2.cvtColor(random.randint(0, 255, size=self.dims), 
                                                                         cv2.COLOR_RGB2GRAY) / 255., self.dims)
        self.target_screen = self.img_store["img" + str(random.randint(1, self.nb_img))]

    def new_random_start(self):
        # used image [key=image, val=relevancy]
        self.img_use.clear()
        self.terminal = False
        for i in range(self.top_nb + self.rad_nb):
            rad = random.randint(1, self.nb_img)
            while self.img_store["img" + str(rad)] in self.img_use:
                rad = random.randint(1, self.nb_img)
            self.img_use[self.img_store["img" + str(rad)]] = self.rel_store["img" + str(rad)]
        self.img_pick()  # img_chosen and relevancy_chosen
        for _ in xrange(random.randint(0, self.random_start - 1)):
            action = random.randint(0, 3)
            self.sort(action)
            self.img_display()
            self.img_pick()
            if self.screen == self.target_screen:
                self.terminal = True
                break                
        return self.screen, 0, action, self.terminal  # screen, reward, action, terminal

    def state_transfer(self, action):
        self.sort(action)
        self.img_display()
        self.img_pick()
        self.reward_pick()
        if self.screen == self.target_screen:
            self.terminal = True
        else:
            self.terminal = False
        return self.screen, self.reward, self.terminal  # screen + action = poststate

    def img_display(self):
        self.img_use.clear()
        for i in range(self.top_nb):
            (k, v) = sorted(self.rel_store.items(), key=lambda d: d[1])[i]
            self.img_use[self.img_store[k]] = self.rel_store[k]
        for j in range(self.rad_nb):
            rad = random.randint(1, self.nb_img)
            while self.img_store["img"+str(rad)] in self.img_use:
                rad = random.randint(1, self.nb_img)
            self.img_use[self.img_store["img"+str(rad)]] = self.rel_store["img"+str(rad)]

    def img_pick(self):
        img_list = self.img_use.keys()
        probas = self.img_use.values()
        x = random.uniform(0, sum(probas))
        culmulated_proba = 0.0
        for item, item_proba in zip(img_list, probas):
            culmulated_proba += item_proba
            if x < culmulated_proba:
                break
        self.screen = item
        self.rel_max = item_proba

    def reward_pick(self):
        x = random.uniform(0, 1)
        img_proba = 1. - self.relevancy/self.rel_max*1.
        pro_l = 1./4
        pro_m = 1./4
        pro_h = 1./4
        if img_proba <= 1./3:
            pro_l = 1./2
        elif img_proba <= 2./3:
            pro_m = 1./2
        else:
            pro_h = 1./2
        if x <= pro_l:
            self.reward = self.LOW
        elif x <= pro_m:
            self.reward = self.MIDDLE
        else:
            self.reward = self.HIGH

    # Interface
    def sort(self, action):

     @property
     def action_size(self):
        """recent action size
        """
        return self.action_size







