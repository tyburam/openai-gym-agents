#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 14:46:11 2018

@author: mateusztybura
"""

import numpy as np
from .agent import Agent

class ThompsonBetaAgent(Agent):
    
    def __init__(self, actions):
        self.total_counts = np.zeros(len(actions), dtype = np.longdouble)
        self.good_counts = np.ones(len(actions), dtype = np.int)
        self.bad_counts = np.ones(len(actions), dtype = np.int)
        super(ThompsonBetaAgent, self).__init__(actions)
        
    def act(self, obs):
        sample = np.random.beta( self.good_counts + 1, self.bad_counts + 1)
        return np.argmax(sample)
    
    def feedback(self, action, reward):
        self.total_counts[action] += 1
        if reward > 0:
            self.good_counts[action] += 1
            return
        self.bad_counts[action] += 1
    