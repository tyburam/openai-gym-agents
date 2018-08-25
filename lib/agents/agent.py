#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:37:57 2018

@author: mateusztybura
"""

from enum import Enum

class AgentType(Enum):
    RANDOM = 1
    GREEDY = 2
    ROUND_ROBIN = 3
    EPSILON_GREEDY = 4
    UCB = 5
    THOMPSON_BETA = 6

class Agent(object):

    def __init__(self, actions):
        self.actions = actions
        self.num_actions = len(actions)

    def act(self, obs):
        raise NotImplementedError
        
    def feedback(self, action, reward):
        pass