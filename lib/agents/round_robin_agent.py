#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 13:31:26 2018

@author: mateusztybura
"""

from .agent import Agent

class RoundRobinAgent(Agent):
    
    def __init__(self, actions):
        self.previous_action = None
        super(RoundRobinAgent, self).__init__(actions)
    
    def act(self, obs):
        if self.previous_action == None:
            self.previous_action = 0
            return 0
        
        current_action = self.previous_action + 1
        if current_action >= self.num_actions:
            current_action = 0
        self.previous_action = current_action
        return current_action