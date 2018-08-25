#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 13:46:55 2018

@author: mateusztybura
"""

import numpy as np
import gym
from .agents.agent import AgentType
from .agents.random_agent import RandomAgent
from .agents.greedy_agent import GreedyAgent
from .agents.round_robin_agent import RoundRobinAgent
from .agents.epsilon_greedy_agent import EpsilonGreedyAgent
from .agents.ucb_agent import UcbAgent
from .agents.thompson_beta_agent import ThompsonBetaAgent

def discrete_to_list(action_space):
    return [x for x in range(action_space.n)]


class GymSimulator:
    
    def __init__(self, env_name, agent_type=1, agent_params = {}, episodes=20, 
                 steps=200):
        self.env = gym.make(env_name)
        
        actions = discrete_to_list(self.env.action_space)
        if agent_type == AgentType.RANDOM:
            self.agent = RandomAgent(actions)
        elif agent_type == AgentType.GREEDY:
            self.agent = GreedyAgent(actions)
        elif agent_type == AgentType.ROUND_ROBIN:
            self.agent = RoundRobinAgent(actions)
        elif agent_type == AgentType.EPSILON_GREEDY:
            self.agent = EpsilonGreedyAgent(actions, agent_params['epsilon'])
        elif agent_type == AgentType.UCB:
            self.agent = UcbAgent(actions, agent_params['epsilon'])
        elif agent_type == AgentType.THOMPSON_BETA:
            self.agent = ThompsonBetaAgent(actions)
            
        self.episodes = episodes
        self.steps = steps
        self.finished_after = np.zeros(episodes, dtype = np.int)
        
    def run(self):
        for i_episode in range(self.episodes):
            observation = self.env.reset()
            for t in range(self.steps):
                self.env.render()
                action = self.agent.act(observation)
                observation, reward, done, info = self.env.step(action)
                self.agent.feedback(action, reward)
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    self.finished_after[i_episode] = t+1
                    break
        self.env.close()
        