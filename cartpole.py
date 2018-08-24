#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:22:22 2018

@author: mateusztybura
"""

import gym
from agents.random_agent import RandomAgent

env = gym.make('CartPole-v0')
agent = RandomAgent([x for x in range(env.action_space.n)])

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()