#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:22:22 2018

@author: mateusztybura
"""

from lib.gym_simulator import GymSimulator
from lib.agents.agent import AgentType

sim = GymSimulator('CartPole-v0', AgentType.THOMPSON_BETA, {'epsilon':0.7})
sim.run()