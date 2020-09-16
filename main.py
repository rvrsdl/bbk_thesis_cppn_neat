#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The entry point for running the system

Created on Tue Sep 15 19:43:24 2020

@author: riversdale
"""

import yaml

with open('config.yaml','r') as f:
    config = yaml.load(f)
