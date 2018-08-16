# -*- coding: utf-8 -*-

"""
========================================
Ensembling with Logistic Regression
========================================
"""
import pandas as pd
import numpy as np
import os
from scipy.special import expit, logit

almost_zero = 1e-5  # 0.00001
almost_one = 1 - almost_zero

scores = {}
positive = {}

df = pd.read_csv("input/")