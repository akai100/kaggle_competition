# -*- coding: utf8 -*-

import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style ="white", color_codes = True)

iris = pd.read_csv("input/Iris.csv")

print (iris.head())

print (iris['Species'].value_counts())

iris.plot(kind = "sctter", x = "SepalLengthCm", y = "sepalWidthCm")

sns.jointplot(x = "SepalLengthCm", y = "SepalWidthjj")