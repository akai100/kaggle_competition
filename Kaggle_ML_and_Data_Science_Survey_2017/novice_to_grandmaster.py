# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import squarify
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected = True)
import plotly.graph_objs as go
import plotly.tools as tls
import base64
import io
from scipy.misc import imread
import codecs
from IPython.display import HTML
from matplotlib_venn import venn2
from subprocess import check_output

# print (check_output(["ls", "input/"]).decode("utf8"))
response = pd.read_csv("input/multipleChoiceResponses.csv", encoding = 'ISO-8859-1')
print (response.head())

# 一些基本分析
print ('The total number of respondents:', response.shape[0])
print ('Total number of Countries with respondents:', response['Country'].nunique())
print ('Country with hightest respondents:', response['Country'].value_counts().index[0], 
	'with', response['Country'].value_counts().values[0], 'respondents')
print ("Youngest respondent: ", response['Age'].min(), ' and Oldest respondent: ', response['Age'].max())