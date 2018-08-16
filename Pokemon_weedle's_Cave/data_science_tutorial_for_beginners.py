# -*- coding: utf-8 -*-
"""
===================================
Data Science Trtorial for Beginners
===================================
原文链接：https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
	data = pd.read_csv("data/pokemon.csv")
	print (data.info())
	f, ax = plt.subplots(figsize = (18, 18))
	sns.heatmap(data.corr(), annot = True, linewidths = 5, fmt = '.1f', ax = ax)
	plt.show()

	print (data.head(10))

	data.Speed.plot(kind = 'line', color = 'g', label = 'Speed', linewidth = 1,
		alpha = 0.5, grid = True, linestyle = ":")
	data.Defense.plot(color = 'r', label = 'Defense', linewidth = 1, alpha = 0.5,
		grid = True, linestyle = '-.')
	plt.legend(loc = 'upper right')
	plt.xlabel ('x axis')
	plt.ylabel(' y axis')
	plt.title('Line Plot')
	plt.show()

	# Scatter Plot
	# x = attack, y = defense
	data.plot(kind = 'scatter', x = 'Attack', y = 'Defense', alpha = 0.5, color = 'red')
	plt.xlabel('Attack')
	plt.ylabel('Defense')
	plt.title('Attack Defense Scatter Plot')
	plt.show()

	data.Speed.plot(kind = 'hist', bins = 50, figsize = (15, 15))
	plt.show()

if __name__ == "__main__":
	main()