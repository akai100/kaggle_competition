#  -*- coding: utf-8 -*-
"""
===================================================
Kaggle 房价预测比赛
===================================================
"""
print (__doc__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
	#  读取训练和测试数据
	train_data = pd.read_csv("data/train.csv")  #  训练集
	y_label = 'SalePrice'
	X_labels = []
	for label in train_data.columns:
		if label != y_label:
			X_labels.append(label)
	train_y = train_data[y_label]   #  训练目标集
	train_X = train_data[X_labels]  #  训练特征集
	n_samples = train_X.shape[0]    #  实例集数量
	m_features = train_X.shape[1]   #  特征数量
	print ("训练集大小：", n_samples)
	print ("训练特征数量：", m_features)
	print ("预测目标类别：", train_y.dtypes)
	#
	print (train_y.describe())
	#  分析特征项缺失信息
	miss_df = train_X.isnull()
	miss_sum_df = miss_df.sum(axis = 0) / n_samples
	miss_sum_df = miss_sum_df.reset_index()
	miss_sum_df.columns = ['label', 'miss_percent']
	fig = plt.figure()
	plt.bar(miss_sum_df.label, miss_sum_df.miss_percent)
	plt.show()


if __name__ == "__main__":
	main()