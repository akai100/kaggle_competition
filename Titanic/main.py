# -*- coding: utf-8 -*-
"""
==========================================
Titanic: Mchine Learning from Disater
==========================================
本代码实现来自Kaggle kernel: Introduction to Ensembling/Stacking in Python,
链接：https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
"""
print (__doc__)

import pandas as pd
import numpy as np
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
# py.init_notebook_mode(connected = False)
import plotly.graph_objs as go
import plotly.tools as tools

from sklearn.cross_validation import KFold
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC

import xgboost as xgb

class SklearnHelper(object):
	def __init__(self, clf, seed = 0, params = None):
		params['random_state'] = seed
		self.clf = clf(**params)

	def train(self, x_train, y_train):
		self.clf.fit(x_train, y_train)

	def predict(self, x):
		return self.clf.predict(x)

	def fit(self, x, y):
		return self,clf.fit(x, y)

	def feature_importances(self, x, y):
		print (self.clf.fit(x, y).feature_importances_)
# 定义从乘客名称中提取称号的函数
def get_title(name):
	title_search = re.search(" ([A-Za-z]+)\.", name)
	# 如果称号存在，提取并返回它
	if title_search:
		return title_search.group(1)
	return ""
def main():
	# 读取训练和测试数据
	train = pd.read_csv("data/train.csv")
	test = pd.read_csv("data/test.csv")

	# 存储乘客ID
	PassengerId = test['PassengerId']

	# 打印训练数据中的前三条数据
	print (train.head(3))

	# 特征工程
	full_data = [train, test]
	# 计算每位乘客名字的长度，并作为新的一列
	train['Name_length'] = train['Name'].apply(len)
	test['Name_length'] = test['Name'].apply(len)
	# 每位乘客是否有船舱
	train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type (x) == float else 1)
	test['Has_Cabin'] = train['Cabin'].apply(lambda x : 0 if type(x) == float else 1)

	# 创建新特征项FamilySize
	for dataset in full_data:
		dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
	# 从FamilySize创建新的特征项IsAlone
	for dataset in full_data:
		dataset['IsAlone'] = 0
		dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
	# 移除Embarked列所有NULL
	for dataset in full_data:
		dataset['Embarked'] = dataset['Embarked'].fillna('S')
	# 去掉Fare列中的所有NULL，并创建新列CategoricalFare
	for dataset in full_data:
		dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
	train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
	# 创建新特征项CategoricalAge
	for dataset in full_data:
		age_avg = dataset['Age'].mean()
		age_std = dataset['Age'].std()
		age_null_count = dataset['Age'].isnull().sum()
		age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)
		dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
		dataset['Age'] = dataset['Age'].astype(int)
	train['CategoricalAge'] = pd.cut(train['Age'], 5)
	# 创建新特征项Tile
	for dataset in full_data:
		dataset['Title'] = dataset['Name'].apply(get_title)
	# 将所有非常见Title分组到一个单一分组中"rare"
	for dataset in full_data:
		dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

		dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
		dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
		dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

	for dataset in full_data:
		# 映射性别
		dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

		# 映射titles
		title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
		dataset['Title'] = dataset['Title'].map(title_mapping)
		dataset['Title'] = dataset['Title'].fillna(0)

		# 映射Embarked
		dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

		# 映射Fare
		dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
		dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] < 14.454), 'Fare'] = 1
		dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
		dataset.loc[dataset['Fare'] > 32, 'Fare'] = 3
		dataset['Fare'] = dataset['Fare'].astype(int)

		# 映射Age
		dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
		dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
		dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
		dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
		dataset.loc[dataset['Age'] > 64, 'Age'] = 4
	# 特征选择
	drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
	train = train.drop(drop_elements, axis = 1)
	train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
	test = test.drop(drop_elements, axis = 1)
	print (train.head(3))

	# 皮尔逊相关热图
	colormap = plt.cm.RdBu
	plt.figure(figsize = (14, 12))
	plt.title('Pearson Correlation of Features', y = 1.05, size = 15)
	sns.heatmap(train.astype(float).corr(), linewidths = 0.1, vmax = 1.0,
		square = True, cmap = colormap, linecolor = 'white', annot = True)

	"""
	plt.figure()
	g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch',
		u'Fare', u'Embarked', u'FamilySize', u'Title']], hue = 'Survived', 
		palette = 'seismic', size = 1.2, diag_kind = 'kde', diag_kws = dict(shade = True),
		plot_kws = dict(s = 10))
	g.set(xticklabels = [])
	"""
	plt.show()

	# 一些有用的参数
	ntrain = train.shape[0]
	ntest = test.shape[0]
	SEED = 0
	NFOLDS = 5
	# 使用k折交叉验证将训练数据分为5份，每一份都包含训练集和测试集
	kf = KFold(ntrain, n_folds = NFOLDS, random_state = SEED)
	#
	def get_oof(clf, x_train, y_train, x_test):
		oof_train = np.zeros((ntrain, ))
		oof_test = np.zeros((ntest, ))
		oof_test_skf = np.empty((NFOLDS, ntest))  # 保存每折中对测试数据的预测

		for i, (train_index, test_index) in enumerate(kf):
			x_tr = x_train[train_index]   # 训练X
			y_tr = y_train[train_index]   # 训练y
			x_te = x_train[test_index]    # 验证集X

			clf.train(x_tr, y_tr)         # 训练模型

			oof_train[test_index] = clf.predict(x_te) # 保存验证集X的预测
			oof_test_skf[i, :] = clf.predict(x_test)  # 保存当前对测试集的预测

		oof_test[:] = oof_test_skf.mean(axis = 0)
		return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

	#  生成第一层模型
	# 随机森林参数
	rf_params = {
	    'n_jobs': -1,
	    'n_estimators': 500,
	    'warm_start': True,
	    'max_depth': 6,
	    'min_samples_leaf': 2,
	    'max_features': 'sqrt',
	    'verbose': 0
	}
	# Extra Trees 参数
	et_params = {
	    'n_jobs': -1,
	    'n_estimators': 500,
	    'max_depth': 8,
	    'min_samples_leaf': 2,
	    'verbose': 0
	}
	# AdaBoost 参数
	ada_params = {
	    'n_estimators': 500,
	    'learning_rate': 0.75,
	}
	# Gradient Boosting 参数
	gb_params = {
	    'n_estimators': 500,
	    'max_depth': 5,
	    'min_samples_leaf': 2,
	    'verbose': 0
	}
	# SVM分类器参数
	svc_params = {
	    'kernel': 'linear',
	    'C': 0.025
	}
	rf = SklearnHelper(clf = RandomForestClassifier, seed = SEED, params = rf_params)
	et = SklearnHelper(clf = ExtraTreesClassifier, seed = SEED, params = et_params)
	ada = SklearnHelper(clf = AdaBoostClassifier, seed = SEED, params = ada_params)
	gb = SklearnHelper(clf = GradientBoostingClassifier, seed = SEED, params = gb_params)
	svc = SklearnHelper(clf = SVC, seed = SEED, params = svc_params)

	y_train = train['Survived'].ravel()
	train = train.drop(['Survived'], axis = 1)
	x_train = train.values
	x_test = test.values

	et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
	rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)
	ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)
	gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)
	svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)

	rf_feature = rf.feature_importances(x_train, y_train)
	et_feature = et.feature_importances(x_train, y_train)
	ada_feature = ada.feature_importances(x_train, y_train)
	gb_feature = gb.feature_importances(x_train, y_train)
	
	rf_features = [0.10474135, 0.21837029, 0.04432652, 0.02249159, 0.05432591, 0.02854371, 0.07570305, 0.01088129, 0.24247496, 0.13685733, 0.06128402]
	et_features = [0.12165657, 0.37098307, 0.03129623, 0.01591611, 0.05525811, 0.028157, 0.04589793, 0.02030357, 0.17289562, 0.04853517, 0.08910063]
	ada_features = [0.028, 0.008, 0.012, 0.05866667, 0.032, 0.008, 0.04666667, 0., 0.05733333, 0.73866667, 0.01066667]
	gb_features = [0.06796144, 0.03889349, 0.07237845, 0.02628645, 0.11194395, 0.04778854, 0.05965792, 0.02774745, 0.07462718, 0.4593142, 0.01340093]

	cols = train.columns.values

	feature_dataframe = pd.DataFrame({'features': cols,
		'Random Forest feature importances': rf_features,
		'Extra Trees feature importances': et_features,
		'AdaBoost feature importances': ada_features,
		'Gradient Boost feature importances': gb_features
	})

	trace = go.Scatter(
		y = feature_dataframe['Random Forest feature importances'].values,
		x = feature_dataframe['features'].values,
		mode = 'markers',
		marker = dict(
			sizemode = 'diameter',
			sizeref = 1,
			size = 25,
			color = feature_dataframe['Random Forest feature importances'].values,
			colorscale = 'Portland',
			showscale = True),
		text = feature_dataframe['features'].values
	)
	data = [trace]

	layout = go.Layout(
		autosize = True,
		title = 'Random Forest Feature Importance',
		hovermode = 'closest',
		yaxis = dict(
			title = 'Feature Importance',
			ticklen = 5,
			gridwidth = 2),
		showlegend = False
	)
	fig = go.Figure(data = data, layout = layout)
	py.iplot(fig, filename='scatter2010')

	trace = go.Scatter(
		y = feature_dataframe['Extra Trees feature importances'].values,
		x = feature_dataframe['features'].values,
		mode = 'markers',
		marker = dict(
			sizemode = 'diameter',
			sizeref = 1,
			size = 25,
			color = feature_dataframe['Extra Trees feature importances'].values,
			colorscale = 'Portland',
			showscale = True
		),
		text = feature_dataframe['features'].values
	)
	data = [trace]
	layout = go.Layout(
		autosize = True,
		title = 'Extra Trees Feature Importance',
		hovermode = 'closest',
		yaxis = dict(
			title = 'Feature Importance',
			ticklen = 5,
			gridwidth = 2
		),
		showlegend = False
	)
	fig = go.Figure(data = data, layout = layout)
	py.iplot(fig, filename = 'scatter2010')

	trace = go.Scatter(
		y = feature_dataframe['AdaBoost feature importances'].values,
		x = feature_dataframe['features'].values,
		mode = 'markers',
		marker = dict(
			sizemode = 'diameter',
			sizeref = 1,
			size = 25,
			color = feature_dataframe['AdaBoost feature importances'].values,
			colorscale = 'Portland',
			showscale = True
		),
		text = feature_dataframe['features'].values
	)
	data = [trace]

	layout = go.Layout(
		autosize = True,
		title = 'AdaBoost Feature Importance',
		hovermode = 'closest',
		yaxis = dict(
			title = 'Feature Importance',
			ticklen = 5,
			gridwidth = 2
		),
		showlegend = False
	)
	fig = go.Figure(data = data, layout = layout)
	py.iplot(fig, filename = 'scatter2010')

	trace = go.Scatter(
		y = feature_dataframe['Gradient Boost feature importances'].values,
		x = feature_dataframe['features'].values,
		mode = 'markers',
		marker = dict(
			sizemode = 'diameter',
			sizeref = 1,
			size = 25,
			color = feature_dataframe['Extra Trees feature importances'].values,
			colorscale = 'Portland',
			showscale = True
		),
		text = feature_dataframe['features'].values
	)
	data = [trace]

	layout = go.Layout(
		autosize = True,
		title = 'Gradient Boosting Feature Importance',
		hovermode = 'closest',
		yaxis = dict(
			title = 'Feature Importance',
			ticklen = 5,
			gridwidth = 2
		),
		showlegend = False
	)
	fig = go.Figure(data = data, layout = layout)
	py.iplot(fig, filename = 'scatter2010')

	feature_dataframe['mean'] = feature_dataframe.mean(axis = 1)
	print (feature_dataframe.head(3))

	y = feature_dataframe['mean'].values
	x = feature_dataframe['features'].values
	data = [go.Bar(
		x = x,
		y = y,
		width = 0.5,
		marker = dict(
			color = feature_dataframe['mean'].values,
			colorscale = 'Portland',
			showscale = True,
			reversescale = False
		),
		opacity = 0.6
	)]
	layout = go.Layout(
		autosize = True,
		title = 'Barplots of Mean Feature Importance',
		hovermode = 'closest',
		yaxis = dict(
			title = 'Feature Importance',
			ticklen = 5,
			gridwidth = 2
		),
		showlegend = False
	)
	fig = go.Figure(data = data, layout = layout)
	py.iplot(fig, filename = 'bar-direct-labels')

	#
	base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
		'ExtraTrees': et_oof_train.ravel(),
		'AdaBoost': ada_oof_train.ravel(),
		'GradientBoost': gb_oof_train.ravel()
	})
	print (base_predictions_train.head())

	data = [
	    go.Heatmap(
	    	z = base_predictions_train.astype(float).corr().values,
	    	x = base_predictions_train.columns.values,
	    	y = base_predictions_train.columns.values,
	    	colorscale = 'Viridis',
	    	showscale = True,
	    	reversescale = True
	    )
	]
	py.iplot(data, filename = 'labelled-heatmap')

	x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, 
		gb_oof_train, svc_oof_train), axis = 1)
	x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, 
		svc_oof_test), axis = 1)

	gbm = xgb.XGBClassifier(
		n_estimators = 2000,
		max_depth = 4,
		min_child_weight = 2,
		gamma = 0.9,
		subsample = 0.8,
		colsample_bytree = 0.8,
		objective = 'binary:logistic',
		nthread = -1,
		scale_pos_weight = 1).fit(x_train, y_train)
	predictions = gbm.predict(x_test)

	StackingSubmission = pd.DataFrame({'PassengerId': PassengerId,
		'Survived': predictions})
	StackingSubmission.to_csv("StackingSubmission.csv", index = False)


if __name__ == "__main__":
	main()