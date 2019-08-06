# -*- coding: utf-8 -*-

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import re

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import Imputer, Normalizer, scale
from sklearn.model_selection import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

import xgboost as xgb
import gzip
import lightgbm as lgbm


def read_csv_gz(path, *params):
    with gzip.open(path, 'rb') as f:
        return pd.read_csv(f, *params)

train_path = r'train.csv.gz'
test_path = r'test.csv.gz'
seed = 0
folds = 10

train_df = read_csv_gz(train_path)
test_df = read_csv_gz(train_path)

train_y = train_df['Survived']
train_X = train_df.drop(['PassengerId', 'Survived'])

test_id = test_df['PassengerId']
test_X = test_df.drop(['PassengerId', 'Survived'])

all_X = pd.concat([train_X, test_X], sort=False)

cat_cols = ['Pclass', 'Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket']
drop_cols = ['Name']

# Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
#       dtype='object')

# Pclass: no na, 1,2,3

# Name:
all_X['Title'] = all_X['Name'].str.extract(r' (\w+)\.')
title_set = set(('Mr', 'Miss', 'Mrs', 'Master'))
all_X['Title'] = all_X['Title'].map(lambda x: x if x in title_set else 'Other')

# Age
all_X['Age'].fillna(-1, inplace=True)
all_X['Age'] = all_X['Age'].map(lambda x: x//5 if x//5 < 13 else 13)

# SibSp
all_X['SibSp'] = all_X['SibSp'].map(lambda x: x if x < 5 else 5)

# Parch
all_X['Parch'] = all_X['Parch'].map(lambda x: x if x < 3 else 3)

# Ticket
tickets = all_X["Ticket"].value_counts()
tickets_set = set(tickets[tickets>1].index)
all_X['Ticket'] = all_X['Ticket'].map(lambda x: x if x in tickets_set else 'Others')

# Fare
all_X['Fare'].fillna(-1, inplace=True)
all_X['Fare'] = all_X['Fare'].map(lambda  x: x//1 if x <= 30 else x//5 + 30 if x <= 100 else 100)

# Cabin
all_X['Cabin'].fillna('Others', inplace=True)
cabins = all_X["Cabin"].value_counts()
cabins_set = set(cabins[cabins>1].index)
all_X['Cabin'] = all_X['Cabin'].map(lambda x: x if x in cabins_set else 'Others')

# Embarked: na 2
all_X['Embarked'].fillna('Others', inplace=True)

# kfold = KFold(n_splits=folds, random_state=seed)
#
# df = pd.read_csv(path)
# test_df = pd.read_csv(test_path)



# def preprocess(df):
#
#     df['Age'] = df['Age'].fillna(df['Age'].mean())
#     df['Fare'].fillna(df['Fare'].mean(), inplace=True)
#     df['Pclass'] = df['Pclass'].astype('object')
#     #    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
#     #    title_map = {
#     #                    "Capt":       "Officer",
#     #                    "Col":        "Officer",
#     #                    "Major":      "Officer",
#     #                    "Jonkheer":   "Royalty",
#     #                    "Don":        "Royalty",
#     #                    "Sir" :       "Royalty",
#     #                    "Dr":         "Officer",
#     #                    "Rev":        "Officer",
#     #                    "the Countess":"Royalty",
#     #                    "Dona":       "Royalty",
#     #                    "Mme":        "Mrs",
#     #                    "Mlle":       "Miss",
#     #                    "Ms":         "Mrs",
#     #                    "Mr" :        "Mr",
#     #                    "Mrs" :       "Mrs",
#     #                    "Miss" :      "Miss",
#     #                    "Master" :    "Master",
#     #                    "Lady" :      "Royalty"
#     #
#     #                    }
#     #    df['Title'] = df['Title'].map(title_map)
#     #    df['Cabin'].fillna('U', inplace=True)
#     #    df['Cabin'] = df['Cabin'].map(lambda c: c[0])
#     dropped_cols = ['PassengerId', 'Ticket', 'Name', 'Cabin']
#     #    df['FamilySize'] = df ['SibSp'] + df['Parch'] + 1
#
#     df = df.drop(dropped_cols, axis=1)
#     df = pd.get_dummies(df, drop_first=True)
#     #    df['SibSp'] = df['SibSp'].map(lambda x: 1 if 1<=x<=2 else 0)
#     #    df['Parch'] = df['Parch'].map(lambda x: 1 if 1<=x<=3 else 0)
#
#     return df

def split(df):
    x = df.drop(['Survived'], axis=1)
    y = df['Survived']
    return x, y

# df = preprocess(df)
# X, Y = split(df)
# model1 = BaggingClassifier(n_jobs=-1)
# model2 = AdaBoostClassifier(random_state=seed, n_estimators=100)
# model3 = GradientBoostingClassifier(random_state=seed, n_estimators=100)
# model4 = SVC(gamma='auto')
# model5 = KNeighborsClassifier(n_jobs=-1)
# model6 = GaussianNB()
# model7 = Perceptron(max_iter=1000, tol=1e-3,n_jobs=-1)
# model8 = SGDClassifier(max_iter=1000, tol=1e-3,n_jobs=-1)
# model9 = LogisticRegression(solver='liblinear')
# model10 = xgb.XGBClassifier(objective="binary:logistic", random_state=seed,n_jobs=-1)
# models = []
# for model in (model1, model2, model3, model4, model5, model6, model7, model8, model9, model10):
#     name = re.search(r'\.([^.\']*)\'', str(model.__class__))[1]
#     models.append((name, model))
#
# model = VotingClassifier(estimators=models)

def weights():
    nx = pd.DataFrame()

    for name, model in models:
        model.fit(X, Y)
        yh = model.predict(X)
        nx[name] = yh
    return nx


def predict():
    test_df = pd.read_csv(test_path)
    id_list = test_df['PassengerId']
    test_df = preprocess(test_df)
    model.fit(X, Y)
    pred = model.predict(test_df)
    pred = pd.concat({'PassengerId': id_list, 'Survived':pd.Series(pred)}, axis=1)
    pred.to_csv('result.csv', index=False)
    return pred

def predict2():
    test_df = pd.read_csv(test_path)
    id_list = test_df['PassengerId']
    test_df = preprocess(test_df)
    nx0 = pd.DataFrame()
    nx1 = pd.DataFrame()
    for name, model in models:
        model.fit(X, Y)
        yh0 = model.predict(X)
        yh1 = model.predict(test_df)
        nx0[name] = yh0
        nx1[name] = yh1
    model.fit(nx0, Y)
    pred = model.predict(nx1)
    pred = pd.concat({'PassengerId': id_list, 'Survived':pd.Series(pred)}, axis=1)
    pred.to_csv('result.csv', index=False)
    return pred

def test_boosting():
    for model in (model4, model5, model6, model7, model8, model9):
        m = AdaBoostClassifier(base_estimator=model, random_state=seed, n_estimators=100)
        result = cross_val_score(m, X, Y, cv=kfold)
        print("%s mean %.4f std %.4f" % (str(m.__class__), result.mean(), result.std()))

def test_bagging():
    for model in (model4, model5, model6, model7, model8, model9, model10):
        m = BaggingClassifier(base_estimator=model)
        result = cross_val_score(m, X, Y, cv=kfold)
        print("%s mean %.4f std %.4f" % (str(m.__class__), result.mean(), result.std()))

def test_xgb():
    model = xgb.XGBClassifier(objective="binary:logistic", random_state=seed)
    result = cross_val_score(model, X, Y, cv=kfold)
    print("%s mean %.4f std %.4f" % ('XGB', result.mean(), result.std()))

def test_model(model):
    result = cross_val_score(model, X, Y, cv=kfold)
    print("%s mean %.4f std %.4f" % ('XGB', result.mean(), result.std()))

def voting():
    result = cross_val_score(model, X, Y, cv=kfold)
    print("%s mean %.4f std %.4f" % ('VotingClassifier', result.mean(), result.std()))

def test_models(X, Y):
    for name, model in models:
        result = cross_val_score(model, X, Y, cv=kfold)
        print("%s mean %.4f std %.4f" % (name, result.mean(), result.std()))

# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']


def check_feature_p(feature):
    if feature == 'Survived':
        return
    print(df[[feature, "Survived"]].groupby([feature]).mean()-342/891)

def check_feature_corr(feature):
    print(df[[feature, 'Survived']].corr())

def pic():
    g = sns.FacetGrid(df, col='Survived')
    g.map(plt.hist, 'Age', bins=20)

    grid = sns.FacetGrid(df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend();

    grid = sns.FacetGrid(df, row='Embarked', height=2.2, aspect=1.6)
    grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    grid.add_legend()

    grid = sns.FacetGrid(df, row='Embarked', col='Survived', height=2.2, aspect=1.6)
    grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
    grid.add_legend()

