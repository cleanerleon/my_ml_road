# -*- coding: utf-8 -*-

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import re
import concurrent.futures
import platform
import pickle
import os
import traceback
import itertools
import multiprocessing

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
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from sklearn.metrics import accuracy_score

import xgboost as xgb
import gzip
import lightgbm as lgbm

def load_pickle(path):
    if platform.system() == 'Linux':
        path = path.replace('\\', '/')
    else:
        path = path.replace('/', '\\')

    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def save_pickle(obj, path):
    if platform.system() == 'Linux':
        path = path.replace('\\', '/')
    else:
        path = path.replace('/', '\\')

    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def read_csv_gz(path, *params):
    with gzip.open(path, 'rb') as f:
        return pd.read_csv(f, *params)


train_path = r'train.csv.gz'
test_path = r'test.csv.gz'
seed = 0
folds = 10

def process(df):
    drop_cols = ['Name']

    # Pclass: no na, 1,2,3

    # Name:
    df['Title'] = df['Name'].str.extract(r' (\w+)\.')
    title_set = set(('Mr', 'Miss', 'Mrs', 'Master'))
    df['Title'] = df['Title'].map(lambda x: x if x in title_set else 'Other')

    # Age
    df['Age'].fillna(-1, inplace=True)
    df['Age'] = df['Age'].map(lambda x: x // 5 if x // 5 < 13 else 13)

    # SibSp
    df['SibSp'] = df['SibSp'].map(lambda x: x if x < 5 else 5)

    # Parch
    df['Parch'] = df['Parch'].map(lambda x: x if x < 3 else 3)

    # Ticket
    tickets = df["Ticket"].value_counts()
    tickets_set = set(tickets[tickets>1].index)
    df['Ticket'] = df['Ticket'].map(lambda x: x if x in tickets_set else 'Others')

    # Fare
    df['Fare'].fillna(-1, inplace=True)
    df['Fare'] = df['Fare'].map(lambda x: x // 1 if x <= 30 else x // 5 + 30 if x <= 100 else 100)

    # Cabin
    df['Cabin'].fillna('Others', inplace=True)
    cabins = df["Cabin"].value_counts()
    cabins_set = set(cabins[cabins>1].index)
    df['Cabin'] = df['Cabin'].map(lambda x: x if x in cabins_set else 'Others')

    # Embarked: na 2
    df['Embarked'].fillna('Others', inplace=True)
    df.drop(drop_cols, inplace=True, axis=1)

    # add count columns
    cnt_cols = []
    cat_cols = list(df.columns)
    for col in df.columns:
        cnt_col = col + ' cnt'
        cnt_cols.append(cnt_col)
        df[cnt_col] = df[col].map(df[col].value_counts())

    # df['count'] = 0
    # for col in cat_cols:
    #     cnt_df = df.groupby([col, 'Survived']).agg({'count': 'count'})
    #     cnt_map = {}
    #     for key in cnt_df.index.levels[0]:
    #         item = [cnt_df.loc[(key, key2), 'count'] if (key, key2) in cnt_df else 0 for key2 in (0, 1, -1) ]
    #         cnt_map[key] = item
    #     idx = 0
    #     for i in (0, 1, -1):
    #         col_name = col + ' ' + str(i) + ' cnt'
    #         df[col_name] = df[col].map(lambda x: cnt_map[x][idx])
    #         idx += 1
    #         cnt_cols.append(col_name)
    # print(cat_cols)
    # print(cnt_cols)
    ohe = OneHotEncoder()
    X1 = ohe.fit_transform(df[cat_cols])
    X2 = sparse.csr_matrix(df[cnt_cols])
    df = sparse.hstack([X1, X2])
    return df.tocsr()

def lgb_eval(path, lgb_param):
    print(lgb_param)
    X, y = load_pickle(path)
    clf = lgbm.LGBMClassifier(random_state=0, n_jobs=-1, objective='binary', tree_learner='data', **lgb_param)
    result = cross_val_score(clf, X, y, n_jobs=-1, scoring='accuracy', cv=5)
    print(result, lgb_param)
    return result

def lgb_test(train_X, train_y, test_X, test_y):
    clf = lgbm.LGBMClassifier(random_state=0, n_jobs=-1, objective='binary', tree_learner='data')
    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)
    print('accuracy', accuracy_score(test_y, pred_y))

def lgb_grid(X, y):
    print('lgb grid search')
    path = r'sav\lgb-grid.pck'
    result_map = load_pickle(path)
    if result_map is None:
        result_map = {}
    params = {
        'n_estimators': [100],
        'num_leaves': [31],
        'colsample_bytree': [.6, .8, 1],
        'reg_alpha': [0, .01, .1, 1],
        'reg_lambda': [0, .01, .1, 1],
        'min_child_weight': [.001, .01, .1, 1],
        'min_child_samples': [1, 10, 20, 100],
        'subsample_freq': [1],
        'subsample': [.6, .8, 1],
    }
    param_grid = itertools.product(*(params[key] for key in params.keys()))
    mp_param_list = []
    best_value, best_params = 0, None
    for param in param_grid:
        lgb_param = {k: v for k, v in zip(params.keys(), param)}
        key = str(lgb_param)
        if key in result_map:
            value = result_map[key]
            if value.mean() > best_value:
                best_value = value.mean()
                best_params = key
            continue
        mp_param_list.append(lgb_param)
    mp_param_list.reverse()

    dat = X, y
    dat_path = r'sav/dat.pck'
    save_pickle(dat, dat_path)
    print('best value %s, params %s' % (best_value, best_params))
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
        worker_map = {executor.submit(lgb_eval, dat_path, lgb_param): str(lgb_param) for lgb_param in mp_param_list}
        for future in concurrent.futures.as_completed(worker_map):
            key = worker_map[future]
            try:
                result = future.result()
            except Exception as e:
                print(e)
                traceback.print_exc()
            else:
                result_map[key] = result
                save_pickle(result_map, path)
                print('accuracy ', result.max(), result.mean())

    result = [(v.mean(), v, k) for k, v in result_map.items()]
    result.sort(key=lambda x: x[0], reverse=True)
    for item in result[:10]:
        print(item)

# without cnt columns
# ("{'n_estimators': 100, 'num_leaves': 31}", array([0.7877095 , 0.81005587, 0.87640449, 0.80898876, 0.82485876]), 0.821603475723521)
# with cnt columns
# ("{'n_estimators': 100, 'num_leaves': 31}", array([0.82681564, 0.7877095 , 0.84269663, 0.8258427 , 0.84745763]), 0.8261044185252292)

def train():
    df = read_csv_gz(train_path)
    df_X = df.drop(['PassengerId', 'Survived'], axis=1)
    df_y = df['Survived']
    df_X = process(df_X)
    lgb_grid(df_X, df_y)

def test():
    params = {'n_estimators': 4000, 'num_leaves': 127, 'colsample_bytree': 0.8, 'reg_alpha': 0.01, 'reg_lambda': 0.1, 'min_child_weight': 1, 'min_child_samples': 1, 'subsample_freq': 1, 'subsample': 1}
    # params = {'n_estimators': 100, 'num_leaves': 31, 'colsample_bytree': 0.6, 'reg_alpha': 0, 'reg_lambda': 0.1, 'min_child_weight': 1, 'min_child_samples': 1, 'subsample_freq': 1, 'subsample': 0.8}
    train_df = read_csv_gz(train_path)
    test_df = read_csv_gz(test_path)
    train_y = train_df['Survived']
    test_id = test_df['PassengerId']
    all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
    all_X = all_df.drop(['PassengerId', 'Survived'], axis=1)
    all_X = process(all_X)
    train_len = len(train_df)
    train_X = all_X[:train_len]
    test_X = all_X[train_len:]
    clf = lgbm.LGBMClassifier(random_state=0, n_jobs=-1, objective='binary', tree_learner='data', **params)
    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)
    pred_df = pd.concat({'PassengerId': test_id, 'Survived':pd.Series(pred_y)}, axis=1)
    pred_df.to_csv('result.csv', index=False)


if __name__ == '__main__':
    __spec__ = 'Titanic'
    print('start')
    test()



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

