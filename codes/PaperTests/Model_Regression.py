# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: Model_Regression.py
@time: 2023/3/12 19:45
"""
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, accuracy_score, roc_curve, roc_auc_score, recall_score, precision_score, classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import pickle

pic_path = r"D:\my_projects_V1\my_projects\论文图片\5"

def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))

def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))

def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))

def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))

def my_confusion_matrix(y_true, y_predict):
    return np.array([[TN(y_true, y_predict)], [FP(y_true, y_predict)],
                     [FN(y_true, y_predict)], [TP(y_true, y_predict)]])

def cal_sn(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    return tp / (tp + fn)

def cal_sp(y_true, y_predict):
    tn = TN(y_true, y_predict)
    fp = FP(y_true, y_predict)
    return tn / (tn + fp)


def my_logistic(x_train, y_train, x_test, y_test, pic_save=False, model_save=False, model_save_name=None):
    print("------------------logistic regression-------------------------")
    clf = LogisticRegression(penalty="l2")
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print("logistic accuracy = ", accuracy)
    f1 = f1_score(y_test, y_predict, average="binary")
    print("logistic f1 = ", f1)
    sn = cal_sn(y_test, y_predict)
    sp = cal_sp(y_test, y_predict)
    print("sn = %s, sp = %s" % (sn, sp))
    print(classification_report(y_test, y_predict))
    cnf_matrix = confusion_matrix(y_test, y_predict)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g")
    plt.title("混淆矩阵", y=1.01)
    plt.ylabel("真实值")
    plt.xlabel("预测值")
    plt.savefig(os.path.join(pic_path, "logistic_matric"), dpi=300)
    plt.show()

    # 绘制ROC曲线
    probs = clf.predict_proba(x_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([-0.1, 1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

    if model_save:
        if model_save_name is not None:
            model_name = model_save_name + ".pickle"
        else:
            model_name = "logistic.pickle"

        with open(model_name, "wb") as f:
            pickle.dump(clf, f)


def my_knn(x_train, y_train, x_test, y_test, pic_save=False, model_save=False, model_save_name=None):
    print("---------------------KNN--------------------")
    knn = KNeighborsClassifier(weights="distance", algorithm="kd_tree", leaf_size=10)
    param_grid = {"n_neighbors": np.arange(2, 18)}
    grid_knn = GridSearchCV(knn, param_grid, scoring="accuracy", refit=True)
    grid_knn.fit(x_train, y_train)
    print("Best Score ==> ", grid_knn.best_score_)
    print("Best Parameters ==> ", grid_knn.best_params_)
    # print("Accuracy on Train Set ==> ", grid_knn.score(x_train, y_train))
    print("Accuracy on Test Set ==>", grid_knn.score(x_test, y_test))

    y_knn_pred = grid_knn.predict(x_test)
    f1 = f1_score(y_test, y_knn_pred)
    print("knn_f1 = ", f1)

    sn = cal_sn(y_test, y_knn_pred)
    sp = cal_sp(y_test, y_knn_pred)
    print("sn = %s, sp = %s" % (sn, sp))

    cnf_matrix = confusion_matrix(y_test, y_knn_pred)
    # sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, fmt="g")
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g")
    plt.title("混淆矩阵", y=1.01)
    plt.ylabel("真实值")
    plt.xlabel("预测值")
    if pic_save:
        plt.savefig(os.path.join(pic_path, "knn_matric"), dpi=300)
    plt.show()

    probs = grid_knn.predict_proba(x_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([-0.1, 1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

    if model_save:
        if model_save_name is not None:
            model_name = model_save_name + ".pickle"
        else:
            model_name = "knn.pickle"

        with open(model_name, "wb") as f:
            pickle.dump(grid_knn, f)


def my_bayes(x_train, y_train, x_test, y_test, pic_save=False):
    """朴素贝叶斯"""
    print("-----------------Naive Bayes-------------------")
    bayes = GaussianNB().fit(x_train, y_train)
    y_predict = bayes.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print("bayes_accuracy = ", accuracy)
    f1 = f1_score(y_test, y_predict)
    print("bayes_f1 = ", f1)

    sn = cal_sn(y_test, y_predict)
    sp = cal_sp(y_test, y_predict)
    print("sn = %s, sp = %s" % (sn, sp))
    print(classification_report(y_test, y_predict))

    cnf_matrix = confusion_matrix(y_test, y_predict)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g")
    plt.title("混淆矩阵", y=1.01)
    plt.ylabel("真实值")
    plt.xlabel("预测值")
    if pic_save:
        plt.savefig(os.path.join(pic_path, "bayes_matric"), dpi=300)
    plt.show()

    probs = bayes.predict_proba(x_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([-0.1, 1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right")
    plt.show()


def my_random_forest(x_train, y_train, x_test, y_test, pic_save=False, model_save=False, model_save_name=None):
    print("---------------Random Forest-------------------")
    rfc = RandomForestClassifier(n_estimators=50)
    rfc.fit(x_train, y_train)
    y_predict = rfc.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print("accuracy = ", accuracy)
    f1 = f1_score(y_test, y_predict)
    print("rf_f1 = ", f1)

    sn = cal_sn(y_test, y_predict)
    sp = cal_sp(y_test, y_predict)
    print("sn = %s, sp = %s" % (sn, sp))
    print(classification_report(y_test, y_predict))
    cnf_matrix = confusion_matrix(y_test, y_predict)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g")
    plt.title("混淆矩阵", y=1.01)
    plt.ylabel("真实值")
    plt.xlabel("预测值")
    if pic_save:
        plt.savefig(os.path.join(pic_path, "randomForest_matric"), dpi=300)
    plt.show()

    # 绘制ROC曲线
    probs = rfc.predict_proba(x_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([-0.1, 1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

    if model_save:
        if model_save_name is not None:
            model_name = model_save_name + ".pickle"
        else:
            model_name = "randomForest.pickle"

        with open(model_name, "wb") as f:
            pickle.dump(rfc, f)

def my_random_forest_grid(x_train, y_train, x_test, y_test):
    rfc = RandomForestClassifier()
    param_grid = {"n_estimators": np.arange(1, 100),
                  "max_depth": np.arange(1, 10),
                  "max_features": ["auto", "sqrt"],
                  "criterion": ["gini", "entropy"]}
    grid_rfc = GridSearchCV(rfc, param_grid, scoring="accuracy", refit=True)
    grid_rfc.fit(x_train, y_train)
    print("Best Score ==> ", grid_rfc.best_score_)
    print("Best Parameters ==> ", grid_rfc.best_params_)
    print("Accuracy on Train Set ==> ", grid_rfc.score(x_train, y_train))
    print("Accuracy on Test Set ==>", grid_rfc.score(x_test, y_test))

    model = grid_rfc.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    cnf_matrix = confusion_matrix(y_test, y_predict)

    probs = grid_rfc.predict_proba(x_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right")
    plt.show()


def my_decision_tree(x_train, y_train, x_test, y_test, pic_save=False):
    print("---------------Decision Tree-------------------")
    dtc = DecisionTreeClassifier()
    param_grid = {"criterion": ["gini", "entropy"],
                  "max_depth": np.arange(2, 30),
                  "min_samples_leaf": np.arange(0.1, 1)}
    grid_dtc = GridSearchCV(dtc, param_grid, scoring="accuracy", refit=True)
    grid_dtc.fit(x_train, y_train)
    print("Best Score ==> ", grid_dtc.best_score_)
    print("Best Parameters ==> ", grid_dtc.best_params_)
    print("Accuracy on Train Set ==> ", grid_dtc.score(x_train, y_train))
    print("Accuracy on Test Set ==>", grid_dtc.score(x_test, y_test))

    model = grid_dtc.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    f1 = f1_score(y_test, y_predict, average="binary")
    print("decison tree f1 = ", f1)
    sn = cal_sn(y_test, y_predict)
    sp = cal_sp(y_test, y_predict)
    print("sn = %s, sp = %s" % (sn, sp))

    cnf_matrix = confusion_matrix(y_test, y_predict)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g")
    plt.title("混淆矩阵", y=1.01)
    plt.ylabel("真实值")
    plt.xlabel("预测值")
    if pic_save:
        plt.savefig(os.path.join(pic_path, "decisionTree_matric"), dpi=300)
    plt.show()

    probs = model.predict_proba(x_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right")
    plt.show()


def my_svm(x_train, y_train, x_test, y_test, pic_save=False, model_save=False, model_save_name=None):
    print("------------------SVM------------------")
    # model = svm.SVC(probability=True)
    # params = [
    #     {"kernel": ["linear"], "C": [1, 10]},
    #     {"kernel": ["rbf"], "C": [1, 10], "gamma": [0.1, 1]}
    # ]
    # grid_svc = GridSearchCV(estimator=model, param_grid=params, cv=3)
    # grid_svc.fit(x_train, y_train)
    svm_model = svm.SVC(probability=True, tol=0.001, cache_size=200, verbose=False, decision_function_shape="ovr")
    kernel = ["linear", "rbf", "sigmoid", "poly"]
    C = [0.01, 0.1, 1, 10]
    max_iter = [10, 20, 30, 50, 70, 100, 200]
    params = dict(kernel=kernel, C=C, max_iter=max_iter)
    grid_svc = GridSearchCV(svm_model, param_grid=params, cv=3)
    grid_svc.fit(x_train, y_train)
    print("Best Score ==> ", grid_svc.best_score_)
    print("Best Parameters ==> ", grid_svc.best_params_)
    # print("Accuracy on Train Set ==> ", grid_svc.score(x_train, y_train))
    print("Accuracy on Test Set ==>", grid_svc.score(x_test, y_test))
    # print(accuracy_score(y_train, grid_svc.predict(x_train)))
    # print(accuracy_score(y_test, grid_svc.predict(x_test)))

    y_predict = grid_svc.predict(x_test)
    # print("predict---", y_predict)
    f1 = f1_score(y_test, y_predict, average="binary")
    print("svm_f1 = ", f1)

    sn = cal_sn(y_test, y_predict)
    sp = cal_sp(y_test, y_predict)
    print("sn = %s, sp = %s" % (sn, sp))
    accuracy = accuracy_score(y_test, y_predict)
    # print("accuracy = ", accuracy)
    # print(classification_report(y_test, y_predict))
    cnf_matrix = confusion_matrix(y_test, y_predict)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g")
    plt.title("混淆矩阵", y=1.01)
    plt.ylabel("真实值")
    plt.xlabel("预测值")
    if pic_save:
        plt.savefig(os.path.join(pic_path, "svm_matric"), dpi=300)
    plt.show()

    # 绘制ROC曲线
    probs = grid_svc.predict_proba(x_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([-0.1, 1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

    if model_save:
        if model_save_name is not None:
            model_name = model_save_name + ".pickle"
        else:
            model_name = "svm.pickle"
        with open(model_name, "wb") as f:
            pickle.dump(grid_svc, f)











