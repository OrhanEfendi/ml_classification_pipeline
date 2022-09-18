from sklearn.metrics import (accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report, 
f1_score,recall_score,precision_score,balanced_accuracy_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.model_selection import cross_val_score
class Modeling:
    def __init__(self, Xtst, ytst, algohritm, bal_name):
        self.Xtst = Xtst
        self.ytst = ytst
        self.algohritm = algohritm
        self.bal_name = bal_name

    def show_confusion_matrix(self, ytestt, ypredd, balancing):
        self.cf_matrix = confusion_matrix(ytestt, ypredd)
        self.con = sns.heatmap(self.cf_matrix, annot=True, cmap='Blues')
        self.con.set_title(balancing);
        self.con.set_xlabel('\nPredicted Values')
        self.con.set_ylabel('Actual Values ');
        self.con.xaxis.set_ticklabels(['False', 'True'])
        self.con.yaxis.set_ticklabels(['True', 'False'])
        plt.show()

    def check_metrics(self, ytst, ytrain, ypred_test, ypred_train):
        self.balanced_accuracy_test = balanced_accuracy_score(ytst, ypred_test)
        self.recall_test = recall_score(ytst, ypred_test)
        self.precision_test = precision_score(ytst, ypred_test)
        self.f1_score_test = f1_score(ytst, ypred_test)
        self.pred_matthew_score_test = matthews_corrcoef(ytst, ypred_test)
        self.balanced_accuracy_train = balanced_accuracy_score(ytrain, ypred_train)
        self.pred_matthew_score_train = matthews_corrcoef(ytrain, ypred_train)
        self.metrics_s = {"f1_score_test": self.f1_score_test,
                          "balanced_accuracy_test": self.balanced_accuracy_test, "recall_test": self.recall_test,
                          "precision_test": self.precision_test,
                          "pred_matthew_score_train": self.pred_matthew_score_train,
                          "self.balanced_accuracy_train": self.balanced_accuracy_train
                          }

    def fit_evaluate_model(self, Xtr, Xts, ytr, yts, balancing_name):
        self.model = self.algohritm
        self.model.fit(Xtr, ytr)
        self.cv_score_train = cross_val_score(self.model, Xtr, ytr, scoring="accuracy", cv=20)
        self.ypred_tr = self.model.predict(Xtr)
        self.ypred_ts = self.model.predict(Xts)
        self.score_tr = self.cv_score_train.mean() * 100
        self.score_ts = self.model.score(Xts, yts)
        self.balanced_accuracy_test = balanced_accuracy_score(yts, self.ypred_ts)
        self.recall_test = recall_score(yts, self.ypred_ts)
        self.precision_test = precision_score(yts, self.ypred_ts)
        self.f1_score_test = f1_score(yts, self.ypred_ts)
        self.pred_matthew_score_test = matthews_corrcoef(yts, self.ypred_ts)
        self.balanced_accuracy_train = balanced_accuracy_score(ytr, self.ypred_tr)
        self.pred_matthew_score_train = matthews_corrcoef(ytr, self.ypred_tr)
        self.confusion_matrix = self.show_confusion_matrix(yts, self.ypred_ts, balancing_name)
        self.results_all = {"Method_name": balancing_name, "Test_score": self.score_ts, "Train_score": self.score_tr,
                            "f1_score_test": self.f1_score_test,
                            "balanced_accuracy_test": self.balanced_accuracy_test, "recall_test": self.recall_test,
                            "precision_test": self.precision_test,
                            "pred_matthew_score_test": self.pred_matthew_score_test,
                            "pred_matthew_score_train": self.pred_matthew_score_train,
                            "self.balanced_accuracy_train": self.balanced_accuracy_train}

        print(self.confusion_matrix)
        return self.results_all

    def run(self):
        self.results_list = []
        for balance_name, value in self.bal_name.items():
            self.results = self.fit_evaluate_model(value[0], self.Xtst, value[1], self.ytst, balance_name)
            self.results_list.append(self.results)

        return pd.DataFrame(data=self.results_list)