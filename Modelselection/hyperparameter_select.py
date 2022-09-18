from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import HalvingGridSearchCV
import optuna
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
class Hyperparameter:
    def __init__(self, X_tr, X_ts, Y_tr, Y_ts):
        self.X_tr = X_tr
        self.X_ts = X_ts
        self.Y_tr = Y_tr
        self.Y_ts = Y_ts

    def Hyper_optuna(self):
        self.random_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(),
        )

        self.tpe_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler())

        self.cmaes_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.CmaEsSampler())
        self.random_study.optimize(self.objective_optuna, n_trials=100)
        self.cmaes_study.optimize(self.objective_optuna, n_trials=100)
        self.tpe_study.optimize(self.objective_optuna, n_trials=100)

        self.result_optuna = {"random_study_best_params": [self.random_study.best_params],
                              "random_study_best_value": [self.random_study.best_value],
                              "cmaes_study_best_params": [self.cmaes_study.best_params],
                              "cmaes_study_best_value": [self.cmaes_study.best_value],
                              "tpe_study_best_params": [self.tpe_study.best_params],
                              "tpe_study_best_value": [self.tpe_study.best_value]}
        return pd.DataFrame(data=self.result_optuna.values(), index=self.result_optuna.keys(), columns=["Value"])

    def objective_optuna(self, trial):
        self.max_depth = trial.suggest_int("max_depth", 1, 500)
        self.n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        self.min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 1000)
        self.criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        self.min_samples_split = trial.suggest_int("min_samples_split", 2, 500)
        self.class_weight = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"])
        self.max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

        ## Create Model
        self.classifier = RandomForestClassifier(n_estimators=self.n_estimators,
                                                 min_samples_leaf=self.min_samples_leaf,
                                                 min_samples_split=self.min_samples_split,
                                                 criterion=self.criterion,
                                                 class_weight=self.class_weight,
                                                 max_features=self.max_features,
                                                 random_state=1)
        self.classifier.fit(self.X_tr, self.Y_tr)

        return self.classifier.score(self.X_ts, self.Y_ts)

    def Halving_Random(self):
        self.parameter_RandomForestClassifier = {"n_estimators": [10,50,60,100,400,600,1000],
                                                 "min_samples_leaf": [2,10,20,15,30,40,100],
                                                 "min_samples_split": [2,10,20,15,30,40,100],
                                                 'max_depth': [2,10,20,15,30,40,100],
                                                 "criterion": ["gini", "entropy"],
                                                 "class_weight": ["balanced", "balanced_subsample"],
                                                 "max_features": ["sqrt", "log2", None]}
        self.model = RandomForestClassifier(random_state=1)
        self.searh_random = HalvingRandomSearchCV(self.model, self.parameter_RandomForestClassifier, random_state=1,
                                                  n_jobs=-1).fit(self.X_tr, self.Y_tr)
        self.result_random = {"Halving_random_best_params": [self.searh_random.best_params_],
                              "Halving_random_best_score": [self.searh_random.score(self.X_ts, self.Y_ts)]}
        return pd.DataFrame(data=self.result_random.values(), index=self.result_random.keys(), columns=["Value"])

    def Halving_Grid(self):
        self.parameter_RandomForest_Classifier = {"n_estimators": [10, 50, 100],
                                                  "min_samples_leaf": [2, 5, 100],
                                                  "min_samples_split": [2, 5, 100],
                                                  'max_depth': [None, 1, 10, 50],
                                                  "criterion": ["gini", "entropy"],
                                                  "class_weight": ["balanced", "balanced_subsample"],
                                                  "max_features": ["sqrt", "log2", None]}
        self.model_gr = RandomForestClassifier(random_state=1)
        self.searh_grid =HalvingGridSearchCV(self.model_gr, self.parameter_RandomForest_Classifier, random_state=1,
                                              n_jobs=-1).fit(self.X_tr, self.Y_tr)
        self.result_grid = {"Halving_grid_best_params": [self.searh_grid.best_params_],
                            "Halving_Grid_best_score": [self.searh_grid.score(self.X_ts, self.Y_ts)]}
        return pd.DataFrame(data=self.result_grid.values(), index=self.result_grid.keys(), columns=["Value"])

    def run(self):
        results_list = []
        for hyper in [self.Halving_Random(),self.Hyper_optuna(), self.Halving_Grid()]:
            self.result_all = hyper
            results_list.append(self.result_all)
        return pd.concat(results_list)