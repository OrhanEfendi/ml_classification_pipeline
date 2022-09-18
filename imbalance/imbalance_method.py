
import imblearn
from imblearn.under_sampling import (
    RandomUnderSampler,
    CondensedNearestNeighbour,
    TomekLinks,
    OneSidedSelection,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    NeighbourhoodCleaningRule,
    NearMiss,
    InstanceHardnessThreshold
)
from imblearn.over_sampling import(
    RandomOverSampler,
    SMOTE,
    SMOTEN,
    SMOTENC,
    BorderlineSMOTE,
    ADASYN,
    BorderlineSMOTE,
    SVMSMOTE,
    KMeansSMOTE
    
)
from sklearn.svm import SVC
from sklearn.cluster import KMeans
class imbalanceHandling:
    #Creation of a new training dataset by resampling the old dataset according to the selected imbalance method
    def __init__(self, Xtrain, ytrain, method):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.method = method

    def undersampling_method(self):
        self.aii = AllKNN(
            sampling_strategy='auto',
            n_neighbors=3,
            kind_sel='all',
            n_jobs=-1,
        )
        ##############################################################################################################################################################
        self.rm = RandomUnderSampler(
            sampling_strategy='auto',

            replacement=False,
        )
        ######################################################################################################################################################
        self.tm = TomekLinks(
            sampling_strategy='auto',
            n_jobs=4,
        )
        ########################################################################################################################
        self.one = OneSidedSelection(
            sampling_strategy='auto',
            n_neighbors=1,
            n_jobs=-1,
        )
        ##########################################################################################
        self.enn = EditedNearestNeighbours(
            sampling_strategy='auto',
            n_neighbors=3,
            kind_sel='all',
            n_jobs=-1,
        )
        ##########################################################################################
        self.renn = RepeatedEditedNearestNeighbours(
            sampling_strategy='auto',
            n_neighbors=3,
            kind_sel='all',
            n_jobs=-1,
            max_iter=100,
        )
        ##########################################################################################
        self.neg = NeighbourhoodCleaningRule(
            sampling_strategy='auto',
            n_neighbors=3,
            kind_sel='all',
            n_jobs=4,
            threshold_cleaning=0.5,
        )

        return self.aii, self.rm, self.tm, self.one, self.enn, self.renn, self.neg

    def oversampling_method(self):
        self.ros = RandomOverSampler(sampling_strategy="auto",
                                     )
        ######################################################################################################################################################

        self.ros_sh = RandomOverSampler(sampling_strategy="auto",

                                        shrinkage=5,
                                        )
        ######################################################################################################################################################
        self.sm = SMOTE(sampling_strategy="auto",

                        k_neighbors=5,
                        )
        ######################################################################################################################################################
        self.smn = SMOTEN(sampling_strategy="auto",

                          k_neighbors=5,
                          n_jobs=-1,
                          )
        ######################################################################################################################################################
        self.ada = ADASYN(sampling_strategy="auto",

                          n_neighbors=5,
                          n_jobs=-1,
                          )
        ######################################################################################################################################################
        self.svmsm = SVMSMOTE(sampling_strategy="auto",

                              k_neighbors=5,
                              m_neighbors=10,
                              n_jobs=-1,
                              svm_estimator=SVC(kernel="linear"))
        ######################################################################################################################################################
        
        ####################################################################################################################################################################################
        return self.ros, self.ros_sh, self.sm, self.smn, self.ada, self.svmsm

    def methods(self, methods_name): 
        
        
        self.Xtrain_m, self.y_train_m = methods_name.fit_resample(self.Xtrain, self.ytrain)
        return self.Xtrain_m, self.y_train_m

    def run(self):
        self.value = dict()
        self.value_2 = dict()
        if (self.method == "undersampling"):
            for met in self.undersampling_method():
                self.Xtrn, self.ytrn = self.methods(met)
                self.value.update({f"{met}": [self.Xtrn, self.ytrn]})
            return self.value

        else :
            for met2 in self.oversampling_method():
                self.Xtrn_1, self.ytrn_1 = self.methods(met2)
                self.value_2.update({f"{met2}": [self.Xtrn_1, self.ytrn_1]})
            return self.value_2