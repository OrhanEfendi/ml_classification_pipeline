class DropFeatures:
    def __init__(self,dataframe):
        self.dataframe=dataframe
        self.drop_cols = [c for c in list(self.dataframe) if self.dataframe[c].nunique() <= 1]
        self.dataframe = self.dataframe.drop(columns=self.drop_cols)
    def remove_features(self,feature_name):
        new_dataframe=self.dataframe.drop(feature_name,axis=1)
        return new_dataframe