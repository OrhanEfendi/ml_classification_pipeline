from sklearn.impute import KNNImputer
class imputer:
  def __init__(self,xtrn,xtst):
    self.xtr=xtrn
    self.xts=xtst
  def run(self):
    self.knn=KNNImputer()
    self.xtr_new=self.knn.fit_transform(self.xtr)
    self.xts_new=self.knn.transform(self.xts)
    return self.xtr_new,self.xts_new