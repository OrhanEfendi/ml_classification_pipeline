from sklearn.preprocessing import StandardScaler
class Scaler:
  def __init__(self,xtrain,xtest,scaler):
    self.xtrain=xtrain
    self.xtest=xtest
    self.scaler=scaler
  def scale(self):
    if (self.scaler=="standard"):
      self.sc=StandardScaler()
      self.xtrain_scaled=self.sc.fit_transform(self.xtrain)
      self.xtest_scaled=self.sc.transform(self.xtest)
      return self.xtrain_scaled,self.xtest_scaled
    else:
      self.min_max=MinMaxScaler()
      self.xtrain_m_scale=self.min_max.fit_transform(self.xtrain)
      self.xtest_m_scale=self.min_max.transform(self.xtest)
      return self.xtrain_m_scale,self.m.xtest_m_scale