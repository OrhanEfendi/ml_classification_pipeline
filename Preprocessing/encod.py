from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders import TargetEncoder
import pandas as pd
import numpy as np

class Encoding:
  def __init__(self,xtr,xts,ytr,yts,encod,feature_name):
    self.xtr=xtr
    self.xts=xts
    self.encod=encod
    self.ytr=pd.DataFrame(ytr)
    self.yts=pd.DataFrame(yts)
    self.feature_name=feature_name
  def encoder(self):
    if (self.encod=="target") and (self.feature_name=="all"):
      self.enc=TargetEncoder()
      self.categ1=self.xtr.select_dtypes(include="object").columns
      for categorical1 in self.xtr[self.categ1]:
        self.xtr[categorical1]=self.enc.fit_transform(self.xtr[categorical1],self.ytr)
        self.xts[categorical1]=self.enc.transform(self.xts[categorical1],self.yts)
      return self.xtr,self.xts
    elif (self.encod=="target"):
      self.enc1=TargetEncoder()
      for name in self.feature_name:
        self.xtr[name]=self.enc1.fit_transform(self.xtr[name],self.ytr)
        self.xts[name]=self.enc1.transform(self.xts[name],self.yts)
      return self.xtr,self.xts
    elif (self.encod=="label") and (self.feature_name=="all"):
      self.enc2=LabelEncoder()
      self.categ2=self.xtr.select_dtypes(include="object").columns
      for categorical2 in self.xtr[self.categ2]:
        self.xtr[categorical2]=self.enc2.fit_transform(self.xtr[categorical2])
        self.xts[categorical2]=self.enc2.transform(self.xts[categorical2])
        return self.xtr,self.xts
    elif (self.encod=="label"):
      if (self.feature_name[0] not in self.xtr):
        self.enc3=LabelEncoder()
        for self.name1 in self.feature_name:
          self.ytr[self.name1]=self.enc3.fit_transform(self.ytr[self.name1])
          self.yts[self.name1]=self.enc3.transform(self.yts[self.name1])
          return self.ytr,self.yts
      else:
        self.enc4=LabelEncoder()
        for name2 in self.feature_name:
          self.xtr[name2]=self.enc4.fit_transform(self.xtr[name2])
          self.xts[name2]=self.enc4.transform(self.xts[name2])
        return self.xtr,self.xts