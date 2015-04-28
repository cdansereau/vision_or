from Proteus.proteus import matrix

__author__ = 'Christian Dansereau'

from ..matrix import tseries as ts
import pandas as pd
import clustering as cls
from sklearn import linear_model
import numpy as np
import prediction
from sklearn import cross_validation

class classif:
   'Prediction tool for multiscale functional neuro imaging'
   empCount = 0

   def __init__(self, x, y, n_cluster):
       data = pd.DataFrame(x, index=y)
       #avg_conn_class0v = data.loc[0].mean(axis=0)
       #avg_conn_class1v = data.loc[1].mean(axis=0)
       #avg_conn_class0m = ts.vec2mat(avg_conn_class0v)
       #avg_conn_class1m = ts.vec2mat(avg_conn_class1v)
       # GLM of the groups
       clf = linear_model.LinearRegression()
       clf.fit (data, data.index.values)
       beta = ts.vec2mat(clf.coef_) # Beta matrix
       beta[np.where(np.identity(beta.shape[0]))] = 0
       # hierachical clustering on the beta matrix
       ind = cls.hclustering(beta, n_cluster)

       # test if clustering works
       #a=cls.ind2matrix(ind)
       #b=cls.ordermat(a,ind) # should return a matrix with the ordered cluster on the diagonal

       # average resulting new partition
       # obtain the new individual conectomes
       vec_features = pd.DataFrame()
       for i1 in range(0, data.shape[0]):
           m = ts.vec2mat(data.iloc[i1, :])
           m = cls.part(m, ind)
           s = pd.Series(ts.mat2vec(m))
           vec_features = vec_features.append(s, ignore_index=True)

       # Training and prediction
       skf = cross_validation.StratifiedKFold(y, n_folds=10)
       score = prediction.basicconn(skf, vec_features.values, y)
       print score

       #prediction.multisplit(skf, vec_features.values, y)


   def predict(self, x):
      # Test
      x_select = self.selectf.transform(x)
      x_select = self.scaler.transform(x_select)
      if len(x_select[0]) != 0:
         pred = self.clf.predict(x_select)
         #print "Prediction : ", pred
         return pred

   def decision_function(self, x):
      # return the decision function
      x_select = self.selectf.transform(x)
      x_select = self.scaler.transform(x_select)
      if len(x_select[0]) != 0:
         df = self.clf.decision_function(x_select)
         return df
      else:
         return []
         print "ZERO!!"
      #print "Decision function : ", df
      #return df

   def support_vec(self):
      # get indices of support vectors
      idx_svec = self.clf.support_
      idx_global = self.selectf.get_support(True)
      return idx_global[idx_svec]
