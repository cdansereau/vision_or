__author__ = 'Christian Dansereau'

import numpy as np
#from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
import predlib as plib
from sklearn import preprocessing
import fselection as fselect
#data_path = '/home/cdansereau/Dropbox/McGill-publication/Papers/PredicAD/prediction_data_p2p/data_connec100_prediction.csv'
#data_path = '/home/cdansereau/Dropbox/McGill-publication/Papers/PredicAD/prediction_data_p2p/data_connec100_prediction_compcor.csv'


from sklearn import linear_model

class ConfoundsRm:

    def __init__(self, confounds, data):
        self.fit(data, confounds)
        
    def fit(self, confounds, data):
        self.reg = linear_model.LinearRegression(fit_intercept=True)
        self.reg.fit(data, confounds)
        
    def transform(self, confounds, data):
        # compute the residual error
        
        return data - self.reg.predict(confounds)

def modelpred(x,y):
    # Preporcessing 
    #scaler = preprocessing.StandardScaler().fit(x)
    #x_scaled = scaler.transform(x)
    x_scaled = x
    #encoder = preprocessing.LabelEncoder()
    #encoder.fit(y)

    # Feature selection
    w = fselect.irelief_cross(x_scaled,y,10)
    #candidat_f = fselect.nBest(w,108)
    fselect.threhold_std()
    
    # grid search and SVM
    clf = get_opt_model(x_scaled[:,candidat_f],y)
    #clf = svm.SVC(kernel='rbf', class_weight='auto')
    #clf = plib.grid_search(clf, x_scaled[:,candidat_f], y, n_folds=10, verbose=True)
    #clf.fit(x_scaled[:,candidat_f],y)
    #clf = plib.classif(x_scaled[:,candidat_f],y)
    return clf, candidat_f, w

def get_opt_model_features_nbest(x,y,w):
    n = len(fselect.threhold_std(w,1))
    best_score = 0
    best_clf = []
    best_i = 0
    all_nfeatures = []
    all_scores = []
    for i in range(5,int(n),20):
         
        # grid search and SVM
        candidat_f = fselect.nBest(w,i)
        clf = svm.SVC(kernel='linear', class_weight='auto')
        clf.probability = True
        clf, score = plib.grid_search(clf, x[:,candidat_f], y, n_folds=10, verbose=False)
        #print score
        all_nfeatures.append(i)
        all_scores.append(score)
        if score > best_score:
            print 'New best: ', best_score
            best_score = score
            best_clf = clf
            best_i = i

    return best_clf,fselect.nBest(w,i),best_i,all_nfeatures,all_scores

def get_opt_model_features_std(x,y,w,std_step=0.25):
    best_score = 0
    best_clf = []
    best_i = 0
    all_nfeatures = []
    all_scores = []
    for i in np.arange(0,2.75,std_step):

        # grid search and SVM
        candidat_f = fselect.threhold_std(w,i)
        print candidat_f.shape
        if len(candidat_f)>0:
            clf = svm.SVC(kernel='linear', class_weight='auto',C=0.01)
            clf.probability = True
            clf, score = plib.grid_search(clf, x[:,candidat_f], y, n_folds=10, verbose=False)
            #print score
            all_nfeatures.append(i)
            all_scores.append(score)
            if score > best_score:
                best_score = score
                best_clf = clf
                best_i = i
                print 'New best: ', best_score

    return best_clf,fselect.threhold_std(w,best_i),best_i,all_nfeatures,all_scores

def sv_metric(n,nsv):
    return nsv/float(n)
    #return (n-nsv)/float(n) #lower the n sv is greater the score

def get_opt_model_features_std_sv(x,y,w,alpha=0.8):
    best_score = 0
    best_clf = []
    best_i = 0
    best_nsv = 0
    all_nfeatures = []
    all_scores = []
    for i in np.arange(1,2.75,0.25):

        # grid search and SVM
        candidat_f = fselect.threhold_std(w,i)
        if len(candidat_f)>0:
            clf = svm.SVC(kernel='linear', class_weight='auto')
            clf.probability = True
            clf, score = plib.grid_search(clf, x[:,candidat_f], y, n_folds=10, verbose=False)
            #print score
            all_nfeatures.append(i)
            all_scores.append(score)
            if best_score < (score - alpha*sv_metric(x.shape[0],np.sum(clf.n_support_))):
                best_score = score-alpha*sv_metric(x.shape[0],np.sum(clf.n_support_))
                best_clf = clf
                best_i = i
                best_nsv = clf.n_support_
                print 'New best: ', best_score

    return best_clf,fselect.threhold_std(w,best_i),best_i,all_nfeatures,all_scores,best_nsv

def get_opt_model(x,y):

    # grid search and SVM
    clf = svm.SVC(kernel='rbf', class_weight='auto')
    clf.probability = True
    #clf = svm.SVC(kernel='rbf')
    clf, best_score = plib.grid_search(clf, x, y, n_folds=10, verbose=False)
    clf.fit(x,y)
    return clf

def basicconn(skf,X,y):
    total_score = 0
    for train_index, test_index in skf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        # Feature selection
        #selectf = SelectFpr().fit(X[train_index],y[train_index])
        #selectf = SelectKBest(f_classif, k=750).fit(X[train_index],y[train_index])
        #tmp_x = selectf.transform(X[train_index])
        # Train
        #clf = RandomForestClassifier(n_estimators=20)
        #clf = clf.fit(tmp_x, y[train_index])
        #clf.feature_importances_
        # SVM
        #clf = svm.LinearSVC()
        #clf = svm.SVC()
        #clf.fit(tmp_x, y[train_index])
        clf = plib.classif(X[train_index], y[train_index])
        #clf.support_vec()
        # Test
        #pred = clf.predict(selectf.transform(X[test_index]))
        pred = clf.predict(X[test_index])
        print "Target     : ", y[test_index]
        print "Prediction : ", pred
        matchs = np.equal(pred, y[test_index])
        score = np.divide(np.sum(matchs), np.float64(matchs.size))
        total_score = score + total_score
    return np.divide(total_score, skf.n_folds)


def splitconn(skf,X,y):
    total_score = 0
    for train_index, test_index in skf:
        # Train
        clf1 = plib.classif(X[train_index, 0:2475:1], y[train_index])
        clf2 = plib.classif(X[train_index, 2475:4950:1], y[train_index])
        pred1 = clf1.decision_function(X[train_index, 0:2475:1])
        pred2 = clf2.decision_function(X[train_index, 2475:4950:1])
        clf3 = svm.SVC()
        y[train_index].shape
        np.array([pred1, pred2])
        clf3.fit(np.array([pred1, pred2]).transpose(), y[train_index])
        #clf3 = plib.classif(np.matrix([pred1,pred2]).transpose(),y[train_index])

        # Test
        pred1 = clf1.decision_function(X[test_index, 0:2475:1])
        pred2 = clf2.decision_function(X[test_index, 2475:4950:1])
        predfinal = clf3.predict(np.matrix([pred1, pred2]).transpose())
        print "Target     : ", y[test_index]
        print "Prediction : ", predfinal
        matchs = np.equal(predfinal, y[test_index])
        score = np.divide(np.sum(matchs), np.float64(matchs.size))
        total_score = score + total_score
    return np.divide(total_score, skf.n_folds)


def multisplit(skf,X,y,stepsize=1000):
    total_score = 0
    for train_index, test_index in skf:
        wl = []
        pred1 = np.matrix([])
        # Training
        for x in range(0, len(X[0]), stepsize):
            clf1 = plib.classif(X[train_index, x:x + stepsize], y[train_index])
            tmp_p = np.matrix(clf1.decision_function(X[train_index, x:x + stepsize]))
            if pred1.size == 0:
                pred1 = tmp_p
            else:
                pred1 = np.concatenate((pred1, tmp_p), axis=1)
            wl.append(clf1)
        #selectf = SelectKBest(f_classif, k=5).fit(pred1, y[train_index])
        selectf = SelectFpr().fit(pred1, y[train_index])
        clf3 = AdaBoostClassifier(n_estimators=100)
        #clf3 = svm.SVC(class_weight='auto')
        #clf3 = RandomForestClassifier(n_estimators=20)
        clf3.fit(selectf.transform(pred1), y[train_index])
        # Testing
        predtest = np.matrix([])
        k = 0
        for x in range(0, len(X[0]), stepsize):
            tmp_p = np.matrix(wl[k].decision_function(X[test_index, x:x + stepsize]))
            if predtest.size == 0:
                predtest = tmp_p
            else:
                predtest = np.concatenate((predtest, tmp_p), axis=1)
            k += 1
        # Final prediction
        predfinal = clf3.predict(selectf.transform(predtest))
        print "Target     : ", y[test_index]
        print "Prediction : ", predfinal
        matchs = np.equal(predfinal, y[test_index])
        score = np.divide(np.sum(matchs), np.float64(matchs.size))
        total_score = score + total_score
    return np.divide(total_score, skf.n_folds)

