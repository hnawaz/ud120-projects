#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###


### import SVM and Accuracy Package
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### slicing the training set down
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

### create and train classifier
t0 = time()
clf = SVC(kernel="rbf", C=10000)
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"


### make prediction
t1 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"


### print predictions based on indices
#print "prediction for indices 10,26 and 50 are :", pred[10], pred[26], pred[50], "respectively" 
print "total number of predicted 1s are:", pred.tolist().count(1)


### test accuracy of prediction
accuracy = accuracy_score(labels_test,pred)

### print out accuracy
print accuracy


#########################################################


