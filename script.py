# TITANIC KAGGLE BY SHAGUN SRIVASTAVA 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series,DataFrame
import csv
import matplotlib.pyplot as plt
import random
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from featureformat import featureFormat, targetFeatureSplit, testtargetFeatureSplit
from featurecreation import newfeature
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


#READ TRAINING AND TESTING DATA
titanic_train = pd.read_csv("train.csv")
titanic_test = pd.read_csv("test.csv")

# CONVERTING TRAINING AND TESTING DATA INTO DICTIONARY FORMAT
def features_dictionary():
    reader = csv.DictReader(open('train.csv', 'rb'))
    reader1 = csv.DictReader(open('test.csv', 'rb'))
    data_dict = []
    test_dict = []
    for line in reader:
        data_dict.append(line)

    for line in reader1:
        test_dict.append(line)
        
    new_data_dict = {}
    new_test_dict = {}

    #train
    for line in data_dict:
        key = line['PassengerId']
        del (line['PassengerId'])
        value = line
        new_data_dict[key] = value

    #test
    for line in test_dict:
        key = line['PassengerId']
        del (line['PassengerId'])
        del (line['Name'])
        del (line['Ticket'])
        del (line['Cabin'])
        value = line
        new_test_dict[key] = value


    return new_data_dict, new_test_dict

dictionary , test_dict = features_dictionary()


# FILLING NULL VALUES IN TRAINING AND TESTING DATA

def fillagevalues():
    ageslist = []
    for key in dictionary:
        if((dictionary[key]['Age']) != ''):
            ageslist.append(float(dictionary[key]['Age']))
               
    arr = np.array([ageslist])
    avg_age = np.mean(arr)
    std_age = np.std(arr)

    for key in dictionary:
        if((dictionary[key]['Age']) == ''):
            dictionary[key]['Age'] = round(random.uniform(avg_age-std_age,avg_age+std_age),1)
            

    # In testing data
    ageslist1 = []
    for key in test_dict:
        if((test_dict[key]['Age']) != ''):
            ageslist1.append(float(test_dict[key]['Age']))
               
    arr = np.array([ageslist1])
    avg_age = np.mean(arr)
    std_age = np.std(arr)

    for key in test_dict:
        if((test_dict[key]['Age']) == ''):
            test_dict[key]['Age'] = round(random.uniform(avg_age-std_age,avg_age+std_age),1)
            



def fillembarkvalues():
    for key in dictionary:
        if((dictionary[key]['Embarked']) == ''):
            dictionary[key]['Embarked'] = 'S'
            

# In testing data
def fillfarevalues():
    for key in new_test_dict:
        farelist = []
        if(new_test_dict[key]['Fare'] != ''):
            farelist.append(float(new_test_dict[key]['Fare']))


    arr = np.array([farelist])
    new_test_dict['1044']['Fare'] = round((np.median(arr)),1)


temp_dict = dictionary
new_test_dict = test_dict


fillagevalues()
fillembarkvalues()
fillfarevalues()


# CREATE NEW FEATURES
newfeature(temp_dict, new_test_dict)


features_list = ['Survived', 'Fare', 'Age', 'Alone', 'Pclass', 'Child' ,'Female', 'Male', 'S', 'C', 'Q']
data = featureFormat(temp_dict, features_list,remove_NaN=True)
labels, features = targetFeatureSplit(data)


# FEATURE SELECTION;SELECTING K BEST FEATURES

k=5
k_best = SelectKBest(k=k)
k_best.fit(features, labels)
scores = k_best.scores_
print(scores)

features_list = ['Survived', 'Fare', 'Age', 'Alone', 'Pclass', 'Child' ,'Female', 'Q', 'C']

data = featureFormat(temp_dict, features_list,remove_NaN=True)
labels, features = targetFeatureSplit(data)

features_list = ['Fare', 'Age', 'Pclass', 'Alone', 'Child' ,'Female', 'Male', 'S', 'C']

data = featureFormat(new_test_dict, features_list,remove_NaN=True)
test_features = testtargetFeatureSplit(data)


# FEATURE SCALING
for i in range(len(features_list)-1):
    tmp =[]
    k=0
    for x in features:
        tmp.append(float(x[i]))
        
    tmp = MinMaxScaler().fit_transform(tmp)
    for x in features:
        x[i]=tmp[k]
        k = k + 1


# GETTING FEATURES AND LABELS TRAINING AND TESTING DATA
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# CHECKING BEST CLASSIFIER

# RANDOM FOREST
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
tree_para={'n_estimators':[70,80,90,100],'max_depth':[4,5,6,7,8,9,10],'min_samples_split':[2,3,4,5]}
clf = GridSearchCV(clf,tree_para)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)


# DECISION TREES
'''
clf=tree.DecisionTreeClassifier()
tree_para={'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],'min_samples_split':[2,3,4,5,8,10,12,15,20,25,30,35,40]}
clf = GridSearchCV(clf,tree_para)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
'''

# SUPPORT VECTOR MACHINE
'''
clf = SVC()
tree_para={'C':[1.0, 10.0, 100.0],'kernel':['linear', 'rbf']}
clf = GridSearchCV(clf,tree_para)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
'''

# GAUSSIAN NAIVE BAYAES
'''
clf = GaussianNB()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
'''

# K NEAREST NEIGHBOURS
'''
clf = KNeighborsClassifier()
tree_para={'n_neighbors':[3,4,5,6,7,8,9,10],'leaf_size':[10,20,30,40]}
clf = GridSearchCV(clf,tree_para)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
'''

# ACCURACY AND RECALL SCORE FOR RANDOM FOREST CLASSIFIER
print("Accuracy Score:")
print (accuracy_score(pred,labels_test))
print("Recall Score:")
print(recall_score(labels_test, pred))

