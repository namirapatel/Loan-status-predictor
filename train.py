import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import _pickle as cPickle



def data_preprocessing(raw_data):
    try:
        #print(data)
        data=raw_data.dropna()
        #print("1")
        data.replace({'Loan_Status':{'Y':1,'N':0}},inplace=True)
        #print("2")
        data.replace(to_replace='3+',value=4,inplace= True)
        #print("3")
        data.replace({'Gender':{'Male':1,'Female':0},
                    'Self_Employed':{'Yes':1,'No':0},
                    'Education':{'Graduate':1,'Not Graduate':0},
                            'Property_Area':{'Rural':0,'Urban':0,'Semiurban':2},
                    'Married':{'Yes':1,'No':0}},
                            inplace=True
                            )
        return data
    except Exception as e:
        print(e)

def split_data(data):
    X= data.drop(columns=['Loan_ID','Loan_Status'],axis=1)
    Y= data['Loan_Status']
    return X,Y

def train(X_train,Y_train):
    classifier=svm.SVC(kernel='linear')
    print(X_train)
    classifier.fit(np.array(X_train),np.array(Y_train))
    return classifier


def save_to_pc(classifier):
    with open('loan_predictor.pkl', 'wb') as fid:
        cPickle.dump(classifier, fid) 

data = pd.read_csv("data.csv")
preprocessed_data = data_preprocessing(data)
print("********************************")
print("Data Preprocessed")
print("********************************")

X,Y = split_data(preprocessed_data)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
classifier= train(X_train,Y_train)

save_to_pc(classifier)


