#model version 3
# we are using #TF_IDF (term frequency-inverse document frequency) for more relavent and efficient model

import pandas as pd
import numpy as np

data = pd.read_csv('finaldata.csv')

from sklearn.model_selection import train_test_split

features_train,features_test,labels_train,labels_test=train_test_split(data['reviewText'], data['Postivity'],random_state=42)


from sklearn.feature_extraction.text import TfidfVectorizer

vect =TfidfVectorizer(min_df=5).fit(features_train) 

features_train_vectorized = vect.transform(features_train)

from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression()

regressor.fit(features_train_vectorized,labels_train)

predictions = regressor.predict(vect.transform(features_test))

from sklearn.metrics import roc_auc_score,confusion_matrix
roc_auc_score(predictions, labels_test)  #90% increased by 1 % 

confusion_matrix(predictions, labels_test)

#saving the model 

import pickle

with open('pickle_model.pkl','wb') as file:
    pickle.dump(regressor, file)

with open('features.pkl','wb') as file:
    pickle.dump(vect.vocabulary_,file)


import joblib

pkl_file = "joblib_model.pkl"

with open('joblib_model.pkl','wb') as file:
    joblib.dump(regressor,file)
    
    
'''
to open 
with open ('joblib_model.pkl','rb') as file:
    joblib_model = joblib.load(file)
    


predict = joblib_model.predict(vect.transform(features_test))

roc_auc_score(predict, labels_test)  #90% increased by 1 % 

confusion_matrix(predict, labels_test)

'''
