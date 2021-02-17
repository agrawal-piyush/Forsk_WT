import pandas as pd

import numpy as np


df = pd.read_csv ('balenced_reviews.csv')

df.isnull().any(axis=0) #return that values are null in reviewText and summary


df.dropna(inplace = True) #drop the null value rows from the dataframe


#now overwrite the dataframe with data except overall == 3 rows

df = df [df['overall']!=3]

df ['Postivity'] = np.where(df['overall']>3,1,0)

df.to_csv('finaldata.csv',index=False)


df = pd.read_csv('finaldata.csv')
#data cleaning

import nltk

nltk.download('stopwords')
#lib to remove stopwards
from nltk.corpus import stopwords
#lib to change word to first form called stemming
from nltk.stem.porter import PorterStemmer
#regular expression
import re

df['reviewText'][0]
#empty list
corpus=[]

for i in range(0,527319):
    #cleaning all extra thing 
    review = re.sub('[^a-zA-Z]',' ',df.iloc[i,1])
    
    review = review.lower()
    review = review.split()
    #removing stopwords
    review = [word for word in review if not  word in set(stopwords.words('english'))]
    
    #stemming
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    #changing back to string
    review = " ".join(review)
    #appending in list
    corpus.append(review)
    print(i)
    
# saving the features 

pd.DataFrame(corpus).to_csv("final_features.csv",index=False)    

from sklearn.model_selection import train_test_split

features_train,features_test,labels_train,labels_test=train_test_split(corpus, df['Postivity'],random_state=42)

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer().fit(features_train)

features_train_vectorized =  vect.transform(features_train)

from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression()

regressor.fit(features_train_vectorized,labels_train)

predictions = regressor.predict(vect.transform(features_test))

from sklearn.metrics import roc_auc_score,confusion_matrix
roc_auc_score(predictions, labels_test) #87%  

confusion_matrix(predictions, labels_test)