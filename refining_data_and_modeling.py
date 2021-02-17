##removing neutral situation from data , we will consider the positive and negative part 
#assumming 1 and 2 as negative 
#assuming 4 and 5 as positive
# 3 would be neutral so we remove it 
#in this file we also remove null values


import pandas as pd

import numpy as np

df = pd.read_csv ('balenced_reviews.csv')

#EDA Activities

df.shape #return the shape (792000,3)

df['overall'].value_counts() #return the count of ratings 

#checking if any null value present

df.isnull().any(axis=0) #return that values are null in reviewText and summary

df.isnull().any(axis=1)

df[df.isnull().any(axis=1)].tail() #return the null value at end of files

#handling missing data

df.dropna(inplace = True) #drop the null value rows from the dataframe

#lets remove neutral that is overall == 3

df['overall'] == 3   #returns the df of the overall ==3

#now overwrite the dataframe with data except overall == 3 rows

df = df [df['overall']!=3]

#lets check the count
df.shape 

# now as per assumotion features is our reviews but label is not present 
#we have four possibilities 1, 2 , 4 and 5
#we consideres 1 and 2 as negative & 4 and 5 as positive
# lets make a new column name positivity which consist of 0 and 1 
#0 declare the negative sentiment and 1 declares the positive one
# we concat both 1 and 2 as 0 &  4 and 5 as 1

df ['Postivity'] = np.where(df['overall']>3,1,0)

# data is now ready features = reviews and labels = positivity

#now we are going to extract our features and labels to train our future model

from sklearn.model_selection import train_test_split

features_train,features_test,labels_train,labels_test =train_test_split(df['reviewText'],df['Postivity'],random_state=42)

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer().fit(features_train) #count the words frequency uniquely

len(vect.get_feature_names()) #there are 66241 unique words in the reviewsText
#these are the names of the unique words treated as features
vect.get_feature_names()[0:10]

features_train_vectorized = vect.transform(features_train) # transform our features_train in the form of a data having unique words with the count

features_train_vectorized[:10].toarray() #to visualize data , without slicing it will give me a memory error , due to large count


#model  version 1

#classifier model -kNN , Logistic Regression, SVM,NaiveBayse

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(features_train_vectorized,labels_train)


from sklearn.metrics import roc_auc_score,confusion_matrix

predictions = model.predict(vect.transform(features_test))

roc_auc_score(labels_test,predictions)      #89%

confusion_matrix(labels_test, predictions)

