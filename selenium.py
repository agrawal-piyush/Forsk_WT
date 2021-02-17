import requests
import pandas as pd
from selenium import webdriver

from time import sleep

url = "https://www.etsy.com/in-en/listing/766705964/opal-ear-crawler-opal-ear-climber-gold?ga_order=most_relevant&ga_search_type=all&ga_view_type=gallery&ga_search_query=&ref=sc_gallery-1-1&plkey=f5df4b09f5fc78e04232e92645451806da2d293b%3A766705964&pro=1&col=1"
df = pd.DataFrame()
browser = webdriver.Chrome("c:/chromedriver.exe")
l=[]
browser.get(url)
j=6
for i in range(26):
        
        st = "//*[@id='reviews']/div[2]/nav/ul/li[position()=last()]/a"
        for j in range(4):
            try:
                review = browser.find_element_by_id("review-preview-toggle-"+str(j))
                l.append(review.text.strip())
                sleep(1)
            except:
                    continue
        sleep(1)
        browser.find_element_by_xpath(st).click()
        
#browser.quit()

df['reviews'] = l

df.to_csv('etsy_open_ear_crawler_reviews.csv',index = False)


import sqlite3 as sql

conn = sql.connect('reviews.db')

df.to_sql('reviews_table', conn)

#loading and fetching data

conn = sql.connect('reviews.db')
cursor = conn.cursor()

cursor.execute("SELECT * from reviews_table")

for record in cursor:
    print(record)
    
#adding positivity column

df.dropna(inplace = True)
import pickle
positivity=[]
model = pickle.load(open('pickle_model.pkl','rb'))
vocab = pickle.load(open('features.pkl','rb'))
from sklearn.feature_extraction.text import TfidfTransformer , TfidfVectorizer
transformer = TfidfTransformer()
loaded_vec = TfidfVectorizer(decode_error="replace",vocabulary=vocab)

for review in df['reviews']:
    review = transformer.fit_transform(loaded_vec.fit_transform([review]))
    positivity.append(model.predict(review)[0])  
    
df['Positivity'] = positivity

    
df.to_csv('Scrappedreviews.csv',index = False)
