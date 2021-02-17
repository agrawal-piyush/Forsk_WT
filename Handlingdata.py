import pandas as pd

df_reader = pd.read_json('Clothing_shoes_and_Jewelry.json',lines=True,chunksize=1000000)

counter = 1

for chunk in df_reader:
    df = pd.DataFrame(chunk[['overall','reviewText','summary']])
    df1 = df[df['overall']==5].sample(4000)
    df2 = df[df['overall']==4].sample(4000)
    df3 = df[df['overall']==3].sample(8000)
    df4 = df[df['overall']==2].sample(4000)
    df5 = df[df['overall']==1].sample(4000)
    
    df6 = pd.concat([df1,df2,df3,df4,df5],axis=0,ignore_index=True)
    
    df6.to_csv(str(counter)+".csv",index=False)
    
    df= None
    counter+=1

