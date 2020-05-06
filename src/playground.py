import pandas as pd
import numpy as np

df = pd.read_csv('ecommerce-dataset/events.csv')

df1=df[(True-df['event'].isin(['addtocart','transaction']))]
print(df1.shape)