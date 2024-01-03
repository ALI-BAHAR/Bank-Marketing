import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# importind dataset
Bank=pd.read_csv(r"bank-additional.csv",delimiter=';')

#replacing the "unknown" values with NaN
Bank[Bank=="unknown"]= np.nan

#number of columns#
Bank.shape[1]
#type of values#
print(Bank.dtypes)

print("Displaying coincise summery" , Bank.info)
print("Statistial sumery" , Bank.describe)

#heking the number of NaN# 
print(Bank.isnull().sum()) 
#there are 1,230 NaN values , replacing with the other frequent values#

Bank_copy=Bank.copy()
#Drop the null values#
Bank.dropna(subset=["housing"],inplace=True)
Bank.dropna(subset=["loan"],inplace=True)
Bank.dropna(subset=["job"],inplace= True)


#Fill the null Values with the values chich are the most"
Bank["marital"].fillna(Bank["marital"].mode , inplace=True)
Bank["education"].fillna("basic.4y",inplace=True)
Bank["default"].fillna("no",inplace=True)

print(Bank.isnull().sum())

 

