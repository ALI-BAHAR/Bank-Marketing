import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#Importind dataset
Bank=pd.read_csv(r"bank-additional.csv",delimiter=';')

#Replacing the "unknown" values with NaN#
Bank[Bank=="unknown"]=np.nan

#Number of columns#
Bank.shape[1]
#type of values#
print(Bank.dtypes)

print("Displaying coincide summery",Bank.info())
print("Statistial summary",Bank.describe())

#Cheking the number of NaN# 
print(Bank.isnull().sum()) 
#there are 1,230 NaN values , replacing with the other frequent values and drop the others#

Bank_copy=Bank.copy()
#Drop the null values#
Bank.dropna(subset=["housing"],inplace=True)
Bank.dropna(subset=["loan"],inplace=True)
Bank.dropna(subset=["job"],inplace= True)


#Fill the null Values with the values chich are the most#
Bank["marital"].fillna(Bank["marital"].mode , inplace=True)
Bank["education"].fillna("basic.4y",inplace=True)
Bank["default"].fillna("no",inplace=True)

print(Bank.isnull().sum())


#Exploratory Data Analysis#

#The result of the campaign#
plt.figure(figsize=(10,6))
plt.pie(Bank["y"].value_counts(),labels=["no","yes"],explode=(0,0.1),autopct="%.2f")
plt.title("Final outcome of the campaign")
plt.legend()
plt.show()

#Age Distribution#
plt.figure(figsize=(10,6))
plt.hist(Bank["age"],bins=20)
plt.ylabel("Number of client")
plt.xlabel("age")
plt.title("Customer Age Analysis")
plt.show()

#Job Distributio#
plt.figure(figsize=(10,6))
plt.hist(Bank['job'],bins=30,color = "r",width=.4)
plt.xticks(rotation=80)
plt.ylabel('Number of clients')
plt.title('Customer job Analysis')
plt.show()

#Education Distributio#
plt.figure(figsize=(10,6))
plt.hist(Bank['education'],bins=15)
plt.xticks(rotation=80)
plt.ylabel('Number of clients')
plt.title('Customer education Analysis')
plt.show()




 

