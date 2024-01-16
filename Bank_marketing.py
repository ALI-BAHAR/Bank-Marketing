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

#create a new column#
Bank["Potential_customers"]=Bank["age"].apply(lambda x:"Yes" if x>25 and x<35 else "No")

#add a column with marital status and age between 25 and 35 , this group shows the most tendency #
Bank['condition_column'] = np.where((Bank['marital'] == 'married') & (Bank['age'] >= 25) & (Bank['age'] <= 35), 'Condition Met', 'Condition Not Met')

#Affection of months on the campaign#
Bank_group=Bank.groupby(Bank["y"])

lst=[]
for i in Bank_group["month"].value_counts()["yes"].index:
    lst.append(Bank_group["month"].value_counts()["yes"][i]/(Bank_group["month"].value_counts()["yes"][i]+Bank_group["month"].value_counts()["no"][i])*100)
plt.bar(Bank_group["month"].value_counts()["yes"].index,lst)
plt.ylabel("percentage of clients")
plt.title("Months affection on the campaign")
plt.show()

#Affection of education on the campaign#
lst=[]
for i in Bank_group["education"].value_counts()["yes"].index:
    lst.append(Bank_group["education"].value_counts()["yes"][i]/(Bank_group["education"].value_counts()["yes"][i] + Bank_group["education"].value_counts()["no"][i])*100)

plt.bar(Bank_group["education"].value_counts()["yes"].index,lst)
plt.xticks(rotation=90)
plt.ylabel("percentage of clients")
plt.title("Education affection on the campaign")
plt.show()

#Affection of job on the campaign#
lst=[]
for i in Bank_group["job"].value_counts()["yes"].index:
    lst.append(Bank_group["job"].value_counts()["yes"][i]/(Bank_group["job"].value_counts()["yes"][i] + Bank_group["job"].value_counts()["no"][i])*100)

plt.bar(Bank_group["job"].value_counts()["yes"].index,lst)
plt.xticks(rotation=90)
plt.ylabel("percentage of clients")
plt.title("job affection on the campaign")
plt.show()

plt.bar(Bank_group["poutcome"].value_counts()["no"].index,Bank_group["poutcome"].value_counts()["no"].values,alpha=0.9,color="orange" ,width=0.4,label="no")
plt.bar(Bank_group["poutcome"].value_counts()["yes"].index,Bank_group["poutcome"].value_counts()["yes"].values,alpha=0.6,color="green",width=0.4,label="yes")
plt.legend()
plt.show()

#preprocessing for ML#

Bank_copy2=Bank.copy()

Bank_copy2=pd.concat([Bank_copy2,pd.get_dummies(Bank["month"])],axis=1)
Bank_copy2.drop(Bank["month"],axis=1,inplace = True)

Bank_copy2=pd.concat([Bank_copy2,pd.get_dummies(Bank["poutcome"])],axis=1)
Bank_copy2.drop(Bank["poutcome"],axis=1,inplace=True)
                     
 




 

