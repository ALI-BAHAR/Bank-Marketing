import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import streamlit as st

st.title("Bank Marketing")
st.write("The information deals with phone-based marketing campaigns carried out by a Portuguese bank. The classification goal is to predict whether the client will subscribe to a term deposit or not.")

#Importind dataset
Bank=pd.read_csv(r"bank-additional.csv",delimiter=';')
st.sidebar.subheader("The original data: ")
st.sidebar.write(Bank)

#Replacing the "unknown" values with NaN#
Bank[Bank=="unknown"]=np.nan
eda_check=st.checkbox("Data Eda : ")
if eda_check:
    # check the data
    st.write("Data Statistics: ")
    st.write(Bank.describe())

#Cheking the number of NaN# 
    st.write("Null values: ")
    st.table(Bank.isnull().sum())
    st.write("Almost 31% of the data is null, so we can not drop them. We have to fill them with the most frequent value of each column.")
#there are 1,230 NaN values , replacing with the other frequent values and drop the others#

Bank_copy=Bank.copy()
#Drop the null values#
Bank.dropna(subset=["housing"],inplace=True)
Bank.dropna(subset=["loan"],inplace=True)
Bank.dropna(subset="job",inplace= True)


#Fill the null Values with the values chich are the most#
Bank["marital"].fillna(Bank["marital"].mode, inplace=True)
Bank["education"].fillna("basic.4y",inplace=True)
Bank["default"].fillna("no",inplace=True)

print(Bank.isnull().sum())


#Exploratory Data Analysis#
if eda_check:

    Chart=plt.figure(figsize=(6,4))
    plt.pie(Bank["y"].value_counts(),labels=["no","yes"],explode=(0,0.1),autopct="%.2f")
    plt.title("Final outcome of the campaign")
    st.write(Chart)

    Chart,ax=plt.subplots(1,2,figsize=(15,6))


    ax[0].hist(Bank["age"],bins=20)
    ax[0].set_ylabel("Number of client")
    ax[0].set_xlabel("age")
    ax[0].set_title("Customer Age Analysis")

    ax[1].hist(Bank['job'],bins=20,color = "r",width=.4)
    ax[1].set_xticks(Bank["job"].unique())
    ax[1].set_xticklabels(ax[1].get_xticklabels(),rotation=90)
    ax[1].set_ylabel('Number of clients')
    ax[1].set_title('Customer job Analysis')
    st.write(Chart)

    Chart=plt.figure(figsize=(10,6))
    plt.hist(Bank['education'],bins=15)
    plt.xticks(rotation=80)
    plt.ylabel('Number of clients')
    plt.title('Customer education Analysis')
    st.write(Chart)

    st.write("Now we want to see the effect of each feature on the final outcome of the campaign:")
    Bank_group=Bank.groupby(Bank["y"])

    lst=[]
    for i in Bank_group["month"].value_counts()["yes"].index:
        lst.append(Bank_group["month"].value_counts()["yes"][i]/(Bank_group["month"].value_counts()["no"][i]+Bank_group["month"].value_counts()["yes"][i])*100)
    Chart=plt.figure(figsize=(10,6))
    plt.bar(Bank_group["month"].value_counts()["yes"].index,lst)
    plt.ylabel("percentage of clients")
    plt.title("Months affection on the campaign")
    st.write(Chart)

    lst=[]
    for i in Bank_group["education"].value_counts()["yes"].index:
        lst.append(Bank_group["education"].value_counts()["yes"][i]/(Bank_group["education"].value_counts()["no"][i] + Bank_group["education"].value_counts()["yes"][i])*100)
    Chart=plt.figure(figsize=(10,6))
    plt.bar(Bank_group["education"].value_counts()["yes"].index,lst)
    plt.xticks(rotation=90)
    plt.ylabel("percentage of clients")
    plt.title("Education affection on the campaign")
    st.write(Chart)

    lst=[]
    for i in Bank_group["job"].value_counts()["yes"].index:
        lst.append(Bank_group["job"].value_counts()["yes"][i]/(Bank_group["job"].value_counts()["no"][i] + Bank_group["job"].value_counts()["yes"][i])*100)
    Chart=plt.figure(figsize=(10,6))
    plt.bar(Bank_group["job"].value_counts()["yes"].index,lst)
    plt.xticks(rotation=90)
    plt.ylabel("percentage of clients")
    plt.title("job affection on the campaign")
    st.write(Chart)

    Chart=plt.figure(figsize=(10,6))
    plt.bar(Bank_group["poutcome"].value_counts()["no"].index,Bank_group["poutcome"].value_counts()["no"].values,alpha=0.9,color="orange" ,width=0.4,label="no")
    plt.bar(Bank_group["poutcome"].value_counts()["yes"].index,Bank_group["poutcome"].value_counts()["yes"].values,alpha=0.6,color="green",width=0.4,label="yes")
    plt.xticks(Bank_group["poutcome"].value_counts()["no"].index,Bank_group["poutcome"].value_counts()["no"].index,rotation=90)
    plt.ylabel("Number of clients ")
    plt.title("previous campaign outcome")
    plt.legend()
    st.write(Chart)

#preprocessing for ML#

Bank_copy2=Bank.copy()

Bank_copy2=pd.concat([Bank_copy2,pd.get_dummies(Bank["month"])],axis=1)
Bank_copy2.drop(["month"],axis=1,inplace = True)

Bank_copy2=pd.concat([Bank_copy2,pd.get_dummies(Bank["poutcome"])],axis=1)
Bank_copy2.drop(["poutcome"],axis=1,inplace=True)
                     
 
Bank_copy2["y"].replace({"yes":1,"no":0},inplace=True)

Bank_copy2.drop(["duration"],axis=1,inplace=True)
Bank_copy2.drop(["contact"],axis=1,inplace=True)

Bank_copy2["default"].replace({"yes":1,"no":0},inplace=True)
Bank_copy2["housing"].replace({"yes":1,"no":0},inplace=True)
Bank_copy2["loan"].replace({"yes":1,"no":0},inplace=True)
Bank_copy2.drop(["job","marital","education","day_of_week"],axis=1,inplace=True)

#corrolation#
if st.checkbox("corrolation: "):
    st.write("now you can see the corrolation between the features: ")
    if st.button("corrolation"):
        corrolation=Bank_copy2.corr()
        st.write("corrolation between y and the other columns:",corrolation["y"].sort_values(ascending=False))

        import seaborn as sns
        Chart=plt.figure(figsize=(20,20))
        sns.heatmap(corrolation, annot=True, cmap='coolwarm')
        plt.title("corrolation between values")
        plt.show()
        st.write(Chart)

#Modeling part
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier

if st.checkbox("Modeling part:"):
    st.write("Now we want to train a model to predict the outcome of the campaign: ")
    st.write("First we split the data into train and test. Then we train a KNN model and predict the outcome of the test data.")
    st.write("Finally we calculate the accuracy of the model:")

    #Split data
    x_train,x_test,y_train,y_test=train_test_split(Bank_copy2.drop(["y"],axis=1).values,Bank_copy2["y"].values,test_size=0.2,random_state=42)
    #KNN Model
    knn=KNeighborsClassifier(n_neighbors=6)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    st.write("Accuracy:",accuracy_score(y_test,y_pred))
    st.write("Confusion Matrix:",confusion_matrix(y_test,y_pred))
    




 
