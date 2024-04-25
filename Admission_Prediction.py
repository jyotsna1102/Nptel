#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics
#%matplotlib inline

#Extracting the csv files

df = pd.read_csv('Admission_Predict.csv')
print(df.head())

#Finding null values

print(df.isnull().sum())

#Describing the datasets

print(df.describe())

#Info of dataset

print(df.info())

#Check for correlation using the pearson method

print(df.corr(method='pearson'))

#PLOTS TO VISUALIZE THE DATASET

#plot GRE Score against Chance of Admit
plt.subplots(figsize=(20,4)) #the size of the plot can be changed to suit a desired size
sns.barplot(x="GRE Score",y="Chance of Admit ",data=df) #barplot with the x and y cordinates and source of data
plt.savefig('GRE_Score_plot.png') #to save the plot as a png file
#plt.show() #to show the plot after execution

#plot TOEFL Score against Chance of admit
plt.subplots(figsize=(25,5))
sns.barplot(x="TOEFL Score",y="Chance of Admit ",data=df)
plt.savefig('TOEFL_Score_plot.png')
#plt.show()

#plot University Rating against Chance of admit
plt.subplots(figsize=(20,4))
sns.barplot(x="University Rating",y="Chance of Admit ",data=df)
plt.savefig('University_Rating_plot.png')
#plt.show()

#plot SOP against Chance of Admit
plt.subplots(figsize=(15,5))
sns.barplot(x="SOP",y="Chance of Admit ",data=df)
plt.savefig('SOP_plot.png')
#plt.show()

#plot CGPA against Chance of Admit
plt.subplots(figsize=(15,4))
sns.barplot(x="CGPA",y="Chance of Admit ",data=df)
plt.savefig('CGPA_plot.png')
#plt.show()

#plot Research against Chance of Admit
plt.subplots(figsize=(15,5))
sns.barplot(x="Research",y="Chance of Admit ",data=df)
plt.savefig('Research_plot.png')
#plt.show()

#Splitting the dataset into dependent and independent features

X = df.iloc[:,1:8] #independent features (excluded the first column
#ie Serial No because chance of admit does not depend on it)
y = df["Chance of Admit "] #dependent variable
print(X.head())

#Splitting the X and y dataset into train and test sets
#80% of data for training and 20% for testing

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2,random_state = 1)

 #training the algorithm using linear regression

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

#make prediction using the test data

y_pred = regressor.predict(X_test)

#view predicted chance of admit 
pred_chance_of_admit = pd.DataFrame({ 'Predicted': y_pred})
#print(pred_chance_of_admit)

#compare actual chance of admit to predicted chance of admit
compare_chance_of_admit = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(compare_chance_of_admit)

#using bar plot to compare actual chance of admit and predicted chance of admit
df = compare_chance_of_admit.head(25)
df.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.savefig('actual_and_predicted_chance_of_admit.png')
#plt.show()

#show the accuracy of linear regression model used
accuracy = regressor.score(X_test,y_test)
print(accuracy*100,'%')
