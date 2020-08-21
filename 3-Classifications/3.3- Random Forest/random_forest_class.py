import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("kanser.csv")
data.drop(["id","Unnamed: 32"] , axis = 1 , inplace =  True)

data.diagnosis = [1 if i == "M" else 0 for i in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis =1)

#%%  normalization

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

#%%   Test

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.15, random_state=42)

#%%   decision tree

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()                                                   #Decision Tree
dt.fit(x_train , y_train)

print("score" , dt.score(x_test , y_test))

#%% random forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()                                                   #Random Forest 
rf.fit(x_train , y_train)
print("random forest result:" , rf.score(x_test , y_test))