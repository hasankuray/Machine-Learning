import pandas as pd
import numpy as np
#%%

data = pd.read_csv("kanser.csv")
data.drop(["id", "Unnamed: 32"] , axis = 1 ,inplace =True)

#%% 

data.diagnosis = [1 if i == "M" else 0 for i in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis = 1)

#%% normalization 

x = (x_data - np.min(x_data)) / (np.max(x_data)- np.min(x_data))

#%% train test

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.15 , random_state = 42)

#%%  random forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100 , random_state = 1 )
rf.fit(x_train,y_train)
print("random forest result", rf.score(x_test, y_test))

#%%confusion matrix

y_pred = rf.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix   
cm = confusion_matrix(y_true , y_pred)

#%% Görselleştirme 

import seaborn as sns
import matplotlib.pyplot as plt

f , ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm,annot = True , linewidths = 1 , linecolor = "red", fmt = ".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()










