import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% dataset

x1 = np.random.normal(25,5,1000) # ortalaması 25 olacak , 1000 tane sayı , sayıların %66 sı 25-5 ve 25+5 arasında.
y1 = np.random.normal(25,5,1000)

x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(65,5,1000)

x3 = np.random.normal(55,5,1000)
y3 = np.random.normal(15,5,1000)

x = np.concatenate((x1,x2,x2) , axis =0)  # x1 , x2 , x3 ü axis 0 da yukardan aşağı birleştirdik.
y = np.concatenate((y1,y2,y3) , axis =0)

dictionary = {"x" : x , "y":y}

data = pd.DataFrame(dictionary)
data.describe()   # datanın özelliklerini gösterir

#%%  kmeans algoritması böyle görecek 

#plt.scatter(x1,y1,color ="black")
#plt.scatter(x2,y2,color ="black")
#plt.scatter(x3,y3,color ="black")
#plt.show()

#%%  kmeans de k yi bulma

from sklearn.cluster import KMeans
wcss = []

for k in range(1,15):
    kmeans = KMeans(n_clusters= k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)  # matematiksel hesaplama yapıyo
    
plt.plot(range(1,15),wcss)
plt.xlabel("k sayısının değeri")
plt.ylabel("wcss")
plt.show()

#%%  k = 3 için modelimiz 

kmeans2 = KMeans(n_clusters= 3)
clusters = kmeans2.fit_predict(data)

data["label"] = clusters

plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color ="red")
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color ="blue")
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color ="green")
plt.show()





















