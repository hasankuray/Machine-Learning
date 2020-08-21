# Bu hierarchical metodu bütün datasetlerini önce clusteringe çevirir.
# Sonra en yakın 2 clustering i alıp tek clustering yapar ve bu adım sürekli tekrarlanır.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% dataset

x1 = np.random.normal(25,5,100) # ortalaması 25 olacak , 1000 tane sayı , sayıların %66 sı 25-5 ve 25+5 arasında.
y1 = np.random.normal(25,5,100)

x2 = np.random.normal(55,5,100)
y2 = np.random.normal(65,5,100)

x3 = np.random.normal(55,5,100)
y3 = np.random.normal(15,5,100)

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

# DATASET BİTTİ

#%%  dendogram

from scipy.cluster.hierarchy import linkage , dendrogram

merg = linkage(data, method="ward")
dendrogram(merg , leaf_rotation = 90)    # çizgileri 90 derece yaptık
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()

# EN UZUN ÇİZGİ KIRMIZI ÇİZGİ OLDUĞUNDAN ORDAN YATAY BİR ÇİZGİ ÇEKİYORUZ VE
# ÇİZGİYİ KESEN NOKTA KADAR CLUSTER OLMASI EN MANTIKLISI OLUYOR. 3

#%%  Hierarchical Clustering

from sklearn.cluster import AgglomerativeClustering

hierarchical_clustering = AgglomerativeClustering(n_clusters= 3 , affinity= "euclidean", linkage = "ward")
cluster = hierarchical_clustering.fit_predict(data)

data["label"] = cluster




