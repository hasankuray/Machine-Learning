from sklearn.datasets import load_iris
import pandas as pd

#%%  Dataset yapıyoruz

iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data , columns = feature_names)
df["sinif"] = y

x= data

#%%  4 featureli datayı 2 feature ye dönüştürüyoruz

from sklearn.decomposition import PCA
pca = PCA(n_components= 2 , whiten = True)  #whiten =true  normalize et demek , 4 featureden 2 featureye düşür diyoruz
pca.fit(x)
x_pca = pca.transform(x)

print("variance ratio", pca.explained_variance_ratio_)

print("sum" , sum(pca.explained_variance_ratio_))

#%%  Görüntüleme

df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1]

color = ["red", "green", "blue"]

import matplotlib.pyplot as plt
for each in range(3):
    plt.scatter(df.p1[df.sinif == each ] , df.p2[df.sinif == each] , color = color[each], label = iris.target_names[each])
    
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()
