import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("tree_regression.csv", sep = ";")

x = df.mesafe.values.reshape(-1,1)   # bu x değerini az sonra değiştircez
y = df.para.values.reshape(-1,1)

#%% decision tree regression  başladı

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)
# x ekseninde olan 1 den 10 a kadar olan değerleri olan değerleri 
#aralarında 0.01 olacak şekilde ayırdık.
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1) 
y_head = tree_reg.predict(x_)       
# %% Görsel

plt.scatter(x,y, color ="red")
plt.plot(x_,y_head, color = "green")
plt.xlabel("mesafe")
plt.ylabel("para")
plt.show()




