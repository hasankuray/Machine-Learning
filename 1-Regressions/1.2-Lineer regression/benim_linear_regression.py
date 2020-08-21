import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_1.csv", sep = ";")
#%%

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

plt.scatter(x,y,color="blue")

#%%

array = np.array([i for i in df.deneyim]).reshape(-1,1)
array_x = np.array(x)

y_head = linear_reg.predict(array_x)

plt.plot(array_x , y_head , color ="red")



