# simple linear regression     y = b0 + b1 * x

# multiple linear regression   y= b0 + b1 * x1 + b2 * x2..

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_multiple.csv", sep = ";")

x = df.iloc[: , [0,2]].values     # : her satır alınsın demek 0 ve 2. sütunlardan
y = df.maas.values.reshape(-1,1)

#%%

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0: " , multiple_linear_regression.intercept_)
print("b1,b2 " , multiple_linear_regression.coef_)

print(multiple_linear_regression.predict(np.array([[10,35],[5,35]])))
#%%
plt.scatter(x,y, color = "blue")
plt.show()

array= np.array(x)                                                              # benim yazdığım kısım
y_head = multiple_linear_regression.predict(array)

plt.plot(array , y_head , color = "red" ) 
plt.show()
#%%