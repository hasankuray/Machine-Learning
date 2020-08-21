# tree den tek farkı 100 tane tree yi bir arada kullanabilmemiz.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("random_forest_regression.csv", sep= ";")

x = df.mesafe.values.reshape(-1,1)
y = df.para.values.reshape(-1,1)

#%%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100 , random_state= 42) # 100 tane tree kullanıldı.
rf.fit(x,y)

print("7.8 seviyesindeki fiyat: ", rf.predict([[7.8]]))

#%%
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

plt.scatter(x,y , color = "red")
plt.plot(x_,y_head)
plt.show()
