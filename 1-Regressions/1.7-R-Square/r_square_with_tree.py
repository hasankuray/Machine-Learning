# tree den tek farkÄ± 100 tane tree yi bir arada kullanabilmemiz.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("random_forest_regression.csv", sep= ";")

x = df.mesafe.values.reshape(-1,1)
y = df.para.values.reshape(-1,1)

#%%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100 , random_state= 42) # 100 tane tree kullanýldý.
rf.fit(x,y)

y_head = rf.predict(x)

#%%  r-square

from sklearn.metrics import r2_score

print("r2_score: ",r2_score(y,y_head))