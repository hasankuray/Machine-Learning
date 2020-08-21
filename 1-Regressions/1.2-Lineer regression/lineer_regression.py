import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("dataset_1.csv", sep = ";")

plt.scatter(df.deneyim , df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

# y = b0 + b1*x  -->  maas = b0 + b1 * deneyim
# b0 = çizginin y yi kestiği nokta(sabit)
# b1 = çizginin eğimi
#%%

# MSE = (sum(residual ^2) )/n
# residual = y - y_head        n = nokta sayısı

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)


b0 = linear_reg.predict([[0]])

b1 = linear_reg.coef_

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11]).reshape(-1,1)  # array ı deneyimden bakarak aynısını yazdık.
array_x = np.array([i for i in df.deneyim]).reshape(-1,1)    # array i deneyimden çekerek aldık.

plt.scatter(x,y)
plt.show()

y_head = linear_reg.predict(array)

plt.plot(array ,y_head , color = "red")

linear_reg.predict([[5]])  # 100 yıl deneyimi olan işçi
