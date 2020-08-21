import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%  ders1 Read csv

data = pd.read_csv("kanser.csv")
print(data.info())
data.drop(["Unnamed: 32", "id"], axis =1 , inplace=True) # axis 0 ise satır 1 ise sütun // inplace true ise dataya kaydet
data.diagnosis = [1 if each =="M" else 0   for each in data.diagnosis]  # list comprension

y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis = 1)

#normalization -- (x - min(x))/(max(x) - min(x))
# birisi 250 diyelim diğer değişken ise 0.001 falan ise küçük olanı ihmal ettirebildiği için normalize edildi
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values   # bütün sayıları 0 la 1 arasında yapar

#%%  ders 2  Dataset Train-Test Split

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)
#  %80     %20       %80       %20     burda %20 sini test olarak saklıyoruz. Diğerlerini kullanıcaz.

x_train = x_train.T
x_test = x_test.T     # index numaraları satır olarak yazıyordu. Sütunlarda ise değişkenler var idi    
y_train = y_train.T   # bunların yerini değiştirdik.
y_test = y_test.T

#%%  Ders 3   weight i bias ve sigmoid tanımlandı

def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w,b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

#%%   Ders 4  forward ve backward
    
def forward_backward_propagation(w,b,x_train,y_train):
    # forward
    z = np.dot(w.T,x_train) + b  # w ters çevirdik çünkü matris çarpılması lazım
    y_head = sigmoid(z)
    loss = -y_train*nplog(y_head)- (1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    # backward
    derivative_weight = (np.dot(x_train,((y_head - y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/ x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight , "derivative_bias": derivative_bias}
    
    return cost,gradients

#%% Ders 5    Updating(learning) parameters
    
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

#%%  # prediction
def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
   
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

# %% logistic_regression
    
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    
    dimension =  x_train.shape[0]  
    w,b = initialize_weights_and_bias(dimension)
    
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 300)    


#%% sklearn with LR

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))






















