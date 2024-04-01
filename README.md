# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
 Build a Recurrent Neural Network (RNN) model to forecast Google's stock price using historical stock price data from the 'Trainset.csv' file. Subsequently, the model's performance will be evaluated using a different dataset, 'testset.csv'. Both datasets contain various columns, but for this task, only the 'Open' column (representing the opening price of the stock) will be utilized for prediction purposes.

## Design Steps

### step1: Reading and preprocessing the training dataset ('Trainset.csv').
### step2: Building and training the RNN model using the 'Open' prices from the training dataset.
### step3: Reading and preprocessing the test dataset ('testset.csv').
### step4: Using the trained model to make predictions on the test dataset.
### step5: Evaluating the accuracy of the model's predictions against the actual stock prices from the test dataset.



## Program
#### Name:Rakshitha devi J
#### Register Number:212221230082
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

dataset_train = pd.read_csv('trainset.csv')


dataset_train.columns

dataset_train.head()

train_set = dataset_train.iloc[:,1:2].values

type(train_set)

train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

training_set_scaled.shape

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))


X_train.shape

length = 60
n_features = 1

model = Sequential()
model.add(layers.SimpleRNN(45,input_shape=(length,n_features)))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mse')

print("Name:Rakshitha Devi J Register Number:212221230082")
model.summary()

model.fit(X_train1,y_train,epochs=100, batch_size=32)

dataset_test = pd.read_csv('testset.csv')

dataset_test.head()

test_set = dataset_test.iloc[:,1:2].values

test_set.shape

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

X_test.shape

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

print("Name:Rakshitha Devi J Register Number:212221230082")
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error as mse
mse = mse(y_test,predicted_stock_price)
print("Mean Squared Error = ",mse)
```

## Output

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/etjabajasphin/rnn-stock-price-prediction/assets/94165326/cbd2abd7-9c45-4909-8094-2e345fdd45ac)


### Mean Square Error

![image](https://github.com/etjabajasphin/rnn-stock-price-prediction/assets/94165326/7ed916ab-f91a-4eb5-91af-7dbccac5132a)


## Result
Thus a Recurrent Neural Network model for stock price prediction is executed successfully.
