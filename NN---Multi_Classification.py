#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
import seaborn as sns
sns.set()
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

i = int(sys.argv[1])
dataset = pd.read_csv('wines.csv')

y = dataset['Class']

sns.scatterplot(x= dataset['Alcohol'], y = y)


y = pd.get_dummies(y)            #### We can not drop one category of y as it is our Predicted

print(y.head())

X = dataset.drop('Class', axis = 1)

print(X.head())

print(X.info())


####### Sequential Model Creation ##########

model = Sequential()

model.add(Dense(units = 12, input_shape = (13,), activation = 'relu', kernel_initializer = 'he_normal'))
model.add(Dense(units = 16, activation = 'relu'))
model.add(Dense(units = 8, activation = 'relu'))
model.add(Dense(units = 10, activation = 'relu'))
if i == 1:
    model.add(Dense(units = 8, activation = 'relu'))

##### Output Layer ##########
model.add(Dense(units = 3, activation = 'softmax'))

print(model.summary())

model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(X,y, epochs = 50)

model.save("wine_model.h5")
accuracy = int(model.history.history['accuracy'][-1]*100)

print(accuracy)

