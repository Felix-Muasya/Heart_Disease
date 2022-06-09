import pandas as pd
import numpy as np
from sklearn import model_selection
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers


# opening the dataset n turning it into a numpy array
data = pd.read_csv("heart.csv")
X = np.array(data.drop(['target'], 1))
y = np.array(data['target'])

# some EDA to improve the model accuracy
mean = X.mean(axis=0)
X -= mean
std = X.std(axis    =0)
X /= std

# dividing into test and train and turning it into a binary classification model
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42, test_size=0.25)

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)

Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()

Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1


def create_binary_model():
    model = Sequential()

    # input layer,relu activation 13 nodes,
    model.add(Dense(16, input_dim=13, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001),
                    activation='relu'))
    model.add(Dropout(0.25))

    # 1st hidden layer, relu activation,  16 nodes
    model.add(Dense(16, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))

    # 2nd hidden layer, relu activation, 16 nodes
    model.add(Dense(16, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))

    # output layer, sigmoid activation, 1 nodes
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    # optimizer
    adam = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


binary_model = create_binary_model()
history = binary_model.fit(X_train, Y_train_binary, validation_data=(X_test, Y_test_binary), epochs=50, batch_size=10)

