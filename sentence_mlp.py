'''
Written by Austin Walters 
Last Edit: January 2, 2018
For use on austingwalters.com

Trains and evaluate a simple MLP
Intended to classify a sentence as one
of the common sentance types:
Question, Statement, Command, Exclamation                                                    

Heavily Inspired by Keras Examples: 
https://github.com/keras-team/keras
'''

from __future__ import print_function

import numpy as np
import keras

from sentence_types import load_encoded_data

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

max_words = 10000
batch_size = 256
epochs = 3

x_train, x_test, y_train, y_test = load_encoded_data(data_split=0.8)

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Constructing model!')

model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs, verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

print('Test accuracy:', score[1])
