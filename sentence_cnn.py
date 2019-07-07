'''
Written by Austin Walters
Last Edit: January 2, 2019
For use on austingwalters.com

A CNN to classify a sentence as one 
of the common sentance types:
Question, Statement, Command, Exclamation

Heavily Inspired by Keras Examples: 
https://github.com/keras-team/keras
'''

from __future__ import print_function

import numpy as np
import keras

from sentence_types import load_encoded_data

from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D

from keras.preprocessing.text import Tokenizer

max_words = 10000
maxlen = 500
batch_size = 64
embedding_dims = 50
filters = 250
kernel_size = 5
hidden_dims = 150
epochs = 2

x_train, x_test, y_train, y_test = load_encoded_data(data_split=0.8,
                                                     embedding_name="data/default",
                                                     pos_tags=True)

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Constructing model!')

model = Sequential()

model.add(Embedding(max_words, embedding_dims,
                    input_length=maxlen))
    
model.add(Dropout(0.2))
    
model.add(Conv1D(filters, kernel_size, padding='valid',
                 activation='relu', strides=1))

model.add(GlobalMaxPooling1D())

model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

print('Test accuracy:', score[1])
