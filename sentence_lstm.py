'''
Written by Austin Walters
Last Edit: January 2, 2018 
For use on austingwalters.com

An LSTM based RNN to classify
of the common sentance types:
Question, Statement, Command, Exclamation 
'''

from __future__ import print_function

import numpy as np
import keras

from sentence_types import load_encoded_data

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer


max_words = 10000
maxlen     = 500
embedding_dims = 150
batch_size = 150
epochs     = 3

x_train, x_test, y_train, y_test = load_encoded_data(data_split=0.8,
                                                     embedding_name="data/default",
                                                     pos_tags=True)

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


print('Constructing model!')
model = Sequential()

model.add(Embedding(max_words, embedding_dims))
model.add(LSTM(embedding_dims, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

print('Training... Grab a coffee')
model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test,batch_size=batch_size)

print('Test accuracy:', score[1])
