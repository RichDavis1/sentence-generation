import sys
import tensorflow as tf
import re
import numpy
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Reshape
from typing import List

tf.test.is_gpu_available()

from lstmWordTrain import LstmWordTrain 

#Training the Model
model = LstmWordTrain('grimmsFairyTalesSmall.txt', sequence_size = 25, epochs = 100, batch_size = 100, output_file = 'model2')
model.train()

#Predicting word sequences
modelInstance = LstmWordTrain('grimmsFairyTalesSmall.txt', sequence_size = 25, epochs = 100, batch_size = 100, output_file = 'model2')
raw_text = modelInstance.rawText()
modelInstance.prepCorpus(raw_text)
modelInstance.prepSamples()

filename = "load your saved model here" 
#filename = "model.hdf5"
model = load_model(filename)

start = numpy.random.randint(0, len(modelInstance.samples))
pattern = modelInstance.samples[start]
print(pattern)
print ("Seed:")
print ("\"", ' '.join([modelInstance.int_to_char[value] for value in pattern]), "\"")

for i in range(20):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(modelInstance.n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = modelInstance.int_to_char[index]
    seq_in = [modelInstance.int_to_char[value] for value in pattern]
    sys.stdout.write(result + ' ')
    pattern.append(index)
    pattern = pattern[1:len(pattern)]