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

class LstmWordTrain:

    def __init__(self, file: str, sequence_size: int, epochs: int, batch_size: int, output_file: str):
        self.file = file
        self.sequence_size = sequence_size or 25
        self.epochs = epochs or 50
        self.batch_size = batch_size or 100 
        self.output_file = output_file or "model"

    def rawText(self):
        raw_text = open(self.file, 'r', encoding='utf-8').read()
        raw_text = raw_text.lower()
        raw_text = re.sub(r'[^\w\s]','',raw_text)
        raw_text = raw_text.split()   
        raw_text = ' '.join(raw_text)

        return raw_text      

    def prepCorpus(self, raw_text):
        self.corpus = corpus = raw_text.split()
        unique_characters = sorted(list(set(corpus)))

        self.char_to_int = dict((c, i) for i, c in enumerate(unique_characters))
        self.int_to_char = dict((i, c) for i, c in enumerate(unique_characters))
        self.n_chars = len(corpus)
        self.n_vocab = len(unique_characters)        

    def prepSamples(self):
        dataX = []
        dataY = []

        for i in range(0, self.n_chars - self.sequence_size, 1):
            seq_in = self.corpus[i:i + self.sequence_size]
            seq_out = self.corpus[i + self.sequence_size]
            dataX.append([self.char_to_int[char] for char in seq_in])
            dataY.append(self.char_to_int[seq_out])
        n_patterns = len(dataX)           

        self.samples = dataX 
        self.targets = dataY 
        
        X = numpy.reshape(dataX, (n_patterns, self.sequence_size, 1))

        self.nSamples = X / float(self.n_vocab)     
        self.nTargets = np_utils.to_categorical(dataY)     

    def fit(self):
        X = self.nSamples 
        y = self.nTargets

        model = Sequential()
        model.add(LSTM(240, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dense(120))
        model.add(LeakyReLU())
        model.add(Dropout(0.2))

        #Hidden Layer
        model.add(BatchNormalization())
        model.add(LSTM(240))
        model.add(BatchNormalization())
        model.add(Dense(120))
        model.add(LeakyReLU())
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        #Output layer
        model.add(Dense(y.shape[1], activation='softmax'))       

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.002)) 

        filepath = self.output_file + ".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks_list)       

    def train(self):
        raw_text = self.rawText()
        self.prepCorpus(raw_text)
        self.prepSamples()
        self.fit()
