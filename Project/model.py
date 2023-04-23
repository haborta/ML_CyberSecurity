# *****************
# Several tensorflow network
# TO BE CONTINUE
# *****************

import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

class cnn_lstm:   
    def __init__(self, cov_filters, cov_kernel, pool_size, LSTM_units):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dropout(0.2))      
        self.model.add(tf.keras.layers.Conv1D(filters=cov_filters,
                                        kernel_size=cov_kernel,
                                        padding='valid',
                                        activation='relu'))

        self.model.add(tf.keras.layers.MaxPool1D(pool_size=pool_size))
        self.model.add(tf.keras.layers.LSTM(units=LSTM_units))
        self.model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
        
    def build(self, input_shape):
        return self.model.build(input_shape=input_shape)

    def summary(self):
        return self.model.summary()
    
    def compile(self, loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']):
        return self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    def fit(self, train_x, train_labels, validation_data, epochs=5):
        return self.model.fit(x=train_x, y=train_labels, epochs=epochs, 
                              validation_data=validation_data)
        
    def evaluate(self, val_x, val_labels):
        return self.model.evaluate(val_x=val_x, val_labels=val_labels)
        

class dense:
    def __init__(self, neural=64):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(neural))
        model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
        
    def build(self, input_shape):
        return self.model.build(input_shape=input_shape)

    def summary(self):
        return self.model.summary()
    
    def compile(self, loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']):
        return self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    def fit(self, train_x, train_labels, validation_data, epochs=5):
        return self.model.fit(x=train_x, y=train_labels, epochs=epochs, 
                              validation_data=validation_data)
        
    def evaluate(self, val_x, val_labels):
        return self.model.evaluate(x=val_x, y=val_labels)
    
class dense_dropout:
    def __init__(self, dropout=0.1, neural_1=256, neural_2=64):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(neural_1))
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(neural_2))
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
        
    def build(self, input_shape):
        return self.model.build(input_shape=input_shape)

    def summary(self):
        return self.model.summary()
    
    def compile(self, loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']):
        return self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    def fit(self, train_x, train_labels, validation_data, epochs=5):
        return self.model.fit(x=train_x, y=train_labels, epochs=epochs, 
                              validation_data=validation_data)
        
    def evaluate(self, val_x, val_labels):
        return self.model.evaluate(val_x=val_x, val_labels=val_labels)