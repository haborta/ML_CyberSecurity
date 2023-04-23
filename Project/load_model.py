# *****************
# Several tensorflow network
# TO BE CONTINUE
# *****************

import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


# pre-trained text embedding model from TensorFlow Hub
# more text-embedding download in https://tfhub.dev/s?module-type=text-embedding
import tensorflow_hub as hub
import tensorflow_text
# samll_bert: Bidirectional Encoder Representations from Transformers
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder_pooled = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2", 
                                trainable=False, output_key="pooled_output")
encoder_seq = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2", 
                            trainable=False, output_key="sequence_output")
# nnlm: Text embeddings based on feed-forward Neural-Net Language Models
hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2", 
                           dtype=tf.string, trainable=False)

# CNN-LSTM Model structure in https://ieeexplore.ieee.org/abstract/document/9178321
def lstm(LSTM_units=100, embedding='bert', dropout=0.1):
    input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    if embedding == "nnlm":
        layer1 = hub_layer(input)
        raise Exception("you should not use nnlm in cnn-LSTM, or reshape nnlm output(TBC)")
    elif embedding == "bert":
        layer2 = preprocessor(input)
        layer1 = encoder_seq(layer2)
    else:
        raise Exception("need embedding method")
    layer2 = preprocessor(input)
    layer1 = encoder_seq(layer2)
    layer4 = tf.keras.layers.LSTM(units = LSTM_units, dropout = dropout)(layer1)
    output = tf.keras.layers.Dense(1,activation='sigmoid')(layer4)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model

def cnn_lstm(cov_filters=32, cov_kernel=4, pool_size=2, LSTM_units=20, dropout=0.2, embedding='bert'):
    input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    if embedding == "nnlm":
        layer1 = hub_layer(input)
        raise Exception("you should not use nnlm in cnn-LSTM, or reshape nnlm output(TBC)")
    elif embedding == "bert":
        layer2 = preprocessor(input)
        layer1 = encoder_seq(layer2)
    else:
        raise Exception("need embedding method")
    layer3 = tf.keras.layers.Dropout(dropout)(layer1)
    layer4 = tf.keras.layers.Conv1D(filters=cov_filters,
                                    kernel_size=cov_kernel,
                                    padding='valid',
                                    activation='relu')(layer3)
    layer5 = tf.keras.layers.MaxPool1D(pool_size=pool_size)(layer4)
    layer6 = tf.keras.layers.LSTM(units=LSTM_units)(layer5)
    output = tf.keras.layers.Dense(1,activation='sigmoid')(layer6)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model

def dense(embedding='bert', neural=64):
    input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    if embedding == "nnlm":
        layer1 = hub_layer(input)
    elif embedding == "bert":
        layer2 = preprocessor(input)
        layer1 = encoder_pooled(layer2)
    else:
        raise Exception("need embedding method")           
    layer3 = tf.keras.layers.Dense(neural)(layer1)
    output = tf.keras.layers.Dense(1,activation='sigmoid')(layer3)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model

def dense_dropout(embedding='bert', dropout=0.1, neural_1=256, neural_2=64):
    input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    if embedding == "nnlm":
        layer1 = hub_layer(input)
    elif embedding == "bert":
        layer2 = preprocessor(input)
        layer1 = encoder_pooled(layer2)
    else:
        raise Exception("need embedding method")
    layer3 = tf.keras.layers.Dense(neural_1)(layer1)
    layer4 = tf.keras.layers.Dropout(dropout)(layer3)
    layer5 = tf.keras.layers.Dense(neural_2)(layer4)
    layer6 = tf.keras.layers.Dropout(dropout)(layer5)
    output = tf.keras.layers.Dense(1,activation='sigmoid')(layer6)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model
