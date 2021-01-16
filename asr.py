import tensorflow as tf
import numpy as np
import sys
import pickle
from math import ceil


BUFFER_SIZE = 400 #calculated based on largest training data length
TRIM_SIZE = 120 #after clips trimmed, padded to this length
VAR_PATH = './variables/'

def one_hot_targets(target_set):
    with open(target_set,'r') as target_file:
        target2idx = {} #associates word with int representation
        idx2target = [] #list of target to retrieve from index

        target_idx = 0 #int representing each class
        indices = [] #indices to activate for one_hot encoder
        
        for word in target_file:
            if word in target2idx:
                indices.append(target2idx[word])
            else:
                target2idx[word] = target_idx
                indices.append(target_idx)
                idx2target.append(word)
                target_idx += 1

        #so same process can be done on test data and original string recoverable
        with open(VAR_PATH+'target2idx.p','wb') as targ2idx:
            pickle.dump(target2idx,targ2idx)
        with open(VAR_PATH+'idx2target.p','wb') as idx2targ:
            pickle.dump(idx2target,idx2targ)

        depth = len(target2idx) #length of one hot vector

        return tf.one_hot(indices,depth,on_value=1.0,off_value=0.0),idx2target

#changes features to integers
#args: x-list of samples; feature_set-list of possible features (fenemes)
def encode_features(sample_file,feature_set):
    feature2idx = {} #associates fenemes with int representation
    feat_idx = 0

    train_x = [] #list of samples with int features

    with open(feature_set,'r') as features:
        for feat in features:
            feat = feat.translate(str.maketrans('','','\n'))
            feature2idx[feat] = feat_idx
            feat_idx += 1
        
        with open(VAR_PATH+'feat2idx.p','wb') as feat2idx:
            pickle.dump(feature2idx,feat2idx) #downloades so same processing can be done on test data

    with open(sample_file,'r') as samples:
        for x in samples:
            x = x.split()
            sample = [] #single sample to be filled with int representatons of features
            for feat in x:
                sample.append(feature2idx[feat])
            train_x.append(sample)
    
    max_length = max([len(vector) for vector in train_x])
    
    i = 0 #for traversing x
    while i < len(train_x):
        if len(train_x[i]) > BUFFER_SIZE:
            train_x[i] = train_x[i][:BUFFER_SIZE]
        else:
            while len(train_x[i]) < BUFFER_SIZE:
                train_x[i].append(0)
        i+=1
    return train_x

def build_model(input_shape,output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,1)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
        tf.keras.layers.Dropout(.4),
        tf.keras.layers.Dense(output_shape,activation='softmax')
    ])
    return model

#Trims samples to include only utterance; window given by endpoints file
#pads samples to all be same length (max length)
def trim_clip(x,endpoints_file):
    start_points = []
    end_points = []

    with open(endpoints_file) as endpts:
        for pair in endpts:
            start,end = [int(pt) for pt in pair.split()]
            start_points.append(start)
            end_points.append(end)
 
    #loop through x and trim based on calculated endpoints
    x = x.reshape(x.shape[0],x.shape[1]).tolist()
    i=0
    while i < len(x):
        startpt = start_points[i]
        endpt = end_points[i]

        x[i] = x[i][startpt:endpt]
        i+=1

    #pad samples to be same length
    x_idx = 0 #for traversing through x samples
    while x_idx < len(x):
        if len(x[x_idx]) > TRIM_SIZE:
            x[x_idx] = x[x_idx][:TRIM_SIZE]
        while len(x[x_idx]) < TRIM_SIZE:
            x[x_idx].append(0)
        x_idx += 1
    
    return x

if __name__=='__main__':
    x = encode_features(sys.argv[1],sys.argv[3]) #arg 1 is list of training samples; arg 3 is list of possible targets
    x = trim_clip(np.asarray(x),sys.argv[4]) #arg 4 is train endpoints; windows of utterance
    y,idx2target = one_hot_targets(sys.argv[2]) #returns matrix of one hot encoded targets

    #shuffle and split data
    data = list(zip(x,y))
    np.random.shuffle(data)
    x,y = zip(*data)

    train_x = np.asarray(x[:ceil(len(x)*.8)])
    train_y = np.asarray(y[:ceil(len(y)*.8)])


    val_x = np.asarray(x[ceil(len(x)*.8):])
    val_y = np.asarray(y[ceil(len(y)*.8):])

    #reshape for lstm 
    train_x = train_x.reshape(train_x.shape[0],train_x.shape[1],1)
    val_x = val_x.reshape(val_x.shape[0],val_x.shape[1],1)

    input_shape = len(train_x[0])
    output_shape = len(train_y[0])
    model = build_model(input_shape,output_shape)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
    model.summary()
    model.fit(x=train_x,y=train_y,epochs=100,validation_data=(val_x,val_y))

    model.save('models/asr_model')
