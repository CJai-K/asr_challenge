import tensorflow as tf
import numpy as np
import sys
from math import ceil

TRIM_LENGTH = 100 #length of final vector after trimmed and padded
BUFFER_SIZE = 400 #length of vector when read in before endpoint clipping

def read_data(sample_file,feature_set):
    feature2idx = {} #associates fenemes with int representation
    feat_idx = 0

    train_x = [] #list of samples with int features

    with open(feature_set,'r') as features:
        for feat in features:
            feat = feat.translate(str.maketrans('','','\n'))
            feature2idx[feat] = feat_idx
            feat_idx += 1

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
        while len(train_x[i]) < BUFFER_SIZE:
            train_x[i].append(0)
        i+=1
    return train_x

def read_endpoints(endpoints_file):
    start_points = []
    end_points = []

    with open(endpoints_file,'r') as endpts:
        for pair in endpts:
            start,end = [int(pos) for pos in pair.split()]

            start_points.append(start)
            end_points.append(end)
    
    return start_points,end_points

def build_start_model(input_shape,output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,1)),
        tf.keras.layers.Conv1D(128,3,activation='relu'),
        tf.keras.layers.Conv1D(128,3,activation='relu'),
        tf.keras.layers.MaxPooling1D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(.7),
        tf.keras.layers.Dense(output_shape,activation='relu')
    ])

    return model

def build_end_model(input_shape,output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,1)),
        tf.keras.layers.Conv1D(64,3,activation='relu'),
        tf.keras.layers.Conv1D(64,3,activation='relu'),
        tf.keras.layers.MaxPooling1D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(.7),
        tf.keras.layers.Dense(output_shape,activation='relu')
    ])

    return model

if __name__=='__main__':
    x = read_data(sys.argv[1],sys.argv[3]) #argv[1] is training feature file and argv[3] is feneme set
    start_points,end_points = read_endpoints(sys.argv[2]) #argv[2] enpoints file

    cat_list = list(zip(x,start_points,end_points))
    np.random.shuffle(cat_list)
    x,start_points,end_points = zip(*cat_list)

    split = .8
    train_x = np.asarray(x[:ceil(len(x)*split)])
    train_start = np.asarray(start_points[:ceil(len(start_points)*split)])
    train_end = np.asarray(end_points[:ceil(len(end_points)*split)])


    val_x = np.asarray(x[ceil(len(x)*split):])
    val_start = np.asarray(start_points[ceil(len(start_points)*split):])
    val_end = np.asarray(end_points[ceil(len(end_points)*split):])

    features = 1 #x.shape[1] = features*time_steps

    train_x = train_x.reshape(train_x.shape[0],train_x.shape[1],1)
    val_x = val_x.reshape(val_x.shape[0],val_x.shape[1],1)
    input_shape = len(train_x[0])
    output_shape = 1

    start_model = build_start_model(input_shape,output_shape)
    start_model.compile(loss=tf.keras.losses.MeanAbsoluteError(),optimizer=tf.keras.optimizers.Adam(),metrics=['mse','mae'])
    start_model.fit(x=train_x,y=train_start,epochs=60,validation_data=(val_x,val_start))

    end_model = build_end_model(input_shape,output_shape)
    end_model.compile(loss=tf.keras.losses.MeanAbsoluteError(),optimizer=tf.keras.optimizers.Adam(),metrics=['mse','mae'])
    end_model.fit(x=train_x,y=train_end,epochs=60,validation_data=(val_x,val_end))

    start_model.save('models/start_model')
    end_model.save('models/end_model')
