import sys
import pickle
import tensorflow as tf
import numpy as np

BUF_SIZE = 400 #determined by longest observation in training data
TRIM_SIZE = 120 #padded size after clips are trimmed at
VAR_PATH = './variables/' #path to pickles

#Reads file with test cases and processes in same way as training data
def read_test(test_file):

    with open(VAR_PATH+'feat2idx.p','rb') as feat_dict:
        feat2idx = pickle.load(feat_dict) #key to convert feats to ints

    test_x = []
    with open(test_file,'r') as test_set:
        for x in test_set:
            x = x.split()
            sample = [] #stores converted feats for each sample

            for feat in x:
                sample.append(feat2idx[feat])

            test_x.append(sample)

    #traverse through test_x to pad
    i = 0 
    while i < len(test_x):
        if len(test_x[i]) > BUF_SIZE:
            test_x[i] = test_x[i][:BUF_SIZE]
        else:
            while len(test_x[i]) < BUF_SIZE:
                test_x[i].append(0)
        i+=1
    
    return test_x

def trim_clip(x):
    start_model = tf.keras.models.load_model('models/start_model')
    end_model = tf.keras.models.load_model('models/end_model')

    x = np.asarray(x)
    x = x.reshape(x.shape[0],x.shape[1],1)
    start_points = [int(pt) for pt in start_model.predict(x=x)]
    end_points = [int(pt) for pt in end_model.predict(x=x)]

    #loop through x and trim each sample at endpoints and pad
    x = x.reshape(x.shape[0],x.shape[1]).tolist()
    i = 0
    while i < len(x):
        x[i] = x[i][start_points[i]:end_points[i]]
        if len(x[i]) > TRIM_SIZE:
            x[i] = x[i][:TRIM_SIZE]
        else:
            while len(x[i]) < TRIM_SIZE:
                x[i].append(0)
        i+=1
    
    return x

def asr_predict(x):
    model = tf.keras.models.load_model('./models/asr_model')

    with open(VAR_PATH+'idx2target.p','rb') as target_dict:
        idx2target = pickle.load(target_dict)

    x = np.asarray(x)
    x = x.reshape(x.shape[0],x.shape[1],1)

    #predict and turn one-hot to target labels
    y_pred = model.predict(x=x)
    y_pred = [list(pred).index(max(pred)) for pred in y_pred]
    y_pred = [idx2target[pred] for pred in y_pred]

    return(y_pred)

if __name__=='__main__':
    test_x = read_test(sys.argv[1]) #argv[1] is test file
    test_x = trim_clip(test_x)

    y_pred = asr_predict(test_x)

    #writes to arg[2] output file
    with open(sys.argv[2],'w') as out_f:
        for pred in y_pred:
            out_f.write(pred)
