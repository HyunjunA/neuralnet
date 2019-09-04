
import random as rd
import numpy as np
import time


    # """Finish FF,BP,predict without SGD"""


def read_img(f):

    """Return a raster of integers from a PGM as a list of lists."""

    assert f.readline() == 'P5\n'
    f.readline()
    (width, height) = [int(i) for i in f.readline().split(' ')]
    depth = int(f.readline())

    assert depth <= 255

    raster = []
    for y in range(int(width)*int(height)):
        raster.append(int(ord(f.read(1)))/float(depth)) #pixels range from 0 to 1
    return raster


def feed_forward(ntw, this_img):

    output = []
    weights = ntw["weight"]
    biases = ntw["bias"]
    wxb = np.dot(np.array(this_img), weights[0].T) + biases[0]#S1
    hid_output = 1.0/(1.0 + np.exp(-wxb)) #sigmoid function
    output.append(hid_output)

    wxb2 = np.dot(hid_output, weights[1].T) + biases[1] #S2
    output.append(1.0/(1.0+np.exp(-wxb2))) #sigmoid function

    return output

def back_forward(ntw, this_img, ff_output, label):
    weights = ntw["weight"]
    biases = ntw["bias"]
    delta_out = (ff_output[1] - label) * (1 - ff_output[1]) * ff_output[1] # * 2

    #update W_output layer
    biases[1] = biases[1] - rate * delta_out
    weights[1] = weights[1] - rate * delta_out * ff_output[0]

    #update W_inputer layer
    one = np.ones(hidden_num,int)
    delta_hidden = weights[1] * delta_out * (one - ff_output[0]) * ff_output[0]
    biases[0] = biases[0] - rate * delta_hidden
    weights[0] = weights[0] - rate * delta_hidden.T * this_img

    ntw["weight"] = weights
    ntw["bias"] = biases

    return ntw

def weight_init(layer_sizes):
    wei_group = zip(layer_sizes[:-1],layer_sizes[1:])
    weights = []
    for y,x in wei_group:
        weights.append(2 * np.random.random_sample((x,y))-1) #weight = random from -1 to 1
    return weights   # weights[0]:100x960  #weigts[1]:1x100

def bias_init(layer_sizes):
    biases = []
    for y in layer_sizes[1:]:
        biases.append(2 * np.random.random_sample(y)-1) #bias[0]:1x100
    return biases


def network(img_data, label_data):
    layer_sizes = [len(img_data[0]), hidden_num, 1] #[(960, 100), (100, 1)]  / x,y
    weights = weight_init(layer_sizes)
    biases = bias_init(layer_sizes)

    ntw = {"weight": weights, "bias": biases}

    for i in range(epochs):
        for j in range(len(img_data)):

            ff_output = feed_forward(ntw, img_data[j])
            label = label_data[j]
            ntw = back_forward(ntw, img_data[j], ff_output, label)

        if i%100 == 0:
            print "In epochs ---" , i

    return ntw


def predict(img_test, label_test, NNW):
    error = 0.0
    correct = 0
    all_predict = [["Record number","Actual label","Predicted label"]]

    for p in range(len(img_test)):
        img_predict = feed_forward(NNW, img_test[p])[1][0][0]
        actual_label = "down" if label_test[p]==1 else "else"
        predict_label = "down" if img_predict > 0.5 else "else"
        all_predict.append([p,actual_label,predict_label])
        diff = label_test[p] - img_predict
        error += (diff) **2
        if actual_label == predict_label:
            correct += 1
    error = error**(1/2) /float(len(label_test))
    accuracy = correct * 100 /float(len(label_test))
    # print accuracy,(correct,len(label_test)-correct)

    return accuracy, error, np.array(all_predict)

def readfile(filename):
    f_list = open(filename)
    f_list_str = f_list.readlines()
    label_data = []
    img_data = [] #train/test imgs data.
    for i in f_list_str:
        f=open(i.strip('\n'),'rb')
        if 'down' in i:
            label_data.append(1)
        else:
            label_data.append(0)
        img_data.append(read_img(f))
    return (img_data, label_data)

time_start=time.time()

# Training Network
hidden_num = 100
epochs = 1000
rate = 0.1
img_data, label_data = readfile('downgesture_train.list')
NNW = network(img_data, label_data)
print "Neraul Network Training has done."


# Predict
img_test, label_test = readfile('downgesture_test.list')
accuracy, error, all_prediction = predict(img_test, label_test, NNW)
print ("Accuracy of prediction is:  %5.2f%%"% accuracy)
print ("Error of prediction is:  %5.2f"% error)

time_end = time.time()
print ("time cost: %5.2f" % (time_end-time_start))


print all_prediction
