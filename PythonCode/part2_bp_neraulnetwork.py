from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
import time

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


time_start = time.time()
img_data, label_data = readfile('downgesture_train.list')
imgsize = len(img_data[0])

#
dsTrain = SupervisedDataSet(imgsize, 1)

#
for i in range(len(img_data)):
    dsTrain.addSample(img_data[i], label_data[i])


# Create Neraul Network
network = buildNetwork(imgsize, 100, 1, bias=True)
trainer = BackpropTrainer(network, dsTrain, batchlearning = True)


img_test, label_test = readfile('downgesture_test.list')
dsTest = SupervisedDataSet(len(img_test[0]), 36)

for j in range(len(img_test)):
    dsTest.addSample(img_test[j], label_test[j])

# Training
for i in range(100):
    trainer.trainEpochs(10)

    trnresult = percentError(trainer.testOnClassData(), dsTrain['target'])
    print("epoch: %4d" % trainer.totalepochs)

testResult = percentError(trainer.testOnClassData(dataset=dsTest), dsTest['target'])
accuracy = 100 - testResult


print("epoch: %4d" % trainer.totalepochs, "  train error: %5.2f%%" % trnresult, "  test accuracy: %5.2f%%" % accuracy)

time_end = time.time()
print ("time cost: %5.2f" % (time_end-time_start))
