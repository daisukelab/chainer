# weight visualizer based on http://d.hatena.ne.jp/shi3z/20151127/1448613415
import matplotlib.pyplot as plt
import numpy as np
import math
import chainer.functions as F
from chainer.links import caffe
from matplotlib.ticker import * 

def plot(layer):
    dim = layer.W.data.shape
    size = int(math.ceil(math.sqrt(dim[0])))
    if len(dim)==4:
	for i, channel in enumerate(layer.W.data):
	    ax = plt.subplot(size,size, i + 1)
	    ax.axis('off')
	    accum = channel[0]
	    for ch in channel:
		accum += ch
	    accum /= len(channel)
	    ax.imshow(accum, interpolation='nearest')
    else:
	plt.imshow(layer.W.data, interpolation='nearest')

def showPlot(layer):
    fig = plt.figure()
    fig.patch.set_facecolor('black')
    fig.suptitle(layer.W.label, fontweight="bold",color="white")
    plot(layer)
    plt.show()


def savePlot(layer,name):
    fig = plt.figure()
    fig.suptitle(name+" "+layer.W.label, fontweight="bold")
    plot(layer)
    plt.draw()
    plt.savefig(name+".png")

def save(func):
    for candidate in func.layers:
	if(candidate[0]) in dir(func):
	    name=candidate[0]
	    savePlot(func[name],name)

## AlexNet visualizer
from chainer.links.caffe import CaffeFunction
import cPickle as pickle
try:    # Load pickled one
    gn = pickle.load(open('bvlc_alexnet.pickle'))
except: # Or load the original & keep it
    print "loading the original caffe model, takes time. hold on and relax..."
    gn = CaffeFunction('bvlc_alexnet.caffemodel')
    pickle.dump(gn, open('bvlc_alexnet.pickle', 'wb'), -1)

showPlot(gn.conv1)
#showPlot(gn.conv2)
#showPlot(gn.conv3)
#showPlot(gn.conv4)
#showPlot(gn.conv5)
#showPlot(gn.fc6)
#showPlot(gn.fc7)
#showPlot(gn.fc8)

# alternative visualizer for conv1 (96, 3, 11, 11)
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.axis('off')
    plt.imshow(gn.conv1.W.data[i].transpose(1, 2, 0))
plt.show()
