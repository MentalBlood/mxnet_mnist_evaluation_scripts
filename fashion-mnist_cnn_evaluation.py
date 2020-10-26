import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='Evaluate CNN model learned on Fashion-MNIST dataset')
parser.add_argument('--model', type=str, nargs='?', required=True,
					default='model.json',
                    help='path to model file')
parser.add_argument('--parameters', type=str, nargs='?', required=True,
					default='parameters.params',
                    help='path to parameters file')
parser.add_argument('--images', type=str, nargs='+', required=True,
                    help='paths to files to evaluate')
args = parser.parse_args()

model = args.model
parameters = args.parameters
images = args.images

def readImage(path):
	imagesBytes = None
	with open(path, 'rb') as image:
		imageBytes = image.read()
		nparr = np.frombuffer(imageBytes, np.uint8)
		grayscale = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
	return np.array(grayscale.astype(np.float32)/255).reshape(28, 28, 1)
data = list(map(readImage, images))



import mxnet.ndarray as nd
from mxnet.gluon import nn

def verifyLoadedModel(net, data):
	data = nd.transpose(nd.array(data), (0, 3, 1, 2))
	out = net(data)
	predictions = nd.argmax(out, axis=1)

	text_labels = 	[
						't-shirt',
						'trouser',
						'pullover',
						'dress',
						'coat',
						'sandal',
						'shirt',
						'sneaker',
						'bag',
						'ankle boot'
					]

	return [(int(p), text_labels[int(p)]) for p in predictions.asnumpy()]

net = nn.SymbolBlock.imports(model, ['data'], parameters)
predictions = verifyLoadedModel(net, data)
print('predictions:', predictions)