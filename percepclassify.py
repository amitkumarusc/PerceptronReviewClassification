import os
import sys
import operator
import numpy as np
from random import shuffle
from collections import Counter

from perceplearn import Perceptron

def getFileContents(filename):
	data = None
	with open(filename, 'r') as f:
		data = f.readlines()
	return data

def getFilesFromCommandLine():
	model_file = sys.argv[1]
	data_file = sys.argv[2]
	return model_file, data_file


if __name__ == '__main__':
	model_file, data_file = getFilesFromCommandLine()

	untagged_data = getFileContents(data_file)
	
	model = Perceptron(data=None, model_file=model_file)
	model.predict(untagged_data)
	
	print "Testing Done"