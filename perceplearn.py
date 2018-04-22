import os
import sys
import operator
import numpy as np
import random
from random import shuffle
from collections import Counter

SEPERATOR = '****************#################*****************###############\n'

def getFileContents(filename):
	data = None
	with open(filename, 'r') as f:
		data = f.readlines()
	return data

def getFileFromCommandLine():
	filename = sys.argv[1]
	return getFileContents(filename)


def clean_sentence(sentence):
	chars_to_remove = ['~', '`','.', '!', '?', '@', '#', '$', '%',\
						'^', '&', ',', '(', ')', '_', '+', '*',\
						'=', '<', '>', ';', ':', '"', '[', ']', '/',\
						'\\', '|', '~', '{', '}']

	chars_to_remove += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

	stop_words =  ['being', 'our', 'd', 'them', 'more', 've', 'd', 'its',\
					's', 'my', 'such', 'from', 'only', 'as', 'should', 'all',\
					'over', 'during', 'yourselves', 'has', 'myself', 'am',\
					'the', 'ourselves', 'did', 'some', 'after', 'that', 'or',\
					'which', 'if', 'this', 'into', 'having', "aren't", 'could',\
					'an', 'would', 'it', 'out', 'won', 're', 'themselves',\
					'whom', 'they', 'couldn', 'is', 'own', 'but', 'up', 'her',\
					'on', 'while', 'before', 'are', 'both', 'each', 'very',\
					'he', 'don', 'at', 'had', 'm', 'how', 'wasn', 'was',\
					'herself', 'nor', 'were', 'yours', 'does', 'down', 'himself',\
					'ought', 'with', 'ours', 'doing', 'in', 'once', 'him',\
					'same', 'a', 'isn', 'until', 'who', 'you', 'be', 'between',\
					'here', 'been', "i'll", 'most', 'itself', 'against', 'under',\
					'so', 'again', 'to', 'when', 'then', 'these', 'of', 'have',\
					'above', 'by', 'why', 'i', 'theirs', 'yourself', 'for',\
					'me', 'those', 'further', 'where', 'let', 'below', 'through',\
					'other', 'than', 'their', 'she', 'your', 'too', 'do', 'and',\
					'hers', 'we', 'there', 'any', 'because', 'about', 'what', 'few',\
					'his', 't', "didn't", "i'll", "we'll", "i've", "we've", "couldn't", "wasn't"]

	# stop_words += ['aren', 'll', 'didn', 'l', 'doesn', 'weren', 'lets', 'haven']

	# stop_words += ['ourselves','hers','between','yourself','but','again','there','about','once','during','out','very','having','with','they','own','an','be','some','for','do','its','yours','such','into','of','most','itself','other','off','is','s','am','or','who','as','from','him','each','the','themselves','until','below','are','we','these','your','his','through','don','nor','me','were','her','more','himself','this','down','should','our','their','while','above','both','up','to','ours','had','she','all','no','when','at','any','before','them','same','and','been','have','in','will','on','does','yourselves','then','that','because','what','over','why','so','can','did','not','now','under','he','you','herself','has','just','where','too','only','myself','which','those','i','after','few','whom','t','being','if','theirs','my','against','a','by','doing','it','how','further','was','here','tha']
	stop_words = list(set(stop_words))  
	sentence = sentence.lower()

	for char in chars_to_remove:
		sentence = sentence.replace(char, ' ')

	words = sentence.split()

	words = [word for word in words if word not in stop_words]

	# print '\n\n'
	# print sentence
	# # words = sentence.split()
	# print words
	# print '\n\n\n'
	words = [ give_base_word(word) for word in words ]
	# words = [word for word in words if len(word) > 2]
	words = [ word for word in words if len(word.strip()) > 0]

	# words = [word[:-2] if (word.endswith('ed')) and (len(word) > 2) else word for word in words]
	# words = [word if not word.endswith('ing') else word[:-3] for word in words]
	# words = list(set(words))
	# print words
	return words


def give_base_word(word):
	base_word = word
	try:
	# return word
		base_word = word
		if word.endswith('ing'):
			# return word
			base_word = word[:-3]
			if len(base_word) > 3:
				if base_word[-1] == base_word[-2]:
					base_word = base_word[:-1]
				elif base_word[-1] == 'e':
					pass
				elif base_word[-1] == 'k' and base_word[-2] == 'c':
					base_word = base_word[:-1]
				elif base_word[-2] in ['e']:
					pass
				else:
					base_word += 'e'

		elif word.endswith('ly') and len(word) > 6:
			base_word = word[:-2]

		elif word.endswith('ed'):
			base_word = word[:-2]
			if len(base_word) < 2:
				base_word = word
			elif base_word[-1] == 'y':
				pass
			elif base_word[-1] == 'e':
				pass
			elif base_word[-1] == 'i':
				base_word = base_word[:-1] + 'y'

			elif len(base_word) > 2 and  base_word[-1] == base_word[-2]:
				# print base_word + 'ed'
				base_word = base_word[:-1]
			elif len(base_word) > 2 and  base_word[-1] == 'k' and base_word[-2] == 'c':
				base_word = base_word[:-1]
			elif len(base_word) > 2 and  base_word[-2] in ['e']:
				pass
			elif len(base_word) > 2 and  base_word[-2] in ['a', 'e', 'i', 'o', 'u']:
				# pass
				base_word += 'e'
			else:
				pass

		# print word.ljust(25),' => ', base_word.ljust(25)
	except:
		print "Exception in base word"
	return base_word

class Perceptron(object):
	def __init__(self, data, iterations=50, model_file=None):
		
		self.emotion_bias = 0
		self.emotion_weights = None
		
		self.authenticity_bias = 0
		self.authenticity_weights = None
		
		self.iterations = iterations
		
		self.raw_data = data
		self.word_to_index = {}
		self.total_words = 0
		self.clean_data = []
		
		self.authenticity_target_value = {'Fake': -1, 'True': 1}
		self.authenticity_target_name = {-1: 'Fake', 1: 'True'}
		
		self.emotion_target_value = {'Neg': -1, 'Pos': 1}
		self.emotion_target_name = {-1: 'Neg', 1: 'Pos'}
		if model_file != None:
			# print "Using model file"
			self.loadModelFromFile(model_file)
		elif data != None:
			# print "Training perceptron from raw data"
			self.initialise()
		else:
			pass
			# print "Provide at least model file or training data"
			

	def loadModelFromFile(self, model_file):
		data = getFileContents(model_file)

		self.word_to_index = {}
		emotion_bias = 0
		emotion_weights = []
		authenticity_bias = 0
		authenticity_weights = []
		
		switch = 0
		for line in data:
			if line == SEPERATOR:
				switch += 1
				continue
				
			if switch == 0:
				word, index = line.strip().split('\t')
				index = int(index)
				self.word_to_index[word] = index
				
			if switch == 1:
				try:
					authenticity_weights = map(float, line.strip().split('\t'))
				except:
					print "Exception raised in Authenticity weights"
			
			if switch == 2:
				try:
					emotion_weights = map(float, line.strip().split('\t'))
				except:
					print "Exception raised in Emotion weights"

			if switch == 3:
				try:
					authenticity_bias = float(line.strip())
				except:
					print "Exception raised in Authenticity bias"

			if switch == 4:
				try:
					emotion_bias = float(line.strip())
				except:
					print "Exception raised in Emotion bias"

		self.authenticity_bias = authenticity_bias
		self.emotion_bias = emotion_bias
		self.authenticity_weights = np.array(authenticity_weights)
		self.emotion_weights = np.array(emotion_weights)
		self.total_words = len(self.word_to_index.keys())
		
	def initialise(self):
		self.getUniqueWords()
		self.authenticity_bias = 0
		self.authenticity_weights = np.zeros((1, self.total_words))
		self.emotion_bias = 0
		self.emotion_weights = np.zeros((1, self.total_words))
		# random.seed(16)
		# shuffle(self.clean_data)
		
	def splitClassNData(self, line):
		tokens = line.strip().split()
		data_id = tokens[0]
		truthfulness = tokens[1]
		emotion = tokens[2]
		data = ' '.join(tokens[3:])
		data = clean_sentence(data)
		return (data_id, truthfulness, emotion, data)
	
	def splitIdNData(self, line):
		tokens = line.strip().split()
		data_id = tokens[0]
		data = ' '.join(tokens[1:])
		data = clean_sentence(data)
		return (data_id, data)
	
	def addWordsToDict(self, words):
		for word in words:
			try:
				word_index = self.word_to_index[word]
			except:
				#New word is encountered
				word_index = self.total_words
				self.word_to_index[word] = word_index
				self.total_words += 1
		
	def getUniqueWords(self):
		for line in self.raw_data:
			processed_data = {}
			try:
				data_id, truthfulness, emotion, words = self.splitClassNData(line)
				target = self.getTargetValue(truthfulness, emotion)
				self.addWordsToDict(words)
				processed_data['data_id'] = data_id
				processed_data['words'] = words
				processed_data['target'] = target
				processed_data['bow'] = None
				self.clean_data.append(processed_data)
			except Exception as e:
				print "Exception raised in getUniqueWords", e

	def train(self):
		restart = 3
		for count in range(self.iterations):
			if self.fit(count) == False:
				if self.model_file_name == 'vanillamodel.txt':
					print "Perceptron converged at [%d] iterations"%count
					restart -= 1
				if restart <= 0:
					if self.model_file_name == 'vanillamodel.txt':
						break
			elif restart != 3:
				print 'Perceptron changed'
				restart = 3
			else:
				pass
				
	def getBagOfWords(self, words):
		word_counter = Counter(words)
		bow = np.zeros((1, self.total_words))
		for word_info in word_counter.most_common():
			word, count = word_info
			try:
				word_index = self.word_to_index[word]
				bow[0][word_index] += count
			except:
				pass
		return bow
	
	def getTargetValue(self, truthfulness, emotion):
		return (self.authenticity_target_value[truthfulness], self.emotion_target_value[emotion])
	
	def fit(self, count):
		# random.seed(234)
		# shuffle(self.clean_data)
		weights_updated = False
		for data_point in self.clean_data:
			try:
				data_id = data_point['data_id']
				words = data_point['words']
				target = data_point['target']
				bow = data_point['bow']
				if type(bow) == type(None):
					bow = self.getBagOfWords(words)
					data_point['bow'] = bow
				
				weights_updated = self.updateWeights(bow, target) or weights_updated
			except Exception as e:
				print "Exception raised in fit", e
		return weights_updated
				
	def getClassForReview(self, bow):
		authenticity, emotion = None, None
		
		a = np.sum(np.multiply(self.authenticity_weights, bow)) + self.authenticity_bias
		if a >= 0:
			authenticity = self.authenticity_target_name[1]
		else:
			authenticity = self.authenticity_target_name[-1]
		
		a = np.sum(np.multiply(self.emotion_weights, bow)) + self.emotion_bias
		if a >= 0:
			emotion = self.emotion_target_name[1]
		else:
			emotion = self.emotion_target_name[-1]
			
		return authenticity, emotion
				
	def predict(self, untagged_data):
		output = ''
		for line in untagged_data:
			data_id, words = self.splitIdNData(line)
			bow = self.getBagOfWords(words)
			classes = self.getClassForReview(bow)
			output += '%s %s %s\n'%(data_id, classes[0], classes[1])
		self.writeOutputToFile(output)
	
	def writeOutputToFile(self, data):
		with open('percepoutput.txt', 'w') as f:
			f.write(''.join(data))
			f.close()
			
	def writeModelToFile(self):
		output = ''
		# Writes all words with their indexes
		for word, index in sorted(self.word_to_index.items(), key=operator.itemgetter(1)):
			output += '%s\t%d\n'%(word, index)
			
		output += SEPERATOR
		output += '\t'.join(map(str, self.authenticity_weights.tolist()[0])) + '\n'
		output += SEPERATOR
		output += '\t'.join(map(str, self.emotion_weights.tolist()[0])) + '\n'
		output += SEPERATOR
		output += str(self.authenticity_bias) + '\n'
		output += SEPERATOR
		output += str(self.emotion_bias) + '\n'
		output += SEPERATOR
		
		with open(self.model_file_name, 'w') as f:
			f.write(output)
			f.close()

class VanillaPerceptron(Perceptron):
	def __init__(self, tagged_data, iterations=10):
		Perceptron.__init__(self, tagged_data)
		self.iterations = iterations
		self.model_file_name = 'vanillamodel.txt'
		
	def updateWeights(self, bow, target):
		y = target[0]
		a = np.sum(np.multiply(self.authenticity_weights, bow)) + self.authenticity_bias
		changed = False
		if a * y <= 0:
			self.authenticity_weights += y * bow
			self.authenticity_bias += y
			changed = True
		
		y = target[1]
		a = np.sum(np.multiply(self.emotion_weights, bow)) + self.emotion_bias
		if a * y <= 0:
			self.emotion_weights += y * bow
			self.emotion_bias += y
			changed = True
		return changed
		
class AveragedPerceptron(Perceptron):
	def __init__(self, tagged_data, iterations=10):
		Perceptron.__init__(self, tagged_data)
		self.iterations = iterations
		self.model_file_name = 'averagedmodel.txt'
		
	def updateWeights(self, bow, target):
		y = target[0]
		a = np.sum(np.multiply(self.authenticity_weights, bow)) + self.authenticity_bias
		changed = False
		if a * y <= 0:
			self.authenticity_weights += y * bow
			self.authenticity_bias += y
			
			self.authenticity_weights_average += y * bow * self.counter
			self.authenticity_bias_average += y * self.counter
			changed = True
			
		y = target[1]
		a = np.sum(np.multiply(self.emotion_weights, bow)) + self.emotion_bias
		if a * y <= 0:
			self.emotion_weights += y * bow
			self.emotion_bias += y
			
			self.emotion_weights_average += y * bow * self.counter
			self.emotion_bias_average += y * self.counter
			changed = True
		self.counter += 1
		return changed
		
	def initialiseWeights(self):
		self.emotion_bias_average = 0
		self.authenticity_bias_average = 0
		self.emotion_weights_average = np.zeros((1, self.total_words))
		self.authenticity_weights_average = np.zeros((1, self.total_words))
		self.counter = 1
		
	def train(self):
		self.initialiseWeights()
		super(AveragedPerceptron, self).train()
		
		self.authenticity_weights -= self.authenticity_weights_average / (self.counter * 1.0)
		self.authenticity_bias -= self.authenticity_bias_average / (self.counter * 1.0)
		
		self.emotion_weights -= self.emotion_weights_average / (self.counter * 1.0)
		self.emotion_bias -= self.emotion_bias_average / (self.counter * 1.0)

if __name__ == '__main__':
	tagged_data = getFileFromCommandLine()
	# tagged_data = getFileContents('data/train-labeled.txt')
	# untagged_data = getFileContents('data/dev-text.txt')

	iterations = 90
	model = VanillaPerceptron(tagged_data, iterations=iterations)
	model.train()
	model.writeModelToFile()

	model = AveragedPerceptron(tagged_data, iterations=iterations+10)
	model.train()
	model.writeModelToFile()

	print "Training Done"