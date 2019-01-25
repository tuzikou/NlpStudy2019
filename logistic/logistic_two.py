import numpy as np 
from functools import reduce
import math

class logistic_two(object):
	
	def __init__(self,dim):
		self.w = [0.1 for _ in range(dim)]
		self.b = 0

	def Predict(self,input_vec):
		zipped = list(zip(input_vec,self.w))
		sumTotal = sum(list(map(lambda x_y:x_y[0]*x_y[1],zipped)))
		return self.sigmoid(sumTotal + self.b)
	    
	def sigmoid(self,x):
		s = 1 / (1 + np.exp(-x))
		return s

	def train(self,input_vecs,labels,rate,iteration):
	    for i in range(iteration):
		    self.train_oneTime(input_vecs,labels,rate)

	def train_oneTime(self,input_vecs,labels,rate):
	    samples = list(zip(input_vecs, labels))
	    for (input_vec,label) in samples:
	    	self.update(input_vecs,labels,rate)
	def update(self,input_vecs,labels,rate):
	    samples = list(zip(input_vecs, labels))
	    n = 0
	    deltaSum = 0
	    deltaRate = 0
	    for (input_vec,label) in samples:
		    output = self.Predict(input_vec)
		    delta = label - output
		    deltaSum += np.dot(input_vec,delta)
		    deltaRate += delta 
		    n += 1
	    deltaSum /= n
	    deltaRate /= n
	    self.w += deltaSum*rate
	    self.b += rate*deltaRate

def getData():
	input_vecs = [[1],[0],[8],[6],[1],[9]]
	labels = [0,0,1,1,0,1]
	return input_vecs,labels

def train_logistic_two():
	p = logistic_two(1)
	input_vecs,labels = getData()
	p.train(input_vecs,labels,0.1,100)
	return p 

if __name__ == '__main__':
	logistic_twos = train_logistic_two()
	print(logistic_twos.Predict([5]))
	print(logistic_twos.Predict([6]))
	print(logistic_twos.Predict([1]))
	print(logistic_twos.Predict([2]))
	print(logistic_twos.Predict([0]))
