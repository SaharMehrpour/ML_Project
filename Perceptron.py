
import numpy as np
from random import shuffle

class Perceptron:
    def __init__(self,W0,r=0.01,epoch=1):
        self.W = W0
        self.r = r
        self.epoch = epoch
        
    def fit(self,data,data_label):
        self.data = [[float(j) for j in i] for i in data]
        self.labels = data_label
        
        for _ in range(0,self.epoch):
            
            # shuffle
            Z = zip(self.data,self.labels)
            shuffle(Z)
            self.data = [z[0] for z in Z]
            self.labels = [z[1] for z in Z]
            
            for d,l in zip(self.data,self.labels):
                predictLabel = -1 if np.sign(np.dot(self.W,d)) <= 0 else 1
                if predictLabel != l :
                    self.W = [float(w) + self.r * float(l) * x for x,w in zip(d,self.W)]  

    def predict(self,test):
        test = [[float(j) for j in i] for i in test]
                
        sign = np.sign([np.dot(self.W,instance) for instance in test])
        return [-1 if s==0 else s for s in sign]