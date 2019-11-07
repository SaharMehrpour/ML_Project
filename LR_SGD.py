import numpy as np
from random import shuffle
import math

class LR_SGD:
    def __init__(self,gamma0,epochs,W0,sigma,C):
        self.epochs=epochs
        self.W=W0
        self.gamma0=gamma0
        self.sigma = float(sigma)
        self.C = C
            
    def fit(self,data,data_label):
        self.data=data
        self.labels=data_label
        self.Objective = []
        
        for _ in range(0,self.epochs): # epoch 
                                               
            # shuffle
            Z = zip(self.data,self.labels)
            shuffle(Z)
            self.data = [z[0] for z in Z]
            self.labels = [z[1] for z in Z]
                                                
            t=1
            for ins,ins_label in Z: # for each example
                gamma = float(self.gamma0/float(1.0+float(pow(self.gamma0,t))/self.C))             
                try:
                    term1 = float(ins_label) / float(1+math.exp(float(ins_label) * np.dot(self.W,ins)))
                except OverflowError:
                    term1 = 1.0-float(ins_label) / float(1+math.exp(-1.0*float(ins_label) * np.dot(self.W,ins)))
                term2 = 2.0/ (self.sigma*self.sigma) if self.sigma!=float('inf') else 0
                self.W = [float(w)+gamma*(term1*float(x)-(term2*float(w))) for w,x in zip(self.W,ins)]
                
                t += 1
            '''    
            tempObj = 1.0/float(self.sigma*self.sigma)+float(np.dot(self.W,self.W))
            for ins,ins_label in Z: # for each example
                try:
                    tempObj += np.log2(1.0+np.exp(-1.0*ins_label*np.dot(self.W,ins)))
                except OverflowError:
                    alpha2 = float(-1.0*ins_label*np.dot(self.W,ins))/2.0
                    tempObj += np.log2(np.exp(-1.0*alpha2)+np.exp(alpha2))+np.log2(np.exp(alpha2))
            self.Objective.append(tempObj)
                '''
    def predict(self,test):
        test = [[float(j) for j in i] for i in test]
        sign = np.sign([np.dot(self.W,instance) for instance in test])
        return [-1 if s==0 else s for s in sign]
    