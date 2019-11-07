import numpy as np
from random import shuffle

class SGD:
    def __init__(self,epochs,r,W0):#,gamma0,sigma,C):
        self.epochs=epochs
        self.W=W0
        self.r=r
        #self.sigma=sigma
        #self.C=C
        #self.gamma0=gamma0
            
    def fit(self,data,data_label):
        self.data=data
        self.labels=data_label
        
        for _ in range(0,self.epochs): # epoch 
                                               
            # shuffle
            Z = zip(self.data,self.labels)
            shuffle(Z)
            self.data = [z[0] for z in Z]
            self.labels = [z[1] for z in Z]
            
            #t=1                        
            for ins,ins_label in Z: # for each example
                #gamma = float(self.gamma0/float(1.0+float(pow(self.gamma0,t))/self.C))
                y = 1 if np.sign(np.dot(self.W,ins)) >= 0 else -1
                #self.W = [(1.0-gamma)*float(w)+gamma*self.C*(ins_label-y)*x/len(self.data) for w,x in zip(self.W,ins)]
                self.W = [float(w)+self.r*(ins_label-y)*x/len(self.data) for w,x in zip(self.W,ins)]
    
    def predict(self,test):
        test = [[float(j) for j in i] for i in test]
        sign = np.sign([np.dot(self.W,instance) for instance in test])
        return [-1 if s==0 else s for s in sign]
    