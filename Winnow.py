import numpy as np
from random import shuffle

class Winnow:
    def __init__(self,balanced=False,param=2.0):
        self.balanced = balanced
        self.param = float(param)
    
    def fit(self,data,labels):
        if self.balanced:
            self.fitBalanced(data,labels)
        else:
            self.fitNormal(data,labels)
        
    def fitNormal(self,data,labels):
        self.theta = len(data[0])
        self.W = [1.0]*self.theta
        
        # shuffle
        Z = zip(data,labels)
        shuffle(Z)
        data = [z[0] for z in Z]
        labels = [z[1] for z in Z]
        
        for d,l in zip(data,labels):
            y = 1 if np.dot(self.W, d) >= float(self.theta) else -1
            if l==1 and y==-1:
                self.W = [self.param*self.W[i] if d[i]>0 else self.W[i] for i in range(0,len(self.W))]
            elif l==-1 and y==1:
                self.W = [self.W[i]/self.param if d[i]>0 else self.W[i] for i in range(0,len(self.W))]
                
    def fitBalanced(self,data,labels):
        self.theta = len(data[0])
        self.Wp = [1.0]*self.theta
        self.Wn = [1.0]*self.theta
        self.W = [0.0]*self.theta
        
        for d,l in zip(data,labels):
            y = 1 if np.dot(self.W, d) >= float(self.theta) else -1
            if l==1 and y==-1:
                self.Wp = [self.param*self.Wp[i] if d[i]>0 else self.Wp[i] for i in range(0,len(self.Wp))]
                self.Wn = [self.Wn[i]/self.param if d[i]>0 else self.Wn[i] for i in range(0,len(self.Wn))]
            elif l==-1 and y==1:
                self.Wp = [self.Wp[i]/self.param if d[i]>0 else self.Wp[i] for i in range(0,len(self.Wp))]
                self.Wn = [self.param*self.Wn[i] if d[i]>0 else self.Wn[i] for i in range(0,len(self.Wn))]
            self.W = [p-n for p,n in zip(self.Wp,self.Wn)]
                
    def predict(self,test):
        test = [[float(j) for j in i] for i in test]
        if self.balanced:
            predict = [np.sign(np.dot(self.W, d)-float(self.theta)) for d in test]
        else:
            predict = [np.sign(np.dot(self.W,d)-float(self.theta)) for d in test]
        return [1 if l>0 else -1 for l in predict] 
    