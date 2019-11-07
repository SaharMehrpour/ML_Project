import numpy as np
from random import shuffle

class SGDSVM:
    def __init__(self,C,ro,epochs,W0,gamma0=1):
        self.C = float(C)
        self.ro = float(ro)
        self.epochs=epochs
        self.W=[float(w) for w in W0]
        self.gamma0=float(gamma0)
            
    def fit(self,data,data_label):
        self.data=[[float(j) for j in i] for i in data]
        self.labels=[float(l) for l in data_label]
        
        for _ in range(0,self.epochs): # epoch 
                                              
            # shuffle
            Z = zip(self.data,self.labels)
            shuffle(Z)
            self.data = [z[0] for z in Z]
            self.labels = [z[1] for z in Z]
                                   
            t = 1
            for ins,ins_label in Z: # for each example
                gamma = self.gamma0/float(1.0+float(pow(self.gamma0,t))/self.C)

                #if np.sign(np.dot(self.W,[float(a) for a in ins]))*float(ins_label)<1:                
                if ins_label*np.dot(self.W,ins) <= 1.0: 
                    self.W = [(1.0-gamma)*w+gamma*self.C*(ins_label*x)/float(len(self.data)) for w,x in zip(self.W,ins)]
                else:
                    self.W = [(1.0-gamma)*w for w in self.W]
                t += 1
    
    def predict(self,test):
                
        test = [[float(j) for j in i] for i in test]
                
        sign = np.sign([np.dot(self.W,instance) for instance in test])
        return [-1 if s==0 else s for s in sign]
    