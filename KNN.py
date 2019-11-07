import numpy as np

class KNN:
    
    def __init__(self,K=1,norm=2): # norm inf
        self.K = K
        self.norm = norm
        
    def fit(self,data,labels):
        self.data = data
        self.labels = labels
        
        self.groups = [[] for i in range(len(data[0])+1)] # assume 0-1 entries
        
        for i in range(len(data)):
            data[i]=[float(d) for d in data[i]]
            self.groups[int(np.dot(data[i],data[i]))].append([data[i],labels[i]])
        
    
    def predict(self,test):
        predict = []
        
        for ins in test:
            tmpPredict = []
            
            card = np.dot(ins,ins)
            for l in range(max(0,int(card-(self.K-1)/2)),int(card+(self.K-1)/2)):
                if l >= len(self.groups):
                    continue
                for index in range(len(self.groups[l])):
                    dist = 0
                    for j in range(len(ins)):
                        dist += np.power(np.abs(float(self.groups[l][index][0][j])-float(ins[j])),self.norm)
                    dist = np.power(dist,1.0/float(self.norm))
                    
                    tmpPredict.append([dist,self.groups[l][index][1]])
            
            tmpPredict = sorted(tmpPredict,key=lambda x: x[0])
            tmpPredict = tmpPredict[0:self.K]
            
            if tmpPredict.count(1.0) > tmpPredict.count(-1.0):
                predict.append(1.0)
            else:
                predict.append(-1.0)
                
        return predict