
import numpy as np

class PreProcess:
    
    def __init__(self,data,n_buckets=0,swap_labels=False,func='bucket'):
        
        self.data = data
        self.dataT = zip(*[map(float, d) for d in data])
        self.n_buckets = n_buckets
        self.buckets = []
        self.swap_labels = swap_labels
        self.func = func
        
        for d in self.dataT:
            temp = [min(d)]
            for i in range(1,self.n_buckets):
                temp.append(np.percentile(d, i* 100/self.n_buckets))
            temp.append(float('inf')) 
            self.buckets.append(temp)
        
        self.findUnnecessaryFeatures()
            
    def fit(self,data):   
        if self.func == 'clean':
            return self.removeUnnecessaryFeatures(data)
        if self.n_buckets == 0:
            return data 
        if self.func == 'bucket':
            return self.bucket(data)
        if self.func == 'normalize':
            return self.normalize(data)
        else:
            return self.makeBoolean(data)
    
    def bucket(self,data):

        newData = []
        for i in range(0,len(data)):
            tmp = []
            for j in range(0,len(data[i])):
                if (float(data[i][j]) == 0):
                    tmp.append(0)
                    continue
                for k in range(0,self.n_buckets):
                    if (float(self.buckets[j][k]) <= float(data[i][j]) and float(data[i][j]) < float(self.buckets[j][k+1])):
                        tmp.append(k)
                        break
            newData.append(tmp)
        newData = self.removeUnnecessaryFeatures(newData)        
        return newData
    
    def makeBoolean(self,data):
        
        newData = []
        for i in range(0,len(data)):
            tmp = []
            for j in range(0,len(data[i])):
                if (float(data[i][j]) == 0):
                    for _ in range(0,self.n_buckets):
                        tmp.append(0)
                    continue
                for k in range(0,self.n_buckets):
                    if (float(self.buckets[j][k]) <= float(data[i][j]) and float(data[i][j]) < float(self.buckets[j][k+1])):
                        tmp.append(1)
                    else:
                        tmp.append(0)
            newData.append(tmp)
        newData = self.removeUnnecessaryFeatures(newData) 
        return newData
                
    def processLabels(self,labels):
        if (self.swap_labels):
            return [-1.0 if d==1.0 else 1.0 for d in labels]     
        else:
            return labels       
                
    def findUnnecessaryFeatures(self):
        self.usefulIndex = []
        dt = np.asarray(self.data).T.tolist()
        
        for i in range(len(self.data[0])):
            if dt[i].count(dt[i][0]) != len(dt[i]):
                self.usefulIndex.append(i)
    
    def removeUnnecessaryFeatures(self,data):
        newData = []
        
        dt = np.asarray(data).T.tolist()
        
        for index in self.usefulIndex:
            newData.append(dt[index])
            
        return np.asarray(newData).T.tolist()  
    
    def normalize(self,data):
        
        dataT = zip(*[map(float, d) for d in data])
        minMax = []
        for d in dataT:
            minMax.append([min(d),max(d),float(max(d))-float(min(d))])
             
        newData = []
        for i in range(0,len(data)):
            tmp = []
            for j in range(0,len(data[i])):
                if (float(data[i][j]) == 0):
                    tmp.append(0)
                elif (float(minMax[j][2]) == 0):
                    tmp.append(float(data[i][j]))
                else:
                    tmp.append((float(data[i][j])-float(minMax[j][0]))/float(minMax[j][2]))
            newData.append(tmp)
        return newData
          
                    