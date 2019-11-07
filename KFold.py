
from random import shuffle

class KFold:
    
    def __init__(self,n_splits=5):
        self.n_splits = n_splits
        
    def split(self,data):
        self.data = zip(data,range(0, len(data)))
        self.test_index =  int((float(len(self.data))/float(self.n_splits)) * (self.n_splits - 1))
        self.train = []
        self.test = []
        self.folded = []
        for _ in range(0,self.n_splits):
            shuffle(self.data)
            train = [self.data[i][1] for i in range(0,self.test_index)]
            test = [self.data[i][1] for i in range(self.test_index,len(self.data))] 
            self.folded.append([train,test])
                  
        #return self.train, self.test # return index
        return self.folded