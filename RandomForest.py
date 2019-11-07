
import DecisionTree
import random

class RandomForest:
    
    def __init__(self,n_trees=5,n_samples=5,n_attribute=8,discretize=False):
        self.n_trees=n_trees
        self.n_samples = n_samples
        self.n_attribute = n_attribute
        self.discretize = discretize
            
    def sample(self):
        self.set = []
        for _ in range(0,self.n_samples):
            self.set.append(random.choice(range(0,len(self.data))))
        return self.set

    def fit(self,data,data_label):
        self.data = data
        self.data_label = data_label
        self.trees = []
                
        for _ in range(0,self.n_trees):
            train_index = self.sample()
            train = [data[j] for j in train_index]
            train_label = [data_label[j] for j in train_index]
            dt = DecisionTree.DecisionTree(n_attribute=self.n_attribute,discretize=self.discretize)
            dt.fit(train, train_label)            
            self.trees.append(dt)
                
    def predict(self,test):
        predict = []
        for tree in self.trees:
            predict.append(tree.predict(test))
        return zip(*predict)
    