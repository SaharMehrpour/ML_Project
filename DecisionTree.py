import collections
import numpy as np
#from numpy import sign
import random

class Node:
    def __init__(self,data,data_label,unclassifiedAttributes,parentClassifier=-1,parentClassifierValue=-1,classifierThreshold=0): 
        self.data = data
        self.data_label = data_label
        self.unclassifiedAttributes = unclassifiedAttributes
        self.children = []
        self.classifier = -1
        self.leafLabel = -2
        self.parentClassifier = parentClassifier
        self.parentClassifierValue = parentClassifierValue
        
        # for discretize
        self.classifierThreshold = classifierThreshold


##########

class Discretize:
    def __init__(self,attrIndex):
        self.attrIndex = attrIndex
    
    def discretize(self,data,labels):
        dataLabels = sorted(zip(data,labels),key=lambda x: x[0][self.attrIndex])
        self.data = zip(*dataLabels)[0]
        self.labels = zip(*dataLabels)[1]
        self.infoGain = []
        for i in range(0,len(self.labels)-1):
            if self.labels[i] != self.labels[i+1]:
                self.infoGain.append([self.computeInfoGain(i),self.data[i][self.attrIndex]])
        
        result = max(self.infoGain, key=lambda x: x[0])
        return result[0],result[1] # return max info gain and best split for this attribute
                
    def computeInfoGain(self,insIndex):
        Sp = self.labels.count(1)
        Sn = len(self.labels) - Sp
        S1p = [self.labels[i] for i in range(0,insIndex+1)].count(1)
        S1n = insIndex+1-S1p
        S2p = [self.labels[i] for i in range(insIndex+1,len(self.labels))].count(1)
        S2n = len(self.labels)-insIndex-1-S2p
       
        infoGain = 0
        if Sp!=0 :
            infoGain -= float(Sp)/float(len(self.labels))*np.log2(float(Sp)/float(len(self.labels)))
        if Sn!=0:
            infoGain -= float(Sn)/float(len(self.labels))*np.log2(float(Sn)/float(len(self.labels)))
        if S1p!=0 :
            infoGain -= float(S1p)/float(S1p+S1n)*np.log2(float(S1p)/float(S1p+S1n))
        if S1n!=0:
            infoGain -= float(S1n)/float(S1p+S1n)*np.log2(float(S1n)/float(S1p+S1n))
        if S2p!=0 :
            infoGain -= float(S2p)/float(S2p+S2n)*np.log2(float(S2p)/float(S2p+S2n))
        if S2n!=0:
            infoGain -= float(S2n)/float(S2p+S2n)*np.log2(float(S2n)/float(S2p+S2n))
        
        return infoGain

##########

class ClassifierSelector:
    
    def __init__(self,node,n_attribute=float('inf')):
        self.node = node
        self.n_attribute = n_attribute
    
    def findClassifier(self):
        Entropy = 0
        #label_set = list(set(self.node.data_label)) # [1,-1] can be generalized later
        
        counter = collections.Counter(self.node.data_label)
        
        for i in range(0,len(counter.keys())):
            tempProb = float(counter.values()[i])/float(len(self.node.data_label))
            if tempProb == 0: # never happens
                continue;
            Entropy -= tempProb * (np.log(tempProb)/np.log(2))
        
        MaxInfoGain = 0; # store the maximum infoGain
        
        attr_to_check = self.node.unclassifiedAttributes if self.n_attribute == float('inf') else random.sample(self.node.unclassifiedAttributes,self.n_attribute)
        
        selectedAttr = attr_to_check[0] 

        for attr in attr_to_check:
            temp = self.infoGainCalc(Entropy, attr)

            if temp > MaxInfoGain:
                selectedAttr = attr
                MaxInfoGain = temp
                        
        return selectedAttr;
        
    def infoGainCalc(self, Entropy, attr): 
        attrValues = [ins[attr] for ins in self.node.data]
        label_set = list(set(self.node.data_label))
        attr_set = list(set(attrValues))
        attrV_label = [[a,l] for a,l in zip(attrValues,self.node.data_label)]
        
        infoGain = Entropy; 
        Sv = 0; 
        Svl = 0;

        for value in attr_set:
            Sv = attrValues.count(value) 
            if Sv==0: 
                continue
            temp = 0;
            for label in label_set:
                Svl = attrV_label.count([value,label])
                if Svl==0: 
                    continue
                temp += (float(Svl)/float(Sv))*(float(np.log(float(Svl)))/float(float(Sv)))/float(np.log(2));

            # update the info gain
            infoGain += (float(Sv)/float(len(self.node.data)))*temp;
                    
        return infoGain;
    
    def findDiscreteClassifier(self):
                
        attr_to_check = self.node.unclassifiedAttributes if self.n_attribute == np.Infinity else random.sample(self.node.unclassifiedAttributes,min(len(self.node.unclassifiedAttributes),self.n_attribute))

        selectedAttr = attr_to_check[0] 
        bestThreshold = 0
        best_infoGain = 0
        
        for attr in attr_to_check:
            Dis = Discretize(attr)
            temp_infoGain, temp_threshold = Dis.discretize(self.node.data,self.node.data_label)
            selectedAttr = attr if temp_infoGain > best_infoGain else selectedAttr
            bestThreshold = temp_threshold if temp_infoGain > best_infoGain else bestThreshold
            best_infoGain = temp_infoGain if temp_infoGain > best_infoGain else best_infoGain
        
        return selectedAttr, bestThreshold

##########

class DecisionTree:
    
    def __init__(self,max_depth=np.Infinity,n_attribute=float('inf'),discretize=False):
        self.max_depth = max_depth
        self.n_attribute = n_attribute
        self.treeDepth = 0
        self.discretize = discretize
        
    def fit(self,data,data_label):
        self.data = data
        self.data_label = data_label 
        self.rootNode = Node(data=self.data,data_label=self.data_label,unclassifiedAttributes=list(range(0,len(self.data[0]))))
        
        if self.discretize == False: 
            self.createChildNode(self.rootNode, 1)
        else:
            self.createDiscretizeChildNode(self.rootNode, 1)

    
    def createChildNode(self,node,depth):
        if depth > self.treeDepth:
            self.treeDepth = depth

        selector = ClassifierSelector(node,n_attribute=self.n_attribute)
        classifier = selector.findClassifier()
        node.classifier = classifier
                        
        #create unclassified attributes
        childUnclassifiedAttributes = node.unclassifiedAttributes
        childUnclassifiedAttributes.remove(classifier)
        
        
        classifierValueSet = list(set([d[classifier] for d in node.data])) # can be generalized later
        
        for value in classifierValueSet: 
            # data and data_label
            childData = []
            childLabel = []

            for i in range(0,len(node.data)):
                if node.data[i][classifier]==value:
                    childData.append(node.data[i]);
                    childLabel.append(node.data_label[i])
            
            if len(childData) == 0: # never happens
                continue
            
            child = Node(childData,childLabel,childUnclassifiedAttributes,parentClassifier=classifier,parentClassifierValue=value)
            
            childLabelCounter = collections.Counter(childLabel)
            if len(childLabelCounter.keys()) == 1: # only one label exists
                child.leafLabel = childLabelCounter.keys()[0]

            elif len(childUnclassifiedAttributes) >= self.n_attribute: # for discretizing
                child.leafLabel = self.mostCommonLabel(childLabel)

            elif len(childUnclassifiedAttributes) == 0: # no attr
                child.leafLabel = self.mostCommonLabel(childLabel)
            
            elif depth == self.max_depth:
                child.leafLabel = self.mostCommonLabel(childLabel)

            else:
                self.createChildNode(child, depth+1);

            node.children.append(child);
     
    def createDiscretizeChildNode(self,node,depth):  
        if depth > self.treeDepth:
            self.treeDepth = depth

        selector = ClassifierSelector(node,n_attribute=self.n_attribute)
        classifier, classifierThreshold = selector.findDiscreteClassifier() # difference
        node.classifier = classifier
        node.classifierThreshold = classifierThreshold
        
        #create children
        child1Data = []
        child1Label = []
        child2Data = []
        child2Label = []
        
        #create unclassified attributes
        childUnclassifiedAttributes = node.unclassifiedAttributes
        childUnclassifiedAttributes.remove(classifier)
        
        for i in range(0,len(node.data)):
            if node.data[i][classifier] <= classifierThreshold:
                child1Data.append(node.data[i]);
                child1Label.append(node.data_label[i])
            else:
                child2Data.append(node.data[i]);
                child2Label.append(node.data_label[i])  
        
        if len(child1Data) != 0:
            child1 = Node(data=child1Data,data_label=child1Label,unclassifiedAttributes=childUnclassifiedAttributes,parentClassifier=classifier)
            
            childLabelCounter = collections.Counter(child1Label)
            if len(childLabelCounter.keys()) == 1: # only one label exists
                child1.leafLabel = childLabelCounter.keys()[0]
    
            elif len(childUnclassifiedAttributes) == 0: # no attribute for classification
                child1.leafLabel = self.mostCommonLabel(child1Label)
    
            elif depth == self.max_depth:
                child1.leafLabel = self.mostCommonLabel(child1Label)
            
            elif len(childUnclassifiedAttributes) >= self.n_attribute: # for discretizing
                child1.leafLabel = self.mostCommonLabel(child1Label)
    
            else:
                self.createDiscretizeChildNode(child1, depth+1);
    
            node.children.append(child1);
        
        if len(child2Data) != 0:
            child2 = Node(data=child2Data,data_label=child2Label,unclassifiedAttributes=childUnclassifiedAttributes,parentClassifier=classifier)
            
            childLabelCounter = collections.Counter(child2Label)
            if len(childLabelCounter.keys()) == 1: # only one label exists
                child2.leafLabel = childLabelCounter.keys()[0]
    
            elif len(childUnclassifiedAttributes) == 0: # no attribute for classification
                child2.leafLabel = self.mostCommonLabel(child1Label)
    
            elif depth == self.max_depth:
                child2.leafLabel = self.mostCommonLabel(child1Label)
            
            elif len(childUnclassifiedAttributes) >= self.n_attribute: # for discretizing
                child2.leafLabel = self.mostCommonLabel(child2Label)
    
            else:
                self.createDiscretizeChildNode(child2, depth+1);
    
            node.children.append(child2);
            
    def mostCommonLabel(self,data_label):
        maxOccurance = 0
        maxOccuredLabel = 0 
        counter=collections.Counter(data_label)
        for i in range(0,len(counter.keys())):
            if counter.values()[i] > maxOccurance:
                maxOccurance = counter.values()[i]
                maxOccuredLabel = counter.keys()[i]
                
        return maxOccuredLabel;        
            
            
    def predict(self,testset):
        if self.discretize == True:
            return [self.traverseDiscretizeTree(instance, self.rootNode) for instance in testset]
        else:
            return [self.traverseTree(instance, self.rootNode) for instance in testset]
        
    def traverseTree(self,instance,node):        
        if node.leafLabel!=-2: 
            return node.leafLabel
        for child in node.children:
            if child.parentClassifierValue == instance[node.classifier]:
                return self.traverseTree(instance, child)
        
        return self.mostCommonLabel(node.data_label)
        #return -2; # undefined
    
    def traverseDiscretizeTree(self,instance,node):
        if node.leafLabel!=-2: 
            return node.leafLabel
        if (float(instance[node.classifier]) <= float(node.classifierThreshold)):
            return self.traverseDiscretizeTree(instance, node.children[0])
        elif len(node.children) == 2:
            return self.traverseDiscretizeTree(instance, node.children[1])
        else:
            return self.mostCommonLabel(node.data_label);
    
    
    
    