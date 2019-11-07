# Compute Accuracy
def Accuracy(predict,test_label):
    correct = 0
    for i in range(0,len(predict)):
        if(predict[i]==test_label[i]):
            correct+=1.0
    return float(correct)/len(predict)
          
          
# F1.0 score
def F1_Score(predict,labels):
    TP = 0
    FP = 0
    FN = 0
    for i in range(0,len(predict)):
        if predict[i]==labels[i]==1.0:
            TP +=1.0
        if labels[i]==1.0 and predict[i]!=1.0:
            FN +=1.0
        if labels[i]!=1.0 and predict[i]==1.0:
            FP +=1.0
    
    if ((TP+FP) == 0): return float('inf')
    if ((TP+FN) == 0): return float('inf')
    
    precision = float(TP)/float(TP+FP)
    recall = float(TP)/float(TP+FN)
    
    return 2*(precision*recall)/(precision+recall)

# Precision
def Precision(predict,labels):
    TP = 0
    FP = 0
    for i in range(0,len(predict)):
        if predict[i]==labels[i]==1.0:
            TP +=1.0
        if labels[i]!=1.0 and predict[i]==1.0:
            FP +=1.0
    
    if ((TP+FP) == 0): return float('inf')
    return float(TP)/float(TP+FP)

# Recall
def Recall(predict,labels):
    TP = 0
    FN = 0
    for i in range(0,len(predict)):
        if predict[i]==labels[i]==1.0:
            TP +=1.0
        if labels[i]==1.0 and predict[i]!=1.0:
            FN +=1.0
    
    if ((TP+FN) == 0): return float('inf')
    return float(TP)/float(TP+FN)

