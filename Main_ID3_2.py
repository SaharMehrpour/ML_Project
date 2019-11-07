import numpy as np
import PreProcess
import DecisionTree
import KFold
import Stat
  
f = open('csvData/train.data')
temp = []
for line in f:
    temp.append(line.rstrip().split(','))
data = temp

data_labels = [float(line.rstrip('\n')) for line in open('csvData/train_label.data')]
   
f = open('csvData/test.data')
temp = []
for line in f:
    temp.append(line.rstrip().split(','))
testset = temp
        
test_labels = [float(line.rstrip('\n')) for line in open('csvData/test_label.data')]
###

# pre-process
 
PP = PreProcess.PreProcess(data,n_buckets=5)#,func='boolean',swap_labels=True)
data = PP.fit(data)
testset = PP.fit(testset) 
  
# cross-validation
   
best_depth = 3
best_f1 = 0  
for d in [10,20,30,40,50,float('inf')]:
    tmp = []
            
    dt = DecisionTree.DecisionTree(n_attribute=8,discretize=True,max_depth=d)
    kfold = KFold.KFold(n_splits=5)
            
    for kf in kfold.split(data): 
        train = [data[i] for i in kf[0]]
        train_label = [data_labels[i] for i in kf[0]]
        test = [data[i] for i in kf[1]]
        test_label = [data_labels[i] for i in kf[1]]
                                    
        dt.fit(train, train_label)
        predict_tmp = dt.predict(test)
        tmp.append(Stat.F1_Score(predict_tmp,test_label))
                    
    if np.mean(tmp) > best_f1:
        best_f1 = np.mean(tmp)
        best_depth = d
                                            
        print("Best result so far >>",best_f1,d)
            
print("best depth:", best_depth)
    
###  
 
dt = DecisionTree.DecisionTree(max_depth=best_depth,n_attribute=8,discretize=True) 
dt.fit(data, data_labels)
  
predictTrain = dt.predict(data)
predictTest = dt.predict(testset)
  
print("Accuracy for training set:")
print(Stat.Accuracy(predictTrain, data_labels))

print("F1 score for training set:")
print(Stat.F1_Score(predictTrain, data_labels))
 
print("Precision for training set:")
print(Stat.Precision(predictTrain, data_labels))
 
print("Recall for training set:")
print(Stat.Recall(predictTrain, data_labels))
   
print("Accuracy for test set")
print(Stat.Accuracy(predictTest, test_labels))
   
print("F1 score for test set")
print(Stat.F1_Score(predictTest, test_labels))
    
print("Precision for test set")
print(Stat.Precision(predictTest, test_labels))
 
print("Recall for test set")
print(Stat.Recall(predictTest, test_labels))

### WeightVector
'''  
print("There is no weight vector!")
  
### Stat
  
filename = 'result/ID3_2/stat' + '_best_r_' + str(best_depth)
  
with open(filename, 'wb') as thefile:
    thefile.write("Training:")
    thefile.write("Accuracy %s\n" % Stat.Accuracy(predictTrain, data_labels))
    thefile.write("F1-score %s\n" % Stat.F1_Score(predictTrain, data_labels))
    thefile.write("Precision %s\n" % Stat.Precision(predictTrain, data_labels))
    thefile.write("Recall %s\n" % Stat.Recall(predictTrain, data_labels))
    thefile.write("Test:")
    thefile.write("Accuracy %s\n" % Stat.Accuracy(predictTest, test_labels))
    thefile.write("F1-score %s\n" % Stat.F1_Score(predictTest, test_labels))
    thefile.write("Precision %s\n" % Stat.Precision(predictTest, test_labels))
    thefile.write("Recall %s\n" % Stat.Recall(predictTest, test_labels))
       
print("Stat is written!")
  
  
filename = 'result/ID3_2/prediction' + '_bestDepth_' + str(best_depth)
  
with open(filename, 'wb') as thefile:
    for item in predictTest:
        thefile.write("%s\n" % item)
       
print("File is written!")
 
### eval
 
f = open('csvData/data_eval_anon')
temp = []
for line in f:
    temp.append(line.rstrip().split(','))
evalset = temp
 
predictEval = dt.predict(evalset)

filename = 'result/ID3_2/eval' + '_bestDepth_' + str(best_depth)
 
with open(filename, 'wb') as thefile:
    for item in predictEval:
        thefile.write("%s\n" % item)
      
print("eval is written!")
'''