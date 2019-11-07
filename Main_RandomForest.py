import numpy as np
import RandomForest
import PreProcess
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
  
# pre-process
 
PP = PreProcess.PreProcess(data,n_buckets=5)
data = PP.bucket(data)
testset = PP.bucket(testset) 
 
###

best_N = 10
best_accuracy = 0

# cross-validation

for N in [10,20,30,40,50]:    
    tmp = []
          
    rf = RandomForest.RandomForest(n_trees=N,n_samples=int(0.5*len(data)),n_attribute=8,discretize=False)
    kfold = KFold.KFold(n_splits=5)
          
    for kf in kfold.split(data): 
        train = [data[i] for i in kf[0]]
        train_label = [data_labels[i] for i in kf[0]]
        test = [data[i] for i in kf[1]]
        test_label = [data_labels[i] for i in kf[1]]
                                  
        rf.fit(train, train_label)
        predict_tmp = [1 if l.count(1) > N/2 else -1 for l in rf.predict(test)]
        tmp.append(Stat.Accuracy(predict_tmp,test_label))
                  
    if np.mean(tmp) > best_accuracy:
        best_accuracy = np.mean(tmp)
        best_N = N
                    
        print("Best result so far >>",best_accuracy,N)
          
print("best N",best_N)

###
    
rf = RandomForest.RandomForest(n_trees=best_N,n_samples=int(0.5*len(data)),n_attribute=8,discretize=False)
rf.fit(data, data_labels)
  
print("Random Forest is built")
   
predictTrain = [1 if l.count(1) > best_N/2 else -1 for l in rf.predict(data)]
predictTest = [1 if l.count(1) > best_N/2 else -1 for l in rf.predict(testset)]
  
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
print("Weight vector is useless!")

### Stat

filename = 'result/Forest/stat' + '_bestN_' + str(best_N)

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

filename = 'result/Forest/prediction' + '_bestN_' + str(best_N)

with open(filename, 'wb') as thefile:
    for item in predictTest:
        thefile.write("%s\n" % item)
     
print("File is written!")

### eval

f = open('csvData/data_eval_anon')
temp = []
for line in f:
    temp.append(line.rstrip().split(','))
evalset = np.c_[np.ones(len(temp)),temp]

##
evalset = PP.bucket(evalset)
##

predictEval = [1 if l.count(1) > best_N/2 else -1 for l in rf.predict(evalset)]

filename = 'result/Forest/eval' + '_bestN_' + str(best_N)

with open(filename, 'wb') as thefile:
    for item in predictEval:
        thefile.write("%s\n" % item)
     
print("eval is written!")
'''