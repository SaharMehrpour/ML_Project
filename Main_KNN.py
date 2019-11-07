#import numpy as np
import PreProcess
import KNN
#import KFold
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
 
PP = PreProcess.PreProcess(data,func='boolean',n_buckets=5)
data = PP.fit(data)
testset = PP.fit(testset) 
'''
data = [[0,0,1,1],[0,0,1,0],[0,0,0,0],[1,0,1,0],[1,0,1,1],[1,1,1,1]]
data_labels = [0,1,2,3,4,5]
testset = [[1,0,1,1]]
'''
# cross-validation
   
best_norm = 2
best_K = 3
best_f1 = 0  
'''
for k in [1,3,5,7]:
    for p in [1,2,3]:
        print("CV",k,p)
        tmp = []
                
        knn = KNN.KNN(norm=p,K=k)
        kfold = KFold.KFold(n_splits=5)
                
        for kf in kfold.split(data): 
            train = [data[i] for i in kf[0]]
            train_label = [data_labels[i] for i in kf[0]]
            test = [data[i] for i in kf[1]]
            test_label = [data_labels[i] for i in kf[1]]
                                        
            knn.fit(train, train_label)
            predict_tmp = knn.predict(test)
            tmp.append(Stat.F1_Score(predict_tmp,test_label))
                        
        if np.mean(tmp) > best_f1:
            best_f1 = np.mean(tmp)
            best_norm = p
            best_K = k
                                                
            print("Best result so far >>",best_f1,p,k)
'''            
print("best norm", best_norm)
print("best K", best_K)
    
###  
 
knn = KNN.KNN(norm=best_norm,K=best_K) 
knn.fit(data, data_labels)
  
predictTrain = knn.predict(data)
predictTest = knn.predict(testset)
'''  
print("Accuracy for training set:")
print(Stat.Accuracy(predictTrain, data_labels))

print("F1 score for training set:")
print(Stat.F1_Score(predictTrain, data_labels))
 
print("Precision for training set:")
print(Stat.Precision(predictTrain, data_labels))
 
print("Recall for training set:")
print(Stat.Recall(predictTrain, data_labels))
'''  
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
  
filename = 'result/KNN/stat' + '_best_norm_' + str(best_norm) + '_best_K_' + str(best_K)
  
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
  
  
filename = 'result/KNN/prediction' + '_best_norm' + str(best_norm) + '_best_K_' + str(best_K)
  
with open(filename, 'wb') as thefile:
    for item in predictTest:
        thefile.write("%s\n" % item)
       
print("File is written!")
''' 
### eval
 
f = open('csvData/data_eval_anon')
temp = []
for line in f:
    temp.append(line.rstrip().split(','))
evalset = temp
evalset = PP.fit(evalset)
 
predictEval = knn.predict(evalset)

filename = 'result/KNN/eval' + '_best_norm_' + str(best_norm) + '_best_K_' + str(best_K)
 
with open(filename, 'wb') as thefile:
    for item in predictEval:
        thefile.write("%s\n" % item)
      
print("eval is written!")
