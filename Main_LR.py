import numpy as np
import LR_SGD
import KFold
import Stat
import PreProcess
#import matplotlib.pyplot as plt

rawData = open('csvData/train.data','rb')
temp = np.loadtxt(rawData,delimiter=',') 
data = np.c_[np.ones(len(temp)),temp]
     
data_labels = [float(line.rstrip('\n')) for line in open('csvData/train_label.data')]
  
rawData = open('csvData/test.data','rb')
temp = np.loadtxt(rawData,delimiter=',')
testset = np.c_[np.ones(len(temp)),temp]
        
test_labels = [float(line.rstrip('\n')) for line in open('csvData/test_label.data')]
###

# pre-process
 
PP = PreProcess.PreProcess(data,n_buckets=10,func='boolean')#,swap_labels=True)
data = PP.fit(data)
testset = PP.fit(testset) 

data_labels = PP.processLabels(data_labels) 
test_labels
  
# cross-validation
  
best_g0 = 1.001
best_C = 2
best_accuracy = 0
best_epoch = 10
best_sigma = 100

'''
for C in [2,0.5,0.25]:
    for g0 in [1.1,1.01,1.001]:
        for epoch in [10,15]:
            for sigma in [0.9,1,10,50,100]:
                tmp = []
                
                print("CV:",C,g0,epoch,sigma)
                  
                sgd = LR_SGD.LR_SGD(gamma0=g0,epochs=epoch,W0=[0]*len(data[0]),sigma=sigma,C=C)
                kfold = KFold.KFold(n_splits=5)
                  
                for kf in kfold.split(data): 
                    train = [data[i] for i in kf[0]]
                    train_label = [data_labels[i] for i in kf[0]]
                    test = [data[i] for i in kf[1]]
                    test_label = [data_labels[i] for i in kf[1]]
                                          
                    sgd.fit(train, train_label)
                    predict_tmp = sgd.predict(test)
                    tmp.append(Stat.Accuracy(predict_tmp,test_label))
                          
                if np.mean(tmp) > best_accuracy:
                    best_accuracy = np.mean(tmp)
                    best_epoch = epoch
                    best_g0 = g0
                    best_sigma = sigma
                    best_C = C
                                                  
                    print("Best result so far >>",best_accuracy,C,g0,epoch,sigma)
           
print("best g0:", best_g0)
print("best epoch:", best_epoch)
print("best sigma:", best_sigma)
print("best C: ", best_C)
'''
###  

sgd = LR_SGD.LR_SGD(gamma0=best_g0,epochs=best_epoch,W0=[0]*len(data[0]),sigma=best_sigma,C=best_C)
sgd.fit(data,data_labels)
   
predictTrain = sgd.predict(data)
predictTest = sgd.predict(testset)
 
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

### weight vector
'''
filename = 'result/LR/weight' + '_best_r_' + str(best_g0) + '_best_epoch_' + str(best_epoch) + '_best_sigma_' + str(best_sigma) + '_best_C_' + str(best_C) + '_bucket'

with open(filename, 'wb') as thefile:
    for item in sgd.W:
        thefile.write("%s\n" % item)
     
print("Weight is written!")

### Stat

filename = 'result/LR/stat' + '_best_r_' + str(best_g0) + '_best_epoch_' + str(best_epoch) + '_best_sigma_' + str(best_sigma) + '_best_C_' + str(best_C) + '_bucket'

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

filename = 'result/LR/prediction' + '_best_r_' + str(best_g0) + '_best_epoch_' + str(best_epoch) + '_best_sigma_' + str(best_sigma) + '_best_C_' + str(best_C) + '_bucket'

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
evalset = PP.fit(evalset)
##

predictEval = sgd.predict(evalset)

##
predictEval = PP.processLabels(predictEval)
##

filename = 'result/LR/eval' + '_best_r_' + str(best_g0) + '_best_epoch_' + str(best_epoch) + '_best_sigma_' + str(best_sigma) + '_best_C_' + str(best_C) + '_bucket'

with open(filename, 'wb') as thefile:
    for item in predictEval:
        thefile.write("%s\n" % item)
     
print("eval is written!")
'''