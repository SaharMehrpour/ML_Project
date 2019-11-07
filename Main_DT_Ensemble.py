import numpy as np
import SGD_SVM
import RandomForest
import PreProcess
import KFold
import Stat
  
f = open('csvData/train.data')
temp = []
for line in f:
    temp.append(line.rstrip().split(','))
data = np.c_[np.ones(len(temp)),temp]
     
data_labels = [float(line.rstrip('\n')) for line in open('csvData/train_label.data')]
   
f = open('csvData/test.data')
temp = []
for line in f:
    temp.append(line.rstrip().split(','))
testset = np.c_[np.ones(len(temp)),temp]
        
test_labels = [float(line.rstrip('\n')) for line in open('csvData/test_label.data')]
  
# pre-process
 
PP = PreProcess.PreProcess(data,n_buckets=5)#,func='boolean',swap_labels=True)
data = PP.fit(data)
testset = PP.fit(testset)

data_labels = PP.processLabels(data_labels) 
test_labels = PP.processLabels(test_labels)
 
###

g_best_N = 10
g_best_accuracy = 0
g_best_C = 2
g_best_ro = 0.01
g_best_accuracy = 0
g_best_epoch = 10
g_best_g0 = 1.1

# cross-validation

for N in [10,20,30,40,50]:
    
    print("N",N)
    
    tmp = []
          
    rf = RandomForest.RandomForest(n_trees=N,n_samples=int(0.5*len(data)),n_attribute=8,discretize=False) # True)
    kfold = KFold.KFold(n_splits=5)
          
    for kf in kfold.split(data): 
        train = [data[i] for i in kf[0]]
        train_label = [data_labels[i] for i in kf[0]]
        test = [data[i] for i in kf[1]]
        test_label = [data_labels[i] for i in kf[1]]
                                  
        rf.fit(train, train_label)
        
        phiTrain = rf.predict(train) 
        phiTest = rf.predict(test)
        
        # cross-validation
        
        best_C = 2
        best_ro = 0.01
        best_accuracy = 0
        best_epoch = 10
        best_g0 = 1.1
  
        for C in [4,2,0.5,0.25,0.125]:#,0.0625,0.03125]:
            for ro in [0.001]:#,0.02,0.03]:
                for epoch in [20]:#,10,15,25]:
                    for g0 in [1.1,1.01,1.001]:
                        tmp = []
                  
                        sgd = SGD_SVM.SGDSVM(C=C,ro=ro,epochs=epoch,W0=[0]*len(phiTrain[0]),gamma0=g0)
                        kfold = KFold.KFold(n_splits=5)
                  
                        for kf in kfold.split(phiTrain): 
                            train2 = [phiTrain[i] for i in kf[0]]
                            train_label2 = [data_labels[i] for i in kf[0]]
                            test2 = [phiTrain[i] for i in kf[1]]
                            test_label2 = [data_labels[i] for i in kf[1]]
                                          
                            sgd.fit(train2, train_label2)
                            predict_tmp = sgd.predict(test2)
                            tmp.append(Stat.Accuracy(predict_tmp,test_label2))
                          
                        if np.mean(tmp) > best_accuracy:
                            best_accuracy = np.mean(tmp)
                            best_C = C
                            best_epoch = epoch
                            best_g0 = g0
                            best_ro = ro
        
        ###
        
        print("mid cross-validation")
        
        sgd = SGD_SVM.SGDSVM(C=best_C,ro=best_ro,epochs=best_epoch,W0=[0]*len(phiTrain[0]),gamma0=best_g0)
        sgd.fit(phiTrain,train_label)
        
        predict_tmp = sgd.predict(phiTest)
        tmp.append(Stat.Accuracy(predict_tmp,test_label))
                  
    if np.mean(tmp) > best_accuracy:
        g_best_accuracy = np.mean(tmp)
        g_best_N = N
        g_best_C = best_C
        g_best_epoch = best_epoch
        g_best_g0 = best_g0
        g_best_ro = best_ro
                                          
        print("Best result so far >>",best_accuracy,N)
          
print("best N",g_best_N)
    
rf = RandomForest.RandomForest(n_trees=g_best_N,n_samples=int(1*len(data)),n_attribute=8,discretize=False) # True)
rf.fit(data, data_labels)
  
print("Random Forest is built")
   
phiTrain = rf.predict(data) 
phiTest = rf.predict(testset)
              
sgd = SGD_SVM.SGDSVM(C=g_best_C,ro=g_best_ro,epochs=g_best_epoch,W0=[0]*len(phiTrain[0]),gamma0=g_best_g0)
sgd.fit(phiTrain,data_labels)
          
predictTrain = sgd.predict(phiTrain)
predictTest = sgd.predict(phiTest)

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

filename = 'result/DT/stat' + '_bestN_' + str(g_best_N)

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

filename = 'result/DT/prediction' + '_bestN_' + str(g_best_N)

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

phiEval = rf.predict(evalset)
predictEval = sgd.predict(phiEval)

##
predictEval = PP.processLabels(predictEval)
##

filename = 'result/DT/eval' + '_bestN_' + str(g_best_N)

with open(filename, 'wb') as thefile:
    for item in predictEval:
        thefile.write("%s\n" % item)
     
print("eval is written!")
'''