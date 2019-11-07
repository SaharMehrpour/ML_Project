import numpy as np
import Stat
import Winnow
import PreProcess
import KFold

###

#data,data_labels=load_svmlight_file('data/data.train')
#testset,test_labels=load_svmlight_file('data/data.eval.anon')

#data= data.toarray()
#testset=testset.toarray()

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
'''
PP = PreProcess.PreProcess(data,n_buckets=5)#,func='boolean',swap_labels=True)
data = PP.fit(data)
testset = PP.fit(testset) 

data_labels = PP.processLabels(data_labels) 
test_labels = PP.processLabels(test_labels)
'''
###

# cross-validation
  
best_bucket = 10
best_param = 2
best_f1 = 0 
''' 
for b in [2,3,4,5,6,10]:
    
    temp_PP = PreProcess.PreProcess(data,n_buckets=b)
    data_tmp = temp_PP.makeBoolean(data)
    
    for p in [1.01,1.1,1.5,2]:
        
        tmp = []  
        win = Winnow.Winnow(balanced=True,param=p)
        kfold = KFold.KFold(n_splits=5)
              
        for kf in kfold.split(data_tmp): 
            train = [data_tmp[i] for i in kf[0]]
            train_label = [data_labels[i] for i in kf[0]]
            test = [data_tmp[i] for i in kf[1]]
            test_label = [data_labels[i] for i in kf[1]]
                                      
            win.fit(train, train_label)
            predict_tmp = win.predict(test)
            tmp.append(Stat.F1_Score(predict_tmp,test_label))
                      
        if np.mean(tmp) > best_f1:
            best_f1 = np.mean(tmp)
            best_bucket = b
            best_param = p
                                              
            print("Best result so far >>",best_f1,best_bucket,best_param)
          
print("best bucket:", best_bucket)
print("best param:" , best_param)
'''  
###  

PP = PreProcess.PreProcess(data,n_buckets=best_bucket,func='boolean')

data = PP.fit(data)
testset = PP.fit(testset)

balanced = Winnow.Winnow(balanced=True,param=best_param)
balanced.fit(data,data_labels)
   
predictTrain = balanced.predict(data)
predictTest = balanced.predict(testset)
 
print("balanced:") 
 
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

filename = 'result/Winnow/weight_balanced_best_bucket_' + str(best_bucket) + '_best_param_' + str(best_param) 

with open(filename, 'wb') as thefile:
    for item in balanced.W:
        thefile.write("%s\n" % item)
     
print("Weight is written!")

### Stat
'''
filename = 'result/Winnow/stat_balanced_best_bucket_' + str(best_bucket) + '_best_param_' + str(best_param) 

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

filename = 'result/Winnow/prediction_balanced_best_bucket_' + str(best_bucket) + '_best_param_' + str(best_param) 

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
#evalset = temp

##
evalset = PP.fit(evalset)
##

predictEval = balanced.predict(evalset)

##
predictEval = PP.processLabels(predictEval)
##

filename = 'result/Winnow/eval_balanced_best_bucket_' + str(best_bucket) + '_best_param_' + str(best_param)  

with open(filename, 'wb') as thefile:
    for item in predictEval:
        thefile.write("%s\n" % item)
     
print("eval is written!")
'''
