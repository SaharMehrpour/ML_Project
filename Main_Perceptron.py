import numpy as np
import KFold
import Perceptron
import Stat
import PreProcess

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
 
PP = PreProcess.PreProcess(data,n_buckets=10,func='boolean')#,swap_labels=True)
data = PP.fit(data)
testset = PP.fit(testset)

#data_labels = PP.processLabels(data_labels) 
#test_labels = PP.processLabels(test_labels)
 
###

# cross-validation
  
best_r = 0.001
best_epoch = 25
best_f1 = 0 
''' 
for epoch in [1,2,5,10,20]:
    for r in [0.001,0.01]:#,0.02,0.03]:
        tmp = []
              
        sgd = Perceptron.Perceptron(r=r,W0=[0]*len(data[0]),epoch=epoch)
        kfold = KFold.KFold(n_splits=5)
              
        for kf in kfold.split(data): 
            train = [data[i] for i in kf[0]]
            train_label = [data_labels[i] for i in kf[0]]
            test = [data[i] for i in kf[1]]
            test_label = [data_labels[i] for i in kf[1]]
                                      
            sgd.fit(train, train_label)
            predict_tmp = sgd.predict(test)
            tmp.append(Stat.F1_Score(predict_tmp,test_label))
                      
            if np.mean(tmp) > best_f1:
                best_f1 = np.mean(tmp)
                best_r = r
                best_epoch = epoch
                                              
                print("Best result so far >>",best_f1,r,epoch)
          
print("best r:", best_r)
print("best epoch:" , best_epoch)
'''  
###  

sgd = Perceptron.Perceptron(r=best_r,W0=[0]*len(data[0]),epoch=best_epoch)
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

filename = 'result/Perceptron/weight' + '_best_r_' + str(best_r) + '_best_epoch' + str(best_epoch)

with open(filename, 'wb') as thefile:
    for item in sgd.W:
        thefile.write("%s\n" % item)
     
print("Weight is written!")
'''
### Stat

filename = 'result/Perceptron/stat' + '_best_r_' + str(best_r) + '_best_epoch' + str(best_epoch)

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

filename = 'result/Perceptron/prediction' + '_best_r_' + str(best_r) + '_best_epoch' + str(best_epoch)

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

filename = 'result/Perceptron/eval' + '_best_r_' + str(best_r) + '_best_epoch' + str(best_epoch)

with open(filename, 'wb') as thefile:
    for item in predictEval:
        thefile.write("%s\n" % item)
     
print("eval is written!")

'''