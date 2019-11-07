import numpy as np
import SGD
import Stat
#import PreProcess

###

data = []
testset = []

for i in range(5):

    rawData = open('splitted/splitted_data_' + str(i),'rb')
    temp = np.loadtxt(rawData,delimiter=',') 
    data.append(np.c_[np.ones(len(temp)),temp])
    
    rawData = open('splitted/splitted_test_' + str(i),'rb')
    temp = np.loadtxt(rawData,delimiter=',')
    testset.append(np.c_[np.ones(len(temp)),temp])
     
data_labels = [float(line.rstrip('\n')) for line in open('csvData/train_label.data')]
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
# cross-validation
  
best_r = 0.01
best_epoch = 25
          
print("best r:", best_r)
print("best epoch:", best_epoch)
  
###  

predictTrains = []
predictTests = []

dataAccuracy = []
testAccuracy = []

for i in range(5):
    sgd = SGD.SGD(r=best_r,epochs=best_epoch,W0=[0]*len(data[i][0]))
    sgd.fit(data[i],data_labels)
    
    predictTrains.append(sgd.predict(data[i]))
    predictTests.append(sgd.predict(testset[i]))
    
    dataAccuracy.append(Stat.F1_Score(sgd.predict(data[i]), data_labels))
    testAccuracy.append(Stat.F1_Score(sgd.predict(testset[i]), test_labels))

trainT = np.asarray(predictTrains).T.tolist()
testT = np.asarray(predictTests).T.tolist()

predictTrain = []
for i in range(len(data[0])):
    probPos = 0
    probNeg = 0
    for j in range(5):
        if predictTrains[j][i] == 1:
            probPos += dataAccuracy[j]
        else:
            probNeg += dataAccuracy[j]
    if probPos > probNeg:
        predictTrain.append(1)
    else:
        predictTrain.append(-1)

predictTest = []
for i in range(len(testset[0])):
    probPos = 0
    probNeg = 0
    for j in range(5):
        if predictTest == 1:
            probPos += testAccuracy[j]
        else:
            probNeg += testAccuracy[j]

    if probPos > probNeg:
        predictTest.append(1)
    else:
        predictTest.append(-1)


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
filename = 'result/SGD/weight' + '_best_r_' + str(best_r) + '_best_epoch' + str(best_epoch)

with open(filename, 'wb') as thefile:
    for item in sgd.W:
        thefile.write("%s\n" % item)
     
print("Weight is written!")

### Stat

filename = 'result/SGD/stat' + '_best_r_' + str(best_r) + '_best_epoch' + str(best_epoch)

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

filename = 'result/SGD/prediction' + '_best_r_' + str(best_r) + '_best_epoch' + str(best_epoch)

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

filename = 'result/SGD/eval' + '_best_r_' + str(best_r) + '_best_epoch' + str(best_epoch)

with open(filename, 'wb') as thefile:
    for item in predictEval:
        thefile.write("%s\n" % item)
     
print("eval is written!")
'''