
import numpy as np
import AdaBoost
import Stat
import PreProcess

rawData = open('csvData/train.data','rb')
temp = np.loadtxt(rawData,delimiter=',') 
data = np.c_[np.ones(len(temp)),temp]
#data = temp
     
data_labels = [float(line.rstrip('\n')) for line in open('csvData/train_label.data')]

rawData = open('csvData/test.data','rb')
temp = np.loadtxt(rawData,delimiter=',')
testset = np.c_[np.ones(len(temp)),temp]
#testset = temp
        
test_labels = [float(line.rstrip('\n')) for line in open('csvData/test_label.data')]


# pre-process

PP = PreProcess.PreProcess(data,n_buckets=10,func='boolean')#,swap_labels=True)
data = PP.fit(data)
testset = PP.fit(testset) 

# read weights

weights = []

for w in range(6):
    tmp = [float(line.rstrip('\n')) for line in open('result/W' + str(w))]
    if len(tmp) != len(data[0]):
        continue
    weights.append(tmp)


ab = AdaBoost.AdaBoost(weightVector=weights)
ab.fit(data, data_labels)

predictTrain = ab.predict(data)
predictTest = ab.predict(testset)
 
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

### Stat
'''
filename = 'result/AdaBoost/stat' 

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

filename = 'result/AdaBoost/prediction'

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

predictEval = ab.predict(evalset)

filename = 'result/AdaBoost/eval'

with open(filename, 'wb') as thefile:
    for item in predictEval:
        thefile.write("%s\n" % item)
     
print("eval is written!")
'''