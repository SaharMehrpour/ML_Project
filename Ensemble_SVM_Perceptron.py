
import Stat
  
predict_labels_Perceptron = [float(line.rstrip('\n')) for line in open('results/test_Perceptron.data')]
eval_labels_Perceptron = [float(line.rstrip('\n')) for line in open('results/eval_Perceptron.data')]
        
predict_labels_SVM = [float(line.rstrip('\n')) for line in open('results/test_SVM.data')]
eval_labels_SVM = [float(line.rstrip('\n')) for line in open('results/eval_Perceptron.data')]

test_labels = [float(line.rstrip('\n')) for line in open('csvData/test_label.data')]

filename = 'ensembleResult/test' + '_ensemble_SVM_Perceptron'

predictTest = [1 if s==1 and p==1 else -1 for p,s in zip(predict_labels_Perceptron,predict_labels_SVM)]
predictEval = [1 if s==1 and p==1 else -1 for p,s in zip(eval_labels_Perceptron,eval_labels_SVM)]
   
print("Accuracy for test set")
print(Stat.Accuracy(predictTest, test_labels))
   
print("F1 score for test set")
print(Stat.F1_Score(predictTest, test_labels))
    
print("Precision for test set")
print(Stat.Precision(predictTest, test_labels))
 
print("Recall for test set")
print(Stat.Recall(predictTest, test_labels))

# filename = 'results/eval_Ensemble_SVM_Perceptron'
# 
# with open(filename, 'wb') as thefile:
#     for item in predictEval:
#         thefile.write("%s\n" % item)
#      
# print("eval is written!")