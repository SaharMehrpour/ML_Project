import numpy as np

class AdaBoost:
    
    def __init__(self,weightVector):
        self.RULES = weightVector
        self.ALPHA = []
        
    def fit(self,data,labels):
        self.data = data
        self.labels = labels
        self.N = len(data)
        self.weights = [1.0/float(self.N)]*self.N
        
        for i in range(len(data)):
            self.labels[i] = float(labels[i])
            for j in range(len(data[i])):
                self.data[i][j] = float(data[i][j])
        
        usedRULE = [0]*len(self.RULES)
        self.used_RULES = []
        
        cnt = 1
        
        for _ in range(len(self.RULES)):
            
            print(cnt)
            cnt += 1
            
            best_RULE = 0
            best_error = 1
            RULE_index = 0
            
            for index in range(len(self.RULES)):
                if(usedRULE[index] == 1):
                    continue
                errorT = 0.5
                for i in range(0,self.N):
                    errorT -= float(0.5*self.weights[i]*float(self.labels[i])*np.sign(float(np.dot(self.RULES[index],self.data[i]))))
                if errorT < best_error:
                    best_error = errorT
                    best_RULE = self.RULES[index]
                    RULE_index = index
                            
            usedRULE[RULE_index] = 1
            alphaT = 0.5*np.exp((1.0-best_error)/best_error)
            self.ALPHA.append(alphaT)
            
            print(RULE_index,best_error,alphaT)
            
            self.used_RULES.append(best_RULE)
            self.ALPHA.append(alphaT)
                
            w = [self.weights[i]*np.exp(-1.0*alphaT*float(self.labels[i])*np.sign(float(np.dot(best_RULE,self.data[i])))) for i in range(self.N)]            
            if(sum(w) == 0 or sum(w) == float('inf')):
                break            
            self.weights = [x/sum(w) for x in w]
                    
                    
    def predict(self,test):   
        predict = []
        for d in test:
            hx = [self.ALPHA[i]*np.sign(np.dot(self.used_RULES[i],d)) for i in range(len(self.used_RULES))]
            sgn = np.sign(sum(hx)) if np.sign(sum(hx))!= 0 else -1
            predict.append(sgn)
        return predict
        return predict