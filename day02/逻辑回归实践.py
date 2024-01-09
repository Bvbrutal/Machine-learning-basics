#import necessray libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from Logistic_Regression import LogisticRegession as LR
from sklearn.linear_model import LogisticRegression as LR_SKL
class LogisticRegession:
#define sigmoid function
    def sigmoid(self,x):
        y_prob=1.0/(1.0+np.exp(-x))
        return y_prob
#define prediction function
    def predict_prob(self,x):
        y_prob = self.sigmoid(np.dot(x,self.w)+self.b) # see Eq.(2.7)
        return y_prob
#define prediction function
    def predict(self,X):
        inst_num = X.shape[0]
        probs = self.predict_prob(X)
        labels = np.zeros(inst_num)
        for i in range(inst_num):
            if probs[i] >=0.5:
                labels[i] = 1
        return probs,labels
#define loss function
    def loss_function(self,train_x,train_y):
        inst_num = train_x.shape[0]
        loss = 0.0
        for i in range(inst_num):
            z = np.dot(train_x[i,:],self.w)+self.b
            loss += -train_y[i]*z + np.log(1+np.exp(z)) #see Eq.(2.10)
        loss = loss / inst_num
        return loss
#define gradient calculation function
    def calculate_grad(self,train_x,train_y):
        inst_num = train_x.shape[0] # data size
        probs = self.sigmoid(train_x.dot(self.w) + self.b) # training prediction
        #Add code here to calculate grad of weights, see Eq.(2.11)
        # Add code here to calculate grad of bias, see Eq.(2.12)
        grad_w = (train_x.T).dot((probs-train_y))/inst_num 
        grad_b = np.sum((probs-train_y))/inst_num
        return grad_w,grad_b
# gradient descent algorithm
    def gradient_descent(self,train_x,train_y,learn_rate,max_iter, epsilon):
        loss_list = []
        for i in range(max_iter):
            loss_old = self.loss_function(train_x, train_y)
            loss_list.append(loss_old)
            grad_w,grad_b = self.calculate_grad(train_x, train_y)
            self.w = self.w - learn_rate*grad_w
            self.b = self.b - learn_rate*grad_b
            loss_new = self.loss_function(train_x, train_y)
            if abs(loss_new-loss_old) <= epsilon:
                break
        return loss_list
# learning linear regression model
    def fit(self,train_x,train_y,learn_rate,max_iter, epsilon):
        feat_num = train_x.shape[1] # feature dimension
        self.w = np.zeros((feat_num,1)) # initialize model parameters
        self.b = 0.0
#learn model parameters using gradient descent algorithm
        loss_list = self.gradient_descent(train_x, train_y, learn_rate, max_iter, epsilon)
        self.training_visualization(loss_list)
# learning process visualization
    def training_visualization(self,loss_list):
        plt.plot(loss_list, color="red")
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.savefig("loss.png", bbox_inches = "tight",dpi=400)
        plt.show()
# load experiment data
f = open('D:\Documents\机器学习\《机器学习基础实践》课程资源\数据集\Stock_Client_loss.csv')
data = pd.read_csv(f)
data_x = data[["x1","x2","x3","x4","x5"]]
data_y = np.array(data["loss"])
# data normalization
scaler = StandardScaler()
data_x = scaler.fit_transform(data_x)
# divide data into train/test, 70% for train, 30% for test
X_train, X_test, Y_train, Y_test = train_test_split(data_x,
data_y,
test_size=0.3,
shuffle=True)
# set training parameters and define eval metric
learnrate = 0.01
maxiter = 1000
eps = 1e-5
def cal_acc(y_test,y_pred):
    acc=0.0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            acc += 1.0
    return acc/len(y_test)
# build prediction model using LR
LR_model = LogisticRegession()
LR_model.fit(train_x=X_train,train_y=Y_train.reshape(-1,1),
learn_rate=learnrate,max_iter=maxiter, epsilon=eps)
_,Y_test_pred_LR = LR_model.predict(X_test)
acc = cal_acc(Y_test,Y_test_pred_LR)
print("LR ACC:%.3f"%(acc))
# build prediction model using LR_SKL
LR_SKL_model = LR_SKL()
LR_SKL_model.fit(X_train,Y_train)
Y_test_pred_LR_SKL = LR_SKL_model.predict(X_test)
acc = cal_acc(Y_test,Y_test_pred_LR_SKL)
print("LR_SKL ACC:%.3f"%(acc))