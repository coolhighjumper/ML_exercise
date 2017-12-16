'''
@author: Chiang
'''
import numpy as np
from math import sqrt

class MLP(object):

    def __init__(self, eta=0.05, n_iter=10,n_hidden=2):
        
        self.n_iter = n_iter
        self.eta = eta
        self.n_hidden=n_hidden

    def fit(self, x_modify, y):
        self.errors_ = []

        self.w_1=np.random.rand(x_modify.shape[1]+1,self.n_hidden)-0.5
        self.w_2=np.random.rand(self.n_hidden+1,1)-0.5
        for _ in range(self.n_iter):
            count=0
            for i in range(x_modify.shape[0]):
                data_input=np.append(x_modify[i],1)
                hidden_input=np.dot(np.array([data_input]),self.w_1)
                hidden_output=1/(1+np.exp(-hidden_input))
                hidden_output=np.append(hidden_output,1)
                output=1/(1+np.exp(-np.dot(hidden_output,self.w_2)))
                delta_2=(y[i]-output)*output*(1-output)[0]
                grad_ho=np.array([delta_2*hidden_output])
                delta_1=np.dot(np.transpose(delta_2*self.w_2),hidden_output*(1-hidden_output))[0]
                grad_ih=delta_1*np.array([data_input])
                self.w_1=self.w_1+self.eta*np.transpose(grad_ih)
                self.w_2=self.w_2+self.eta*np.transpose(grad_ho)
            
    
            for i in range(x_modify.shape[0]):
                data_input=np.append(x_modify[i],1)
                hidden_input=np.dot(np.array([data_input]),self.w_1)
                hidden_output=1/(1+np.exp(-hidden_input))
                hidden_output=np.append(hidden_output,1)
                output=1/(1+np.exp(-np.dot(hidden_output,self.w_2)))
                #print(output)
                if y[i]!=np.round(output):
                    count+=1

            self.errors_.append(count)
        return self

