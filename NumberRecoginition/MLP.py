'''
@author: Chiang
'''
import numpy as np
from math import sqrt

class MLP(object):

    def __init__(self, eta=0.05, n_iter=10,n_hidden=2,n_out_put=1):

        self.n_iter = n_iter
        self.eta = eta
        self.n_hidden=n_hidden
        self.n_out_put=n_out_put

    def fit(self, x_modify, y):
        self.errors_ = []
        self.entropy_= []

        self.w_1=np.random.rand(x_modify.shape[1],self.n_hidden)-0.5
        self.w_2=np.random.rand(self.n_hidden,self.n_out_put)-0.5
        self.b_1=np.random.rand(self.n_hidden)[:,None]
        self.b_2=np.random.rand(self.n_out_put)[:,None]

        self.n_out_put = y.shape[1]
        print(self.n_out_put)
        for _ in range(self.n_iter):
            count=0
            for i in range(x_modify.shape[0]):

                hidden_input=np.dot(x_modify[i],self.w_1)[:,None]+self.b_1
                hidden_output=1/(1+np.exp(-hidden_input))



                z=np.dot(np.transpose(hidden_output),self.w_2)
                z=np.transpose(z)+self.b_2

                #soft-max
                output=(np.exp(z)/np.sum(np.exp(z),axis=0))
                #index of max
                ans=np.zeros(self.n_out_put)

                ans[np.where(output==output.max())[0][0]]=1
                #print(output.shape)        (10,)
                #print(hidden_output.shape) (21,)
                #print(hidden_input.shape)  (1, 20)

                #calculate deltas

                #print(temp_yz.shape)       (10,)
                #print(amp.shape)           (1, 20)
                #print(self.w_2.shape)      (21, 10)

                delta_2=np.dot((output-y[i][:,None]),np.transpose(hidden_output))


                delta_1=np.transpose((output-y[i][:,None])).dot(np.transpose(self.w_2))
                delta_1=delta_1*np.transpose(hidden_output)*np.transpose(1-hidden_output)
                delta_1=np.transpose(delta_1)
                delta_1=np.dot(delta_1,np.transpose(x_modify[i][:,None]))

                #delete last row
                delta_b1=np.transpose(output-y[i][:,None]).dot(np.transpose(self.w_2))*np.transpose(hidden_output)*np.transpose(1-hidden_output)
                delta_b2=np.transpose(output-y[i][:,None])

                self.w_1=self.w_1-self.eta*np.transpose(delta_1)/(1+_)**0.5
                self.w_2=self.w_2-self.eta*np.transpose(delta_2)/(1+_)**0.5
                self.b_1=self.b_1-self.eta*np.transpose(delta_b1)/(1+_)**0.5
                self.b_2=self.b_2-self.eta*np.transpose(delta_b2)/(1+_)**0.5


            for i in range(x_modify.shape[0]):
                hidden_input=np.dot(x_modify[i],self.w_1)[:,None]+self.b_1
                hidden_output=1/(1+np.exp(-hidden_input))



                z=np.dot(np.transpose(hidden_output),self.w_2)
                z=np.transpose(z)+self.b_2

                #soft-max
                output=(np.exp(z)/np.sum(np.exp(z),axis=0))
                #index of max
                ans=np.zeros(self.n_out_put)

                ans[np.where(output==output.max())[0][0]]=1


                flag=True;
                for j in range(self.n_out_put):
                    if ans[j]!=y[i][j]:
                        flag=False
                        break;
                    else:
                        continue;
                if flag==False:
                    count+=1
            self.errors_.append(count)
            print(str(_+1)+' round finished')
            print(str(count)+"\t"+str(-(np.sum(y[i][:,None]*np.log(output)))))
            self.entropy_.append(-(np.sum(y[i][:,None]*np.log(output))))
            #if count==0:
            	#break
        return self

