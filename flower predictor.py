#!/usr/bin/env python
# coding: utf-8

# In[176]:


import numpy as np
import pandas as pd


# In[177]:


p=pd.read_csv("IRIS.csv")


# In[178]:


p


# In[179]:


p1=p.values[:,:]
np.random.shuffle(p1)


# In[180]:


p1.shape


# In[181]:


x=p1[:130,0:4]


# In[182]:


y=p1[:130,4]


# In[183]:


y=y.reshape(130,1)


# In[184]:


y.shape


# In[185]:


y


# In[186]:


k=np.zeros((3,130))
y=y.T
for i in range(130):
    if y[0][i]=='Iris-setosa':
        k[0][i]=1
    if y[0][i]=="Iris-versicolor":
        k[1][i]=1
    if y[0][i]=="Iris-virginica":
        k[2][i]=1
        
        


# In[187]:


k


# In[188]:


y1=k


# In[189]:


mean=np.mean(x,axis=0)


# In[190]:


mean


# In[191]:


sigma=np.std(x,axis=0,ddof=1,dtype=np.float32)


# In[192]:


x=(x-mean)/sigma
x=x.T


# In[193]:


layer=[4,10,3]


# In[194]:


def parameter(ly):
    L=len(ly)
    parameters={}
    
    for l in range(1,L):
        parameters["w"+str(l)]=np.random.randn(ly[l],ly[l-1])*0.1
        parameters["b"+str(l)]=np.zeros((ly[l],1))
        
        assert(parameters['w' + str(l)].shape == (ly[l], ly[l - 1]))
        assert(parameters['b' + str(l)].shape == (ly[l], 1))
    return parameters


# In[195]:


def linear_forward(A,w,b):
    Z=np.dot(w,A)+b
    cache=(A,w,b)
    return Z,cache


# In[196]:


def sigmoid(z):
    b=np.multiply(-1,z).astype(float)
    a=1/(1+np.exp(b))
    return a,z


# In[197]:


def forward_activation(A_prev,w,b):
    Z,linear_cache=linear_forward(A_prev,w,b)
    A,activation_cache=sigmoid(Z)
    cache=(linear_cache,activation_cache)
    return A,cache


# In[198]:


def forward_lmodel(x,parameters):
    A=x
    caches=[]
    L=len(parameters)//2
    for l in range(1,L+1):
        A_prev=A
        A,cache=forward_activation(A_prev,parameters["w"+str(l)],parameters["b"+str(l)])
        caches.append(cache)
        
        
    
    return A,caches


# 

# In[199]:


def cost(AL,y):
    m=y.shape[1]
    cost=(-1/m)*np.sum((np.multiply(y,np.log(AL))+np.multiply(1-y,np.log(1-AL))))
    cost = np.squeeze(cost) 
    return cost


# In[200]:


def linear_backward(dZ,cache):
    A_prev,w,b=cache
    m=A_prev.shape[1]
    
    
    dw=(1/m)*np.dot(dZ,A_prev.T)
    db=(1/m)*np.sum(dZ,axis=1,keepdims=True )
    dA_prev=np.dot(w.T,dZ)
    return dA_prev, dw, db


# In[201]:


def linear_activation_backward(dA, cache):
    linear_cache,activation_cache=cache
    Z=activation_cache
    f,z=sigmoid(Z)
    dZ=np.multiply(dA,np.multiply(f,1-f))
    dA_prev, dW, db =linear_backward(dZ,linear_cache)
    
    return dA_prev, dW, db


# In[202]:


def backward_lmodel(AL,y,caches):
    grads={}
    L = len(caches)
    m = AL.shape[1]
    y= y.reshape(AL.shape)
    
    dAL = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))
    dA_prev_temp=dAL
    for i in reversed(range(L)):
        current_cache = caches[i]
        dA_prev_temp, dW_temp, db_temp =linear_activation_backward(dA_prev_temp,current_cache) 
        grads["dA" + str(i)] = dA_prev_temp
        grads["dw" + str(i+1)] = dW_temp
        grads["db" + str(i+1)] =db_temp
        
        
    return grads    


# In[203]:


def update(grads,parameters,learning_rate):
    L=len(parameters)//2
    for i in range(1,3):
        parameters["w"+str(i)]=parameters["w"+str(i)]-learning_rate*grads["dw" + str(i)]
        parameters["b"+str(i)]=parameters["b"+str(i)]-learning_rate*grads["db" + str(i)]
        
    return parameters 


# In[204]:


parameters=parameter(layer)


# In[205]:


def L_layer_model(x, y, layer, learning_rate = 0.7, num_iterations = 3000, print_cost=False):
    parametersy=parameter(layer)
    costs = []
    
    for i in range(0,num_iterations):
    
        AL,caches=forward_lmodel(x,parametersy)
        cost1 = cost(AL, y)
        grads=backward_lmodel(AL,y,caches)
        parametersy = update(grads,parametersy,learning_rate)
        
        costs.append(cost1)
            
    return parametersy,costs    


# In[206]:


train_x=x[0:,:]
train_y=y1[0:,:]


# In[207]:


parameters, costs = L_layer_model(train_x, train_y, layer, num_iterations=10000, print_cost = True)


# In[208]:


costs[9999]


# In[213]:


x_test=p1[130:,0:4]
x_test=(x_test-mean)/sigma
x_test=x_test.T


# In[214]:


y_test=p1[130:,4]


# In[215]:


P,k=forward_lmodel(x_test,parameters)


# In[216]:


P


# In[219]:


k=np.zeros((20,1))
for i in range(20):
    c=0.5
    for j in range(3):
        if P[j][i]>0.5:
            k[i][0]=j+1


# In[220]:


k


# In[227]:


for i in range(20):
    if y_test[i][0]=='Iris-setosa':
        y_test[i][0]=1
    if y_test[i][0]=="Iris-versicolor":
        y_test[i][0]=2
    if y_test[i][0]=="Iris-virginica":
        y_test[i][0]=3


# In[228]:


y_test


# In[ ]:




