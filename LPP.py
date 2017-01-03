#!/usr/bin/python
#encoding=utf-8

import math
import numpy as np


def calulateDistance(data,frame_num):
    D=np.zeros((frame_num,frame_num),dtype=float)
    beta=0
    for i in range(frame_num):
        for j in range(i+1,frame_num):
            D[i,j]=np.linalg.norm(data[:,j]-data[:,i])
            D[j,i]=D[i,j]
            beta=beta+2*D[i,j]
    beta=(beta/(frame_num**2))**2
    print("beta="+str(beta))
    #print("max="+str(max))
    return D,beta
    #print(D)


def supervisedLearningA(D,beta,frame_num,k_hood,label):    
    #sum_cal_t=0
    for i in range(frame_num):
        for j in range(i,frame_num):
            if label[i]==label[j]:
                D[i,j]=math.sqrt(1-math.exp(-D[i,j]**2/beta))
            else:
                D[i,j]=math.sqrt(math.exp(-D[i,j]**2/beta))
            D[j,i]=D[i,j]
            #sum_cal_t = sum_cal_t+D[i,j]+D[j,i]
    
    for i in range(frame_num):
        dis=list()
        for j in range(frame_num):
            dis.append(D[i,j])
        dis.sort(reverse=False)
        for j in range(frame_num):
            if D[i,j] >= dis[k_hood+1]:
                D[i,j]=0
    
    for i in range(frame_num):
        for j in range(frame_num):
            if D[i,j]==0:
                D[i,j]=D[j,i]
    
    return D
    #print(D)

def newLearningA(D,beta,frame_num,k_hood,label):    
    #sum_cal_t=0
 
    for i in range(frame_num):
        dis=list()
        for j in range(frame_num):
            dis.append(D[i,j])
        dis.sort(reverse=False)
        for j in range(frame_num):
            if D[i,j] >= dis[k_hood+1]:
                D[i,j]=0
            elif label[i]==label[j]:
                D[i,j]=math.sqrt(math.exp(-D[i,j]**2/beta))
            else:
                D[i,j]=math.sqrt(1-math.exp(-D[i,j]**2/beta))
    for i in range(frame_num):
        for j in range(frame_num):
            if D[i,j]==0:
                D[i,j]=D[j,i]
    
    return D

def supervisedLearningB(D,beta,frame_num,k_hood,label):    
    max_same=0
    max_dif=0
    for i in range(frame_num):
        for j in range(i,frame_num):
            if label[i]==label[j]:
                D[i,j]=math.sqrt(math.exp(-D[i,j]**2/beta))
                if D[i,j]>max_same:
                    max_same=D[i,j]
            else:
                D[i,j]=math.sqrt(1-math.exp(-D[i,j]**2/beta))
                if D[i,j]>max_dif:
                    max_dif=D[i,j]
            D[j,i]=D[i,j]
    z=max_same/max_dif
    for i in range(frame_num):
        dis=list()
        for j in range(frame_num):
            dis.append(D[i,j])
        dis.sort(reverse=False)
        for j in range(frame_num):
            if D[i,j] >= dis[k_hood+1]:
                D[i,j]=0
    for i in range(frame_num):
        for j in range(frame_num):
            if D[i,j]==0:
                D[i,j]=D[j,i]
    for i in range(frame_num):
        for j in range(frame_num):
            if label[i]!=label[j]:
                D[i,j]=D[i,j]*z
    return D

def calulateAlpha(D,data,frame_num):
    L=np.zeros((frame_num,frame_num),dtype=float)
    for i in range(frame_num):
        for j in range(frame_num):
            L[i,i]=L[i,i]+D[i,j]
    #print(L)
    D=L-D
    #print(D)
    L=np.mat(L)
    D=np.mat(D)
    data=np.mat(data)
    finalMatrix=((data*L*data.T).I)*(data*D*data.T)
    #print(finalMatrix)
    value,alpha=np.linalg.eig(finalMatrix)
    return alpha
    #alpha=np.arange(25).reshape(5,5)
    


