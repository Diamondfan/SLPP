#!/usr/bin/python
#encoding=utf-8

import math
import numpy as np
from scipy.sparse import linalg
from scipy.sparse import lil_matrix
from scipy.sparse import dia_matrix
from scipy.sparse import csc_matrix
from sklearn.neighbors import NearestNeighbors 

def find_k_max(distance,k_hood):
    k_max=0
    tmp_dis=list()
    for i in range(len(distance)):
        tmp_dis.append(distance[i])
    tmp_dis.sort(reverse=False)
    k_max=tmp_dis[k_hood]
    #print(tmp_dis)
    return  k_max

#计算k近邻关系，强行计算
'''
def traditionalLPP(data,frame_num,k_hood,beta):    
    #计算每一帧与frame_num帧的距离
    #D=np.zeros((frame_num,frame_num),dtype=float)
    D=lil_matrix((frame_num,frame_num))
    for i in range(frame_num):
        distance=list()
        #wi=data[:,i]/np.max(data[:,i])
        avr=0
        for j in range(frame_num):
            #计算欧式距离
            #wj=data[:,j]/np.max(data[:,j])
            euclid=np.linalg.norm(data[:,j]-data[:,i])
            distance.append(euclid)
            avr+=euclid
        k_max=find_k_max(distance,k_hood)
        avr=avr/frame_num
        print(i,avr)
        for j in range(frame_num):
            if distance[j] < k_max:
                D[i,j]=math.exp(-distance[j]**2/(beta*avr*avr))
                #D[i,j]=1
    #转化为对称矩阵
    #print(D.toarray())
    DT=D.transpose()
    #print(DT.toarray())
    D=lil_matrix.maximum(D,DT)
    #print(D.toarray())
    return D
'''
#使用sklearn中的B树存储结构，加快速度
def traditionalLPP(data,frame_num,k_hood,beta): 
    D=lil_matrix((frame_num,frame_num))
    nbrs=NearestNeighbors(n_neighbors=k_hood,algorithm='ball_tree').fit(data)
    distance,indices=nbrs.kneighbors(data)
    for i in range(frame_num):
        #avr=distance[i].mean()
        for j in range(k_hood):
            index=indices[i][j]
            dis=distance[i][j]
            D[i,index]=math.exp(-dis**2/(beta))
        #print(i,dis)
    #转化为对称矩阵
    DT=D.transpose()
    D=lil_matrix.maximum(D,DT)
    #print(D.toarray())
    return D

def calulateAlpha(D,data,frame_num,ndim,ldim):
    #L=np.zeros((frame_num,frame_num),dtype=float)
    L=dia_matrix((D.sum(0),0),shape=(frame_num,frame_num))
    #print(type(D))
    #print(type(L))
    D=L-D 
    #L=np.mat(L.toarray())
    #D=np.mat(D.toarray())
    #print(L.toarray())
    #print(D.toarray())
    data=csc_matrix(data)
    LMatrix=data*L.tocsc()*data.transpose()
    #对称半正定矩阵
    LMatrix=csc_matrix.maximum(LMatrix,LMatrix.transpose())
    RMatrix=data*D.tocsc()*data.transpose()
    RMatrix=csc_matrix.maximum(RMatrix,RMatrix.transpose())
    finalMatrix=linalg.inv(LMatrix)*RMatrix
    finalMatrix=finalMatrix.toarray()
    value,alpha=np.linalg.eig(finalMatrix)
    VA=dict()
    print("特征值个数为:"+str(len(value)))
    for i in range(len(value)):
        VA[value[i]]=alpha[:,i]
    value_sort=sorted(VA)
    print(value_sort[i])
    arr=np.zeros(ndim*ldim).reshape(ndim,ldim)
    #print("arr:",arr)
    for i in range(ldim):
        arr[:,i]=VA[value_sort[i]]
    return arr
    #alpha=np.arange(25).reshape(5,5)

def cal_sub_alpha(LMatrix,RMatrix,ndim,ldim):
    LMatrix=csc_matrix.maximum(LMatrix,LMatrix.transpose())
    RMatrix=csc_matrix.maximum(RMatrix,RMatrix.transpose())
    finalMatrix=linalg.inv(LMatrix)*RMatrix
    finalMatrix=finalMatrix.toarray()
    value,alpha=np.linalg.eig(finalMatrix)
    VA=dict()
    print("特征值个数为:"+str(len(value)))
    for i in range(len(value)):
        VA[value[i]]=alpha[:,i]
    value_sort=sorted(VA)
    for i in range(len(value_sort)):
        if value_sort[i]>=0:
            loc=i
            break
    print(value_sort[loc:])
    arr=np.zeros(ndim*ldim).reshape(ndim,ldim)
    #print("arr:",arr)
    for i in range(ldim):
        arr[:,i]=VA[value_sort[loc+i]]
    return arr

def supervisedLearningA(D,beta,frame_num,k_hood,label):    
    #sum_cal_t=0
 
    for i in range(frame_num):
        dis=list()
        for j in range(frame_num):
            dis.append(D[i,j])
        dis.sort(reverse=False)
        for j in range(frame_num):
            if D[i,j] >= dis[k_hood]:
                D[i,j]=0
            elif label[i]==label[j]:
                D[i,j]=math.sqrt(math.exp(-D[i,j]/beta))
            else:
                D[i,j]=math.sqrt(1-math.exp(-D[i,j]/beta))
    for i in range(frame_num):
        for j in range(frame_num):
            if D[i,j]==0:
                D[i,j]=D[j,i]
    return D


def supervisedLearningB(D,beta,frame_num,k_hood,label):    
    max_same=0
    max_dif=0
    for i in range(frame_num):
        dis=list()
        for j in range(frame_num):
            dis.append(D[i,j])
        dis.sort(reverse=False)
        for j in range(frame_num):
            if D[i,j] >= dis[k_hood]:
                D[i,j]=0
            elif label[i]==label[j]:
                D[i,j]=math.sqrt(math.exp(-D[i,j]/beta))
                if D[i,j]>max_same:
                    max_same=D[i,j]
            else:
                D[i,j]=math.sqrt(1-math.exp(-D[i,j]/beta))
                if D[i,j]>max_dif:
                    max_dif=D[i,j]
    z=max_same/max_dif
    for i in range(frame_num):
        for j in range(frame_num):
            if D[i,j]==0:
                D[i,j]=D[j,i]
    for i in range(frame_num):
        for j in range(frame_num):
            if label[i]!=label[j]:
                D[i,j]=D[i,j]*z
    return D


    
