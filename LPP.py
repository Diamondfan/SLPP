#!/usr/bin/python
#encoding=utf-8

import math
import numpy as np

k_hood=200
dim=39
frame_num=40000


label=list()
label_file=open("test_label.txt",'r')
for line in label_file.readlines():
    label=line.strip().split()
label_file.close()
true_frame_num=len(label)
print("计算帧数完毕   frame_num="+str(frame_num))

data=np.zeros((dim,true_frame_num),dtype=float)
fea_file=open("test.txt",'r')
row=0
for line in fea_file.readlines():
    line=line.strip().split()
    for i in range(len(line)):
        data[row,i]=line[i]
    row+=1
#print(data)
print("读入数据完毕")
fea_file.close()

D=np.zeros((frame_num,frame_num),dtype=float)
beta=0
max=0
def calulateDistance():
    global max
    global D
    global beta
    for i in range(frame_num):
        for j in range(i,frame_num):
            D[i,j]=0
            for k in range(dim):
                D[i,j]=(D[i,j]+(data[k,j]-data[k,i])*(data[k,j]-data[k,i]))
            D[i,j]=math.sqrt(D[i,j])
            D[j,i]=D[i,j]
            if D[i,j]>max:
                max=D[i,j]
            beta=beta+D[i,j]+D[j,i]
    beta=(beta/frame_num/frame_num)*(beta/frame_num/frame_num)
    print("beta="+str(beta))
    print("max="+str(max))
    #print(D)

sum_cal_t=0
MAX=0
def supervisedLearning():    
    global sum_cal_t
    global D
    global MAX
    global beta
    global max
    MAX=max*frame_num
    for i in range(frame_num):
        for j in range(i,frame_num):
            if label[i]==label[j]:
                D[i,j]=math.sqrt(math.exp(-D[i,j]*D[i,j]/beta))
            else:
                D[i,j]=math.sqrt(1-math.exp(-D[i,j]*D[i,j])/beta)
            D[j,i]=D[i,j]
            sum_cal_t = sum_cal_t+D[i,j]+D[j,i]
    #print("sum_cal_t="+str(sum_cal_t))
    #print(D)
    for i in range(frame_num):
        dis=list()
        for j in range(frame_num):
            dis.append(D[i,j])
        dis.sort(reverse=False)
        for j in range(frame_num):
            if D[i,j] >= dis[k_hood+1]:
                D[i,j]=MAX
    for i in range(frame_num):
        for j in range(frame_num):
            if D[i,j]==MAX:
                D[i,j]=D[j,i]
    #print(D)

def calulateAlpha():
    global D
    global data
    global sum_cal_t
    global MAX
    t=sum_cal_t/frame_num/frame_num
    print("t="+str(t))
    for i in range(frame_num):
        for j in range(frame_num):
            if D[i,j]!=MAX:
                D[i,j]=math.exp(-D[i,j]*D[i,j]/t)
            elif D[i,j]==MAX:
                D[i,j]=0
    #print(D)
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
    print(finalMatrix)
    value,alpha=np.linalg.eig(finalMatrix)
    print(alpha)
    #alpha=np.arange(25).reshape(5,5)
    out=open('alpha_k'+str(k_hood)+'.txt','w')
    out.writelines('[ '+'\n')
    for i in range(dim):
        for j in range(dim):
            out.writelines(str(alpha[i,j])+' ')
        if i!=dim-1:
            out.writelines('\n')
        else:
            out.writelines(']')
    out.close()

print("开始计算欧式距离")
calulateDistance()
print("开始进行监督学习")
supervisedLearning()
print("监督学习完毕，开始计算W")
calulateAlpha()
