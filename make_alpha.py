#!/usr/bin/python
#encoding=utf-8

import math
import numpy as np
import LPP
import datetime
import random

k_hood=250
dim=39
frame=200
frame_num=198*frame

start=datetime.datetime.now()
print("k="+str(k_hood))
label_dict=dict()
label_file=open("test_label.txt",'r')
for line in label_file.readlines():
    line=line.strip().split()
    for i in range(len(line)):
        if line[i] not in label_dict:
            label_dict[line[i]]=list()
        label_dict[line[i]].append(i)
label_file.close()
#true_frame_num=len(label)
print("计算帧数完毕   frame_num="+str(frame_num))
print("开始抽样，每个状态抽取"+str(frame)+"帧")
label_loc=list()    #记录标签和抽取的位置
for state in label_dict:
    for i in range(frame):
        choose=random.randint(0,len(label_dict[state])-1)
        location=label_dict[state][choose]
        label_loc.append([location,state])

#print(len(label_loc))
label=list()
data=np.zeros((dim,frame_num),dtype=float)
fea_file=open("test.txt",'r')
row=0
for line in fea_file.readlines():
    line=line.strip().split()
    for i in range(frame_num):
        data[row,i]=line[label_loc[i][0]]  #读取相应位置的数据
        label.append(label_loc[i][1])    #读取标签
    row+=1
#print(data)
print("读入数据完毕")
fea_file.close()

print("开始计算欧式距离")
D,beta=LPP.calulateDistance(data,frame_num)
#D_txt="D_frame20000.txt"
#D=np.loadtxt(D_txt)
#beta=6051.48990544
#print(D)
print("开始进行监督学习")
D=LPP.supervisedLearningA(D,beta,frame_num,k_hood,label)
print("监督学习完毕，开始计算alpha")
alpha=LPP.calulateAlpha(D,data,frame_num)
alpha=alpha.T
print(alpha)
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
end=datetime.datetime.now()
print("time used:"+str(end-start))
