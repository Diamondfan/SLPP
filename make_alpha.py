#!/usr/bin/python
#encoding=utf-8

import math
import numpy as np
import LPP
import datetime

k_hood=350
dim=39
frame_num=40000

start=datetime.datetime.now()
print("k="+str(k_hood))
label=list()
label_file=open("test_label.txt",'r')
for line in label_file.readlines():
    label=line.strip().split()
    label_file.close()
#true_frame_num=len(label)
print("计算帧数完毕   frame_num="+str(frame_num))

data=np.zeros((dim,frame_num),dtype=float)
fea_file=open("test.txt",'r')
row=0
for line in fea_file.readlines():
    line=line.strip().split()
    for i in range(frame_num):
        data[row,i]=line[i]
    row+=1
#print(data)
print("读入数据完毕")
fea_file.close()


print("开始计算欧式距离")
D,beta=LPP.calulateDistance(data,frame_num)
#D_txt="D_frame"+str(frame_num)+'.txt'
#np.savetxt(D_txt,D)
#print(D)
print("开始进行监督学习")
D=LPP.supervisedLearningB(D,beta,frame_num,k_hood,label)
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
