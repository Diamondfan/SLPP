#!/usr/bin/python
#encoding=utf-8
#author: Richardfan
#the script is to make Matrix alpha for LPP using the idea of CTC

import math
import numpy as np
import LPP
import datetime
import random
import sys,getopt

def usage():
    print("usage:python "+sys.argv[0]+"[option]")
    print("-k    :   set k_hood value")


k_hood=100
dim=39
frame=100
frame_num=200*frame

opts,argvs=getopt.getopt(sys.argv[1:],"hk:")
for op,value in opts:
    if op == "-k":
        k_hood=int(value)
    elif op=="-h":
        usage()
        sys.exit()
if len(sys.argv)!=3:
    usage()
    sys.exit(1)

start=datetime.datetime.now()
print("k="+str(k_hood))
label_position=list()


#读取每帧的label，连续相同的只取一针
#many-to-one,此处选取中间一帧

label_file=open("label.txt",'r')
for line in label_file.readlines():
    line=line.strip().split()
    tmp_label=line[0]
    tmp_count=0
    for i in range(len(line)):
        if line[i]==tmp_label:
            tmp_count+=1
        else:
            position=i-(tmp_count/2+1)
            label_position.append([tmp_label,position])
            tmp_label=line[i]
            tmp_count=1
label_file.close()
num=len(label_position)
print("计算帧数完毕   frame_num="+str(num))
print("开始抽样,抽取"+str(frame_num)+"帧")

label_count=dict()
for i in range(len(label_position)):
    if label_position[i][0] not in label_count:
        label_count[label_position[i][0]]=list()
    label_count[label_position[i][0]].append(label_position[i][1])

label_loc=list()
for state in label_count:
    for i in range(frame):
        choose=random.randint(0,len(label_count[state])-1)
        location=label_count[state][choose]
        label_loc.append([location,state])

#这样的结果最后的帧数frame_num=394825  还是大大超过计算范围
#继续从这些帧里面进行采样


label=list()
data=np.zeros((dim,frame_num),dtype=float)
fea_file=open("data.txt",'r')
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
#print(D)
print("开始进行监督学习")
D=LPP.traditionalLPP(D,beta,frame_num,k_hood)
#D=LPP.supervisedLearningB(D,beta,frame_num,k_hood,label)
print("监督学习完毕，开始计算alpha")
alpha=LPP.calulateAlpha(D,data,frame_num)
alpha=alpha.T
print(alpha)
out=open('new_alpha_k'+str(k_hood)+'.txt','w')
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
