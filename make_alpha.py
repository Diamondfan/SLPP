#!/usr/bin/python
#encoding=utf-8

import math
import numpy as np
import sparse_LPP
import datetime
import random
import sys,getopt

def usage():
    print("usage:python "+sys.argv[0]+"[option]")
    print("-k    :   set k_hood value")


k_hood=100
ndim=91
beta=2
ldim=39
frame=100
frame_num=200*frame

opts,argvs=getopt.getopt(sys.argv[1:],"hk:")
for op,value in opts:
    if op == "-k":
        k_hood=int(value)
    elif op=="-h":
        usage()
        sys.exit()

start=datetime.datetime.now()
print("k="+str(k_hood))

label_dict=dict()
label_file=open("label.txt",'r')
for line in label_file.readlines():
    line=line.strip().split()
    true_frame_num=len(line)
    for i in range(true_frame_num):
        if line[i] not in label_dict:
            label_dict[line[i]]=list()
        label_dict[line[i]].append(i)
label_file.close()
print("计算帧数完毕   frame_num="+str(true_frame_num))
print("开始抽样,抽取"+str(frame_num)+"帧")
print(len(label_dict))
label_loc=list()    #记录标签和抽取的位置
for state in label_dict:
    choose=random.sample(range(0,len(label_dict[state])),frame)
    for i in range(frame):        
        location=label_dict[state][choose[i]]
        label_loc.append([location,state])

#print(len(label_loc))
#label=list()

#choose=random.sample(range(0,true_frame_num),frame_num)
#print(choose)
data=np.zeros((ndim,frame_num),dtype=float)
fea_file=open("data.txt",'r')
row=0
for line in fea_file.readlines():
    line=line.strip().split()
    #print("line:",len(line))
    for i in range(frame_num):
        data[row,i]=line[label_loc[i][0]]  #读取相应位置的数
        #label.append(label_loc[i][1])    #读取标签
    row+=1
#print(data)
print("读入数据完毕")
fea_file.close()

print("开始计算欧式距离")
#D,beta=LPP.calulateDistance(data,frame_num)
print("开始进行监督学习")
D=sparse_LPP.traditionalLPP(data,frame_num,k_hood,beta)
#D=LPP.traditionalLPP(D,beta,frame_num,k_hood)
print("监督学习完毕，开始计算alpha")
alpha=sparse_LPP.calulateAlpha(D,data,frame_num,ndim,ldim)
alpha=alpha.T
print(alpha)
out=open('alpha_k'+str(k_hood)+'_beta'+str(beta)+'_frame7.txt','w')
out.writelines('[ '+'\n')
for i in range(ldim):
    for j in range(ndim):
        out.writelines(str(alpha[i,j])+' ')
    if i!=ldim-1:
        out.writelines('\n')
    else:
        out.writelines(']')
out.close()
end=datetime.datetime.now()
print("time used:"+str(end-start))
