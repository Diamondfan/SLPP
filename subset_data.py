#!/usr/bin/python
#encoding=utf-8

import sys
import os

def Usage():
    print("Usage:python "+sys.argv[0]+"[data file] [label file]")
    print("The script is to subset the data file according to the label")
    print("")

if len(sys.argv)!=3:
    Usage()
    sys.exit(1)

def processint(le):
    yu=le%100
    return le-yu

print("Reading the label_file:")
lable_loc=dict()
label_file=open(sys.argv[2],'r')
n=1
for line in label_file.readlines():
    line=line.strip().split()
    for loc in range(len(line)):
        if line[loc] not in lable_loc:
            lable_loc[line[loc]]=list()
        lable_loc[line[loc]].append(loc)

        if n*processint(len(line))==(loc*10):
            print('*'*(n)+' '+str(n*10)+'%')
            n+=1
label_file.close()

print("Subset the data file")
#os.mkdir('sub_data')
data_file=open(sys.argv[1],'r')
line=data_file.readlines()
os.chdir("./sub_data")
for label in lable_loc:
    data=open(label+'.txt','w')
    for loc in lable_loc[label]:
        data.writelines(line[loc])
    data.close()
    print(label+" down")
data_file.close()
print("Subset down")

