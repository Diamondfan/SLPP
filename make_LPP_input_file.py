#!/usr/bin/python
#encoding=utf-8

import sys
import numpy as np

if len(sys.argv)!=3:
    print("usage:python "+sys.argv[0]+" [frame file] [label file]")
    sys.exit(1)

frame_file=sys.argv[1]
label_file=open(sys.argv[2])

frame=np.loadtxt(frame_file)
frame_tran=frame.T
np.savetxt("test.txt",frame_tran,fmt='%0.7f')

#label=list()
#num=0
out=open("test_label.txt",'w')
for line in label_file.readlines():
    line=line.strip().split()
    for i in range(len(line)):
        if i > 0 :
            out.writelines(line[i]+' ')
            #num+=1
out.close()

