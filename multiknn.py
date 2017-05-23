from sklearn.metrics import hamming_loss,zero_one_loss,coverage_error,label_ranking_loss,average_precision_score

import csv
import numpy as np
import math
import operator

loc=14  #length of class
print 'enter k'
k=int(raw_input())
s=1		#smoothning parameterr here 1
H1=[]
H0=[]
for i in range(loc):
	H1.append(0)
	H0.append(0)

def dataload(filename):													#load data and convert into a list
	with open(filename,'rb') as csvfile:
		dataset=csv.reader(csvfile)
		data=list(dataset)
		data.remove([])
		datafinal=[]
		c=0
		for x in data:
			if(x==[]):
				c=0
			elif(x[0]=='@data'):
				c=1
			if(c==1):
				datafinal.append(x)
		del(datafinal[0])
		for x in range(len(datafinal)):
			for y in range(len(datafinal[0])):
				datafinal[x][y]=float(datafinal[x][y])

	
		return datafinal


data=dataload('yeast-train.csv')
data=data[:100]
m=len(data)  #total training data
lod=len(data[0])   #length of features +labels
dictfeatures={}
dictlabels={}
for x in range(len(data)):
	dictfeatures[x]=data[x][:(lod-loc)]
	dictlabels[x]=data[x][lod-loc:]




def func_y(x,l):
	#print dictlabels[x][l]
	if(dictlabels[x][l]==1):
		return 1
	else:
		return 0

def distanceeuclid(data1,data2):									#calculate the euclidean distance 
	distance=0
	for i in range(len(data1)):
		distance+=pow((data1[i]-data2[i]),2)
	return math.sqrt(distance)

def getneighbours(xi):
	distances=[]
	for j in range(m):
		dist=distanceeuclid(dictfeatures[xi],dictfeatures[j])
		distances.append((j,dist))
	distances.sort(key=operator.itemgetter(1))
	neigbour=[]
	for x in range(1,k+1):
		neigbour.append(distances[x][0])

	return neigbour	






for l in range(loc):
	s1=0
	for i in range(m):
		s1=s1+func_y(i,l)
	H1[l]=float(s+s1)/float(s*2+m)
	H0[l]=1-H1[l]



c=[]
c1=[]
EX1=[]
EX0=[]
for l in range(loc):
	e1=[]
	e2=[]
	for j in range(k+1):
		c.append(0)
		c1.append(0)
	for i in range(m):
		Nxi=getneighbours(i)
		s2=0
		for a in Nxi:
			s2=s2+func_y(a,l)
		delta=s2
		if(func_y(i,l)==1):
			c[delta]=c[delta]+1
		else:
			c1[delta]=c1[delta]+1

	for j in range(k+1):
		csum=0
		c1sum=0
		for p in range(k+1):
			csum=csum+c[delta]
			c1sum=c1sum+c1[delta]
		e1.append(float(s+c[j])/float(s*(k+1)+csum))
		e2.append(float(s+c1[j])/float(s*(k+1)+c1sum))
	EX1.append(e1)
	EX0.append(e2)

#print EX1
#print EX0



testdata=dataload('yeast-test.csv')
testdata=testdata[:20]
mt=len(testdata)  #total testing data
lod=len(testdata[0])   #length of features +labels
dicttestfeatures={}
dicttestlabels={}
for x in range(len(testdata)):
	dicttestfeatures[x]=testdata[x][:(lod-loc)]
	dicttestlabels[x]=testdata[x][lod-loc:]
Ytotaltrue=[]
for x in range(mt):
	Ytotaltrue.append(dicttestlabels[x])

def gettestneigbours(xt):
	distances=[]
	for j in range(m):
		dist=distanceeuclid(dictfeatures[j],dicttestfeatures[xt])
		distances.append((j,dist))
	distances.sort(key=operator.itemgetter(1))
	neigbour=[]
	for x in range(k):
		neigbour.append(distances[x][0])

	return neigbour	


Ytotalpred=[]
sumhl=0.0
sumoneerror=0.0
sumcoverageer=0.0
sumavgpre=0.0
for j in range(mt):
	yt=[]
	for l in range(loc):
		Nt=gettestneigbours(j)
		s2=0
		for a in Nt:
			s2=s2+func_y(a,l)
		CT=s2
		m1=H1[l]*EX1[l][CT]
		m0=H0[l]*EX0[l][CT]
		if(m1>m0):
			yt.append(1)
		else:
			yt.append(0)
	Ytotalpred.append(yt)
	y_pred=yt
	y_true=dicttestlabels[j]
	hammingloss=hamming_loss(y_true,y_pred)
	sumhl=sumhl+hammingloss

	oneerror=zero_one_loss(y_true, y_pred)
	sumoneerror=sumoneerror+oneerror

	avgpre=average_precision_score(y_true,y_pred)
	sumavgpre=sumavgpre+avgpre









meanhl=sumhl/float(mt)
meanonerror=sumoneerror/float(mt)
coverageerr=coverage_error(Ytotaltrue,Ytotalpred)
rankloss=label_ranking_loss(Ytotaltrue,Ytotalpred)
meanavgpre=sumavgpre/float(mt)


print 'mean hamming loss '+repr(meanhl)
print 'one error '+repr(meanonerror)
print 'coverage error '+repr(coverageerr)
print 'rank loss '+repr(rankloss)
print 'average precision '+repr(meanavgpre)










