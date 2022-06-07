import os
import matplotlib.pyplot as plt # For general plotting
from matplotlib import cm
import matplotlib
import numpy as np
from modules import models, prob_utils
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal # MVN not univariate
from scipy import random, linalg
from sklearn.metrics import confusion_matrix


np.random.seed(7)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

t1 = 20
t2 = 200
t3 = 2000
v = 10000
m= np.array([[3, 0],
                [0, 3],
                [2,2]])
m=np.transpose(m)
#print(m)
c = np.array([[[2, 0],
                [0, 1]],

                [[1, 0],
                [0,2]],

                [[1,0],
                [0,1]],
                
                ])

classPrior=np.array([0.325,0.325,0.35])
C=3

gauss_params = prob_utils.GaussianMixturePDFParameters(classPrior,C,m,np.transpose(c))

gauss_params.print_pdf_params()

X_train=[]
y_train=[]

n=m.shape[0]
#print(n)
Xt1,yt1 = prob_utils.generate_mixture_samples(t1, n, gauss_params, False)
yt1=yt1>1
Xt1 = np.row_stack((np.ones(20), Xt1))
X_train.append(Xt1)
y_train.append(yt1)

Xt2,yt2 = prob_utils.generate_mixture_samples(t2, n, gauss_params, False)
yt2=yt2>1
Xt2 = np.row_stack((np.ones(200), Xt2))
X_train.append(Xt2)
y_train.append(yt2)



Xt3,yt3 = prob_utils.generate_mixture_samples(t3, n, gauss_params, False)
yt3=yt3>1
Xt3 = np.row_stack((np.ones(2000), Xt3))
X_train.append(Xt3)
y_train.append(yt2)
print(y_train[0])

X_valid,y_valid = prob_utils.generate_mixture_samples(v, n, gauss_params, False)
y_valid=y_valid>1

#print("labels: ")
#print(y)
fig0 = plt.figure(figsize=(4, 4), dpi=200)
ax1 = fig0.gca()
color=['blue', 'red']
for i in range(2):
    ax1.scatter(X_valid[0, (y_valid==i)],
              X_valid[1, (y_valid==i)],
              marker='.', c=color[i],alpha=0.6,label="class"+str(i))
ax1.set_xlim((-5, 5))
ax1.set_ylim((-5, 5))
ax1.set_title('Q2 Samples')
plt.tight_layout()
plt.savefig('Q1_Samples.jpg')
plt.show()

m=np.transpose(m)
discriminant_score=np.zeros(v)
print(X_valid.shape)
for i in range(v):
    discriminant_score[i]=np.log(multivariate_normal.pdf(X_valid[:,i],mean=m[2],cov=c[2]))

#print("discrimnent score: ")
#print(discriminant_score)
threshold = np.sort(discriminant_score)

sumOfLabel0=np.sum(y_valid==0)
sumOfLabel1=np.sum(y_valid==1)
#print(str(sumOfLabel0)+"and"+str(sumOfLabel1))

size=threshold.size
p01=np.zeros(size)
p10=np.zeros(size)
p11=np.zeros(size)
perr=np.zeros(size)
label=y_valid==1

#print(label)
decisions=np.zeros((threshold.size,v))

for i in range(size):
    decision=discriminant_score>threshold[i]
    decisions[i]=decision
    p01[i]=np.sum(~decision & label)/sumOfLabel1
    p10[i]=np.sum(decision & ~label)/sumOfLabel0
    p11[i]=np.sum(decision & label)/sumOfLabel1
    perr[i]=p01[i]*classPrior[1]+p10[i]*classPrior[0]

threshold_best = threshold[np.argmin(perr)]
decision_best=decisions[np.argmin(perr)]
#gamma=float(np.sum(decision==0))/float(np.sum(decision==1))
correct=0
for i in range(v):
    if decision_best[i]==y_valid[i]:
        correct+=1
print("total correct case: "+str(correct))
print("Minimum Perr possible: "+str(min(perr)))
print("Threshold that achived minimun probability of error is :"+str(threshold_best))

fig1 = plt.figure(figsize=[4, 3], dpi=200)
ax1 = fig1.add_subplot(111)
ax1.plot(p10, p11, linewidth=1)
#print(p10[np.argmin(perr)])
#print(p11[np.argmin(perr)])
ax1.scatter(p10[np.argmin(perr)], p11[np.argmin(perr)], c='r',
            marker='x', label=r'minimum Perr')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_xlabel(r'$PFP$')
ax1.set_ylabel(r'$PTP$')
ax1.set_title('ROC curve with Ture class Q1a')
ax1.legend()
plt.tight_layout()
plt.savefig('Q1a_roc.jpg')
plt.show()



fig2 = plt.figure(figsize=(4, 4), dpi=200)
ax1 = fig2.gca()
for i in range(2):
    ax1.scatter(X_valid[0, ((decision_best==i) & (y_valid==i))], 
              X_valid[1, ((decision_best==i) & (y_valid==i))], 
              marker='.', c='green', alpha=.6,label="Class"+str(i)+"correct")
    ax1.scatter(X_valid[0, ((~(decision_best==i)) & (y_valid==i))], 
              X_valid[1, ((~(decision_best==i)) & (y_valid==i))], 
              marker='.', c='red', alpha=.6,label="Class"+str(i)+"wrong")
ax1.set_xlim((-5, 9))
ax1.set_ylim((-5, 9))
ax1.set_title('Q2 result')
plt.tight_layout()
#plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.05),
          #ncol=1, fancybox=True, shadow=True,)
plt.savefig('Q2 result.jpg')
plt.show()

