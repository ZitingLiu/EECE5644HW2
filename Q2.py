import matplotlib.pyplot as plt # For general plotting
from matplotlib import cm

from math import ceil, floor 
import numpy.matlib
import numpy as np
from numpy.linalg import inv
import seaborn as sns
from scipy.stats import norm, multivariate_normal
import os
np.random.seed(7)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def hw2q2():
    Ntrain = 100
    data = generateData(Ntrain)
    plot3(data[:, 0], data[:, 1], data[:, 2], name="Training")
    xTrain = data[:, 0:2]
    yTrain = data[:, 2]

    Ntrain = 1000
    data = generateData(Ntrain)
    plot3(data[:, 0], data[:, 1], data[:, 2], name="Validation")
    xValidate = data[:, 0:2]
    yValidate = data[:, 2]

    return xTrain, yTrain, xValidate, yValidate


def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3, .4, .3]  # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:, :, 0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:, :, 1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:, :, 2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    X = generateDataFromGMM(N, gmmParameters)
    return X


def generateDataFromGMM(N, gmmParameters):
    #    Generates N vector samples from the specified mixture of Gaussians
    #    Returns samples and their component labels
    #    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors']  # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]  # Data dimensionality
    C = len(priors)  # Number of components
    X = np.zeros((n, N))
    labels = np.zeros((1, N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1, N))
    thresholds = np.zeros((1, C + 1))
    thresholds[:, 0:C] = np.cumsum(priors)
    thresholds[:, C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:, l]))
        Nl = len(indl[1])
        labels[indl] = (l + 1) * 1
        u[indl] = 1.1
        X[:, indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:, l], covMatrices[:, :, l], Nl))

    # NOTE TRANPOSE TO GO TO SHAPE (N, n)
    return X.transpose()


def plot3(a, b, c, name="Training", mark="o", col="b"):
    # Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a, b, c, marker=mark, color=col)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$y$")
    plt.title("{} Dataset".format(name))
    # To set the axes equal for a 3D plot
    ax.set_box_aspect((np.ptp(a), np.ptp(b), np.ptp(c)))
    plt.show()



#gamma_array = np.arange(3,7,0.5)

#print(gamma_array)

xtrain, ytrain, xvalid, yvalid=hw2q2()


x_t = np.transpose(xtrain)
y_t = np.transpose(ytrain)
x_v = np.transpose(xvalid)
y_v = np.transpose(yvalid)

n_t=x_t[0].size
n_v=x_v[0].size


validate=np.array([np.ones((n_v)),
                    x_v[0],
                    x_v[1],
                    np.power(x_v[0],2),
                    np.power(x_v[1],2),
                    np.multiply(x_v[0],x_v[1]),
                    np.power(x_v[0],3),
                    np.power(x_v[1],3),
                    np.multiply(np.power(x_v[0],2),x_v[1]),
                    np.multiply(x_v[0],np.power(x_v[1],2))])




train=np.array([np.ones((n_t)),
                    x_t[0],
                    x_t[1],
                    np.power(x_t[0],2),
                    np.power(x_t[1],2),
                    np.multiply(x_t[0],x_t[1]),
                    np.power(x_t[0],3),
                    np.power(x_t[1],3),
                    np.multiply(np.power(x_t[0],2),x_t[1]),
                    np.multiply(x_t[0],np.power(x_t[1],2))])

#print(train)
#ML
gamma = np.arange(0.0001,10000,0.1)

print(train.shape)
print(y_t.shape)
#print(y_t)
validate=np.transpose(validate)
train=np.transpose(train)
temp=np.matmul(np.transpose(train),train)
ML=np.matmul(inv(temp),np.transpose(train))
print(ML.shape)
ML=np.matmul(ML,y_t)

errorML=np.mean(np.power(y_t-np.matmul(train,ML),2))
yh=np.matmul(validate,ML)

#print(gamma.size)

#MAP
lmda=train[1].size
lamI=np.identity(lmda)
w_map=np.zeros((10,100000))
MSE_MAP=np.zeros(100000)

for i in range(gamma.size):
    temp=inv(np.multiply(gamma[i],lamI)+np.matmul(np.transpose(train),train))
    temp2=np.matmul(np.transpose(train),y_t)
    w_map[:,i]=np.matmul(temp,temp2)

    temp3=y_v-np.matmul(validate,w_map[:,i])
    MSE_MAP[i]=np.mean(np.power(temp3,2))
    
    
MSE_ML=np.mean(np.power(y_v-yh,2))
print(MSE_ML)
print(gamma)
print(MSE_MAP)
print(gamma.size)
print(MSE_MAP.shape)
MSEmin=min(MSE_MAP)

w_map_min=w_map[:,MSE_MAP==MSEmin]
yh_map=np.matmul(validate,w_map_min)


fig1 = plt.figure(figsize=[4, 3], dpi=200)
ax1 = fig1.add_subplot(111)
ax1.plot(gamma, MSE_MAP, linewidth=1,label="MAP")
ax1.plot(gamma, np.matlib.repmat(MSE_ML,gamma.size,1), linewidth=1,label="ML")
#print(p10[np.argmin(perr)])
#print(p11[np.argmin(perr)])
ax1.set_xlim([0,1000])
ax1.set_ylim([4.6, 5.6])
ax1.set_xlabel('GAMMA')
ax1.set_ylabel('MSE')
ax1.set_title('Question 2')
plt.tight_layout()
plt.legend()
plt.savefig('Q2.jpg')
plt.show()
