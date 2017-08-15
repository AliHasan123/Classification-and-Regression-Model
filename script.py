import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
import pprint as pprint

def create_List(k, d):
	temp_List = []
	for i in range(k):
		temp_List.append(np.zeros(shape = d))
	return temp_List
	
def generate_Local_Mean_List(X, y_Size, temp_List, k, reconstruct_Indexes):
	local_Mean_List = []
	for i in range(y_Size):
		#print(temp_List[reconstruct_Indexes[i]])
		temp_List[reconstruct_Indexes[i]] = np.vstack((temp_List[reconstruct_Indexes[i]], X[i]))
	#print(temp_List)
	for j in range(k):
		temp_List[j] = np.delete(temp_List[j],(0),axis=0) # remove first array entry that indicates seperation boundary for data linked to specific classes
		local_Mean_List.append(np.mean(temp_List[j],axis=0))
	#print(local_Mean_List)
	return local_Mean_List, temp_List


def ldaLearn(X,y):
	# Inputs
	# X - a N x d matrix with each row corresponding to a training example
	# y - a N x 1 column vector indicating the labels for each training example
	#
	# Outputs
	# means - A d x k matrix containing learnt means for each of the k classes
	# covmat - A single d x d learnt covariance matrix 
	
	# IMPLEMENT THIS METHOD
	
	#print(X)
	d = X[0].shape
	#print(d)
	y_Size = len(y)
	#print(y_Size)
	covmat = np.cov(X, rowvar=0)
	# reconstruct_Indexes used to reconstruct the matrix later on
	unique_Y, reconstruct_Indexes = np.unique(y, return_inverse = True)         # unique_Y has 1, 2, 3, 4, 5
	k = unique_Y.size           # size is 5
	temp_List = []              # list of k class arrays of size d = 2 (2-dimensional)
	local_Mean_List = []
	# creating an empty list of 5 arrays, each of d entries. 
	temp_List = create_List(k, d) # k x d type matrix being generated for computation of means and covariances
	
	# generate local mean list
	local_Mean_List, temp_List = generate_Local_Mean_List(X, y_Size, temp_List, k, reconstruct_Indexes)
	local_Mean_List_Size = len(local_Mean_List)
	means = np.vstack((local_Mean_List[0], local_Mean_List[1])) # combining a tuple of two arrays into one matrix
	for i in range(2, local_Mean_List_Size):
		means = np.vstack((means, local_Mean_List[i]))
	means = np.transpose(means)

	return means,covmat
	
	'''
	print(means)
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	print(covmat)
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	print(means.shape)
	print(covmat.shape) # dimensions confirmed
	'''
	
	

def qdaLearn(X,y):
	# Inputs
	# X - a N x d matrix with each row corresponding to a training example
	# y - a N x 1 column vector indicating the labels for each training example
	#
	# Outputs
	# means - A d x k matrix containing learnt means for each of the k classes
	# covmats - A list of k d x d learnt covariance matrices for each of the k classes
	
	# IMPLEMENT THIS METHOD
	d = X[0].shape
	y_Size = len(y)
	unique_Y, reconstruct_Indexes = np.unique(y, return_inverse = True)         # unique_Y has 1, 2, 3, 4, 5
	k = unique_Y.size           # size is 5
	temp_List = []              # list of k class arrays of size d = 2 (2-dimensional)
	local_Mean_List = []
	# creating an empty list of 5 arrays, each of d entries. 
	temp_List = create_List(k, d)
	temp_List_Size = len(temp_List)
	
	# generate local mean list
	local_Mean_List, temp_List = generate_Local_Mean_List(X, y_Size, temp_List, k, reconstruct_Indexes)
	local_Mean_List_Size = len(local_Mean_List)

	means = np.vstack((local_Mean_List[0], local_Mean_List[1]))
	for i in range(2, local_Mean_List_Size):
		means = np.vstack((means, local_Mean_List[i]))
	means = np.transpose(means)

	
	covmats = None
	for j in range(temp_List_Size):
		if covmats is None:
			covmats = [np.array(np.cov(temp_List[j],rowvar=0))]
		else:
			covmats.append(np.array(np.cov(temp_List[j],rowvar=0)))
	
	'''
	covmats = [np.array(np.cov(temp_List[0],rowvar=0))]
	for j in range(1, temp_List_Size):
		covmats.append(np.array(np.cov(temp_List[j],rowvar=0)))
	
	
	print('####################################################################')    
	print(means)
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	print(covmats)
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	print(means.shape)
	print(covmats.shape)
	'''
	
	return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
	# Inputs
	# means, covmat - parameters of the LDA model
	# Xtest - a N x d matrix with each row corresponding to a test example
	# ytest - a N x 1 column vector indicating the labels for each test example
	# Outputs
	# acc - A scalar accuracy value
	# ypred - N x 1 column vector indicating the predicted labels
	
	# IMPLEMENT THIS METHOD
	
	means = means.transpose()

	dim = len(means)
	pdf_all = []
	ypred = None

	for cur_row in Xtest:
		for cur_mean in means:

			intermediate = 1/(np.power((2*3.142),dim/2)*(np.power(np.linalg.det(covmat),0.5)))
			x_subtracted = np.subtract(cur_row,cur_mean)
			sigma_inverse = np.linalg.inv(covmat)
			
			exp_numerator = -(np.dot(np.dot(x_subtracted, sigma_inverse),np.transpose(x_subtracted)))/2
			pdf = float(np.exp(exp_numerator)*intermediate)

			pdf_all.append(pdf)

		pdf_max = max(pdf_all)

		if ypred is None:
			ypred = np.array(float(pdf_all.index(pdf_max)+1))

		else:
			ypred = np.vstack((ypred,float(pdf_all.index(pdf_max)+1)))

		#if ypred == ytest:
		#   y_bool = True
		#else:
		#   y_bool = False

		pdf_all = []
		pdf_max = []

	acc = str(100*np.mean((np.equal(ytest,ypred)).astype(float)))

	return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
	# Inputs
	# means, covmats - parameters of the QDA model
	# Xtest - a N x d matrix with each row corresponding to a test example
	# ytest - a N x 1 column vector indicating the labels for each test example
	# Outputs
	# acc - A scalar accuracy value
	# ypred - N x 1 column vector indicating the predicted labels

	# IMPLEMENT THIS METHOD

	pdf_all = []
	ypred = None

	dim = len(means)
	means = means.transpose()
	
	for cur_row in Xtest:
		for x in range(len(means)):

			intermediate = 1 / (np.power((2 * 3.142), dim/2) * np.power(np.linalg.det(covmats[x]),0.5))
			x_subtrctd = np.subtract(cur_row,means[x])
			sigma_inverse = np.linalg.inv(covmats[x])
			
			ans = - np.dot(np.dot(x_subtrctd, sigma_inverse), np.transpose(x_subtrctd)) / 2	
			pdf = float(intermediate * np.exp(ans))
			pdf_all.append(pdf)

		pdf_max = max(pdf_all)
	
		if ypred is None:
			ypred = np.array(float(pdf_all.index(pdf_max)+1))
		else:
			ypred = np.vstack((ypred, float(pdf_all.index(pdf_max)+1)))
			
		pdf_all = []
		pdf_max = []
		
	acc = 100*np.mean((ypred == ytest).astype(float))
	return acc, ypred

	
	'''
	pdf = []
	initialPrediction = None

	float_val = float(1/1)
	
	#number of rows in the 'means' matrix
	rowCount = len(means)
	
	#transpose the means matrix
	means = means.transpose()
	transposedCount = len(means)
	
	for cur_row in Xtest:
		for index in range(rowCount):
			mean   = means[index]
			covmat = covmats[index]
			
			#Determinant of the covariance matrix    
			covmatDeterminant = np.linalg.det(covmat)
			
			#To be used in calculating the significant**********************
			v1 = np.power(6.284, (rowCount/2))
			v2 = np.power(covmatDeterminant, 0.5)
			
			#TODO: Rename this variable ****************************************
			significant = 1/( v1 * v2)
			
			#try np.subtract(row,mean)
			subtraction = np.subtract(cur_row,mean)
			reciprocal  = np.linalg.inv(covmat)
			
			product = -(np.dot(np.dot(subtraction, reciprocal), np.transpose(subtraction)))/2
			
			exponential = np.exp(product)

			pdf.append(float_val * significant * exponential)
		
		maximumVal = max(pdf)
		
		#This returns location of first instance of maximumVal in the pdf
		firstInstance = pdf.index(maximumVal)
		
		if initialPrediction is None:
			labelPrediction = np.array(float(firstInstance + 1))
		else:
			labelPrediction = np.vstack((labelPrediction, float(firstInstance + 1)))
		
		#clear pdf
		#del pdf[:]
		pdf = []
		maximumVal = []

		
	ypred = labelPrediction
	acc = 100*np.mean((ypred == ytest).astype(float))
	
	return acc,ypred
	'''

def learnOLERegression(X,y):
	# Inputs:                                                         
	# X = N x d 
	# y = N x 1                                                               
	# Output: 
	# w = d x 1 
	
	# IMPLEMENT THIS METHOD
	return np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))

def learnRidgeRegression(X,y,lambd):
	# Inputs:
	# X = N x d                                                               
	# y = N x 1 
	# lambd = ridge parameter (scalar)
	# Output:                                                                  
	# w = d x 1                                                                

	# IMPLEMENT THIS METHOD 
	i = lambd*X.shape[0]*np.identity(X.shape[1])
	j = np.dot(X.T,X)
	return np.dot(np.linalg.inv(i+j), np.dot(X.T,y))                                             

def testOLERegression(w,Xtest,ytest):
	# Inputs:
	# w = d x 1
	# Xtest = N x d
	# ytest = X x 1
	# Output:
	# mse
	
	# IMPLEMENT THIS METHOD  
	a = (ytest-np.dot(Xtest,w))
	a_matrix = np.dot(a.T,a)
	return np.sqrt(a_matrix.sum(axis=0)) / (Xtest.shape[0])

def regressionObjVal(w, X, y, lambd):

	# compute squared error (scalar) and gradient of squared error with respect
	# to w (vector) for the given data X and y and the regularization parameter
	# lambda                                                                  

	# IMPLEMENT THIS METHOD
   
	i = (np.matrix(np.dot(X,w))).T
	j = y - i
	j_star = np.dot(j.T,j)

	error = (lambd*np.dot(w.T,w)/2)+(j_star.sum(axis=0)/(2*X.shape[0]))

	j = np.dot(y.T,X)
	i = np.dot(X.T,X)
	k = np.matrix(np.dot(w.T,i))

	error_grad = np.squeeze(np.asarray( ((k-j)/X.shape[0]) + (np.matrix(np.dot(lambd,w))) ))
									   
	return error, error_grad

def mapNonLinear(x,p):
	# Inputs:                                                                  
	# x - a single column vector (N x 1)                                       
	# p - integer (>= 0)                                                       
	# Outputs:                                                                 
	# Xd - (N x (d+1)) 
	
	# IMPLEMENT THIS METHOD
	N = x.shape[0]
	y = np.matrix(np.ones([N,]))

	for j in range(1+p):
		if j!=0:

			pow = np.matrix(np.power(x,j))
			y = np.concatenate((y,pow),axis=0)

	y = y.T      # transpose    
	return y

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
	X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
	X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
	X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
	X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
	w_l = learnRidgeRegression(X_i,y,lambd)
	mses3_train[i] = testOLERegression(w_l,X_i,y)
	mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
	i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
	args = (X_i, y, lambd)
	w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
	w_l = np.transpose(np.array(w_l.x))
	w_l = np.reshape(w_l,[len(w_l),1])
	mses4_train[i] = testOLERegression(w_l,X_i,y)
	mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
	i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
	Xd = mapNonLinear(X[:,2],p)
	Xdtest = mapNonLinear(Xtest[:,2],p)
	w_d1 = learnRidgeRegression(Xd,y,0)
	mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
	mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
	w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
	mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
	mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
