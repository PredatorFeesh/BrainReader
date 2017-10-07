import os
import numpy as np
from functools import reduce
from flask import Flask, request, redirect, url_for, render_template, Blueprint

api = Blueprint('api', __name__)

class utils(object):
    def __init__(self):
        pass

    def bigSum(self, start, end, f):  # f passed as lambda
        s = 0
        if start > end:
            return s
        for i in range(start, end + 1):
            s += f(i)
        return s

    def bigProduct(self, start, end, f):
        p = 1
        if start > end:
            return 0
        for i in range(start, end + 1):
            p *= f(i)
        return p

    def mean(self, x):
        n = len(x)
        return (1 / n) * self.bigSum(start=0, end=n - 1, f=lambda i: x[i])

    def variance(self, x):
        n = len(x)
        m = self.mean(x)
        return ((1 / (n - 1))) * self.bigSum(start=0, end=n - 1, f=lambda i: (x[i] - m) ** 2)

    def split_fal(self, data, num_points=200, labels=None,
                  processing=True):  # This returns the data. LABELS MUST BE CONSECUTIVE. NO MIXING LABELS. IE. labels = [1,2,1] NOT GOOD.
        # features are in [sigma, delta, mean], given num_points points of features
        # Reset the features and feature_labels to avoid errors.
        # Example data: [ [1,2......,2], [...] ]
        # Example label: [0, 1]
        lenD = len(data)
        f = []
        l = []
        # Need to know how many data
        if labels is not None:
            times = len(labels)  # This makes sure we iter all labels, beacause of how data is
            step = int(times / num_points)  # The number of points between each two. Taking lower in case not divisible.
            for ii in range(num_points - 1):  # In between points...
                start = step * ii
                end = step * ii + step  # Range data
                if labels[start] != labels[end] or start == end: continue
                mean = self.mean(x=data[start:end])
                var = self.variance(x=data[start:end])
                delt = data[end] - data[start]
                feat = [mean, var, delt]
                label = labels[start]
                f.append(feat)
                l.append(label)
            if not processing:  # Is not processing, we return what seen under. processing = True when used by probabilities func.
                return [f, l]  # looks something like f = [ [l1,f2,l3],[...],[...] ,[...]], l = [0,0,1,1]
            else:  # Size of the labels and features should be equal. Otherwise, nasty error would have occured already.
                fN = []
                fR = []
                fL = []
                for y, label in enumerate(l):
                    if label == l[
                                y - 1 if y != 0 else 0]:  # So we don't start off with an error. Last element and first elemnt most likely different.
                        fN.append(f[y])
                    else:
                        fR.append(fN)
                        fN = [f[y]]
                        fL.append(l[y - 1])
                return [fR, fL]  # Should return like this [ [  [[...],[...],[...]],[[...],[....],[....]]   ,  [0,1] ]
        else:
            step = int(lenD / num_points)  # The number of points between each two. Taking lower in case not divisible.
            for ii in range(num_points - 1):  # In between points...
                start = step * ii
                end = step * ii + step + 1  # Range data
                mean = self.mean(x=data[start:end])
                var = self.variance(x=data[start:end])
                delt = data[end] - data[start]
                feat = [mean, var, delt]
                f.append(feat)
            return f  # Returns like f=[ [...],[...] ]
            # ------------------------------------------------------------


'''class regress(object):
	def __init__(self, m): 
		self.w = []
		self.m = m
		self.utils = utils()
    
	def y(self, x):
		return bigSum(0,self.m, lambda j: self.w[j]*x**j)
    
	def error(self, x, t):
		return bigSum(0, len(x)-1, lambda n: (self.y(M,x[n]) - t[n])**2 )
    
	def errorRMS(self,x, t):
		return ( 2*error(M, x, t) / len(x) )**0.5
    
	def train(self, x, y): # M is the order, N is number of points, t = training Y, x = training X
		# N = number of points
		# General formula is: bigSum(0,M, lambda j: w[j]*bigSum(1, N, lambda n: x[n]**(i+j))) = bigSum(t[n]*x[n]**i)
		# Or, if set vars to:
		# A[i][j] = bigSum(1,N lambda n: w[j]*x[n]**(i+j) )
		# T[i] = bigSum(1,N, lambda n: t[n]*x**i)
    
		# THIS IS A PURE LINEAR REGRESSION STYLE OF LEARNING. NO BAYSEIAN OR FREQUENTIST APPROACH
		N = len(x)
		self.w = [0 for i in range(self.m+1)]
		s = lambda k: self.utils.bigSum(0, N-1, lambda i:x[i]**k)
		S = np.array( [ s(ii) for ii in range(0, 2*self.m+1)] )
		sA = []
		for i in range(0, self.m+1):
			sA.append([ S[ii] for ii in range(i,self.m+i+1) ])
		sA = np.array(sA)
			  
		t = lambda k: self.utils.bigSum(0, N-1, lambda i: y[i]*x[i]**k)
		T = np.array([ t(ii) for ii in range(0,self.m+1) ])
		print(T.shape, sA.shape)
		results = np.linalg.solve(sA,T)
		for i in range(0,len(self.w)):
			self.w[i] = results[i]'''


class channel(object):
    def __init__(self):
        self.utils = utils()
        self.data = []  # Data = [ [....],[.....] ]
        self.labels = []  # Labels =  [0,1,1,1,1]
        self.num_points = 200  # NEEDS TO BE EQUAL FOR EVERY CHANNEL

        self.features = []  # ONLY EDIT BY PROGRAM
        self.feature_labels = []  # ONLY EDIT BY PROGRAM

    # LABEL CORRESPONDS TO FEATURE LABEL. FEATURES = [ [[],[],[],[],[,][],[]], [[],[],[],[],[]] ]
    #                                     LABEL = [0,1] LABELS HAVE NO REPEATING VALUES

    def set_data(self, data,
                 labels):  # Can't add same label twice. Label is int. This is NOT to set features. Only the overall data and labels.
        # Don't skip this function. It is very important. Make sure it is set like this:
        # data = [[data_1],[data_2] ]
        # label = [ label_1, label_2 ]. MAKE SURE NO LABELS ARE THE SAME. VERY IMPORTANT. RUINS EVERYTHING.
        self.data = data
        self.labels = labels
        self.features = []
        self.feature_labels = []
        self.split_fal()

    def dist(self, f1, f2):
        return np.sum(np.abs(np.array(f2) - np.array(f1)), dtype=np.int32)

    def split_fal(self):  # fal = features and labels. Based on self.num_points
        d = self.utils.split_fal(num_points=self.num_points, data=self.data, labels=self.labels)
        self.features = d[0]
        self.feature_labels = d[1]

    # WE ARE NOT LOOKING FOR THE MAX. WE ARE LOOKING FOR THE MIN DISTANCE. THE VARIABLES ARE CALLED MAX. DON'T GET CONFUSED.
    # Just feed it the data from the channel as newFeat. newFeat should look like regular data [.......]. Split for you newFeat should be data.
    def getProbabilities(self,
                         newFeat,
                         retNearest=False):  # Make sure that len(newFeat) = len of each featrue set. If not, we will make len of newFeat = len(self.feature[0])		newFeat = self.utils.split_fal(newFeat, labels=None, processing=True)
        mines = []  # Array appended into it as [dist, l] where l is label and dist is distance. This should have the closest feature (and label) to each feature in newFeat.
        newFeat = self.utils.split_fal(data=newFeat)
        for f in newFeat:  # Iterate each feature in newFeatures . First loop.
            # minIF basically stores the min of each label. len(minIF) should equal the # of labels
            minIF = []  # Mins in features. We store in this list before appending it to maxes.
            for l in range(len(self.feature_labels)):  # Going through each label. l = label ID as number. Second loop.
                # Remember that self.features looks like this: [ [[],[],[],[],[],[],[]],[[],[],[],[],],[],[],[],[]] ] where the len(self.features[x]) is associated with label x
                minn = self.dist(self.features[l][0],
                                 f)  # minn is the distance between the current feature being examined from newFeat and features.  This is a placeholder (first value).
                for oF in self.features[l]:  # oF is features in the CURRENT feature label. Third loop.
                    if (minn - self.dist(oF, f)) > 0:
                        minn = self.dist(oF,
                                         f)  # Here is where we get to the juice. We iter all the features in the label, and find the minimum
                        # This gets the min of the current label.
                minIF.append(minn)  # The minIF works ----
                # Process the maxes IF. Go through each and find the min, and associate it with the label
            finalMin = minIF[0]
            finalLabel = 0
            '''for i,m in enumerate(minIF):
                if f > m:
                    finalLabel = i
                    finalMin = m
            mines.append([finalMin, finalLabel])'''
            for i in range(1, len(minIF)):
                if (minIF[i] - finalMin) < 0:
                    finalMin = minIF[i]
                    finalLabel = i
            mines.append([finalMin, finalLabel])  # tested, works ----
            # YES, so at this point, maxes has the array with greatest dist of each in the newFeat, with the label
        # To get the highest probability label, return the label which pops up most in maxes[:][1]
        if retNearest:
            return mines
        l = len(self.feature_labels)
        timesLabelShown = [0 for i in range(l)]  # This depends on the fact that labels is a set.
        for minn in mines[:]:  # Iterates all the finalLabel in mines. We need to count the # of times each label shown.
            mr = minn[1]
            timesLabelShown[mr] += 1  # This organizes it into the times each label is shown.
            # Now, we need to get the probability. To do this, we do the timesLabelShown[i]/len(self.features[i])
        probabilities = [0 for i in range(l)]
        for i in range(l):
            probabilities[i] = (timesLabelShown[i]) / len(newFeat)
        return probabilities

    def getNearestLabel(self, newFeat):  # Return label index with highest probability
        probs = self.getProbabilities(newFeat)
        im = 0
        vm = probs[0]
        for i, v in enumerate(probs):
            if vm > v:
                vm = v
                im = i
        return im

        # SAVE THE PROP TIME BY USING JUST A SPECIFIC WELL KNOWN TO WORK CHANNEL INSTEAD OF USING THE MANAGER


class class_manager(object):  # CHANNEL'S APPEND RESETS THE DATA, FEATURES AND LABELS
    def __init__(self):
        self.channels = []
        self.names = []
        self.catagories = []
    # THE API
    # ------------------------------------------------------------
    def append(self, channel):  # Make sure that all data is the same length in each channel
        # If new channel is 8000, whereas all others are 9000, make all the 9000 in 8000
        # Loop through the list to find the minimum
        self.channels.append(channel)
    def setCatagoriesAndNames(self, catagories, names):
        self.catagories = catagories
        self.names = names
    def mostLikelyLabel(self,
                        data):  # DATAg MUST LOOK LIKE [[],[],[],[]] WITH THE 8000(maybe) NUMBERS FROM EACH CHANNEL IN []
        cProbs = []
        # The probs are equal for each
        for i in range(len(self.channels)):
            cProbs.append(np.array(self.channels[i].getProbabilities(data)))
            # Now we need to multiple these things together
        probs = reduce((lambda x, y: x * y), cProbs)
        print(probs)
        # Now we find the max probability
        ii = 0
        iv = probs[0]
        for i, v in enumerate(probs):
            if iv > v:
                iv = v
                ii = i
        return ii




