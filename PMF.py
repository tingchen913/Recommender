#-*-coding:utf-8-*-
"""
Reference paper:
    "Probabilistic Matrix Factorization"
    Ruslan Salakhutdinov and Andriy Mnih
"""
import numpy as np
import time

from recommender_utils import *
class PMF():
    def __init__(self, trainPath):
        self.userData = {} # to storage the ratings as this format: {userID: {itemID: score}}
        self.trainPath = trainPath
            
    def setParams(self, **params):
        self.learnRate = params.get('learnRate', 0.01)
        self.regU = params.get('regU', 0.05)
        self.regI = params.get('regI', 0.06)
        self.numFeatures = params.get('numFeatures', 30)
        self.iterNum = params.get('iterNum', 15)
        print 'learnRate: %s, regU: %s, regI: %s, numFeatures: %s, iterNum: %s'\
              %(self.learnRate, self.regU, self.regI, self.numFeatures, self.iterNum)
        
    def loadData(self, delimiter):
        """
        delimiters for example: '\t' ',' ','
        """
        self.maxScore = 0
        self.minScore = np.inf
        self.userIDMapper = IDMapper()
        self.itemIDMapper = IDMapper()
        for line in open(self.trainPath):
            lineArr = line.strip().split(delimiter)
            userID, itemID, score = map(int, lineArr[:3])
            self.userIDMapper.getId(userID)
            self.itemIDMapper.getId(itemID)
            self.userData.setdefault(userID, {})
            self.userData[userID][itemID] = score
            if score < self.minScore:
                self.minScore = score
            if score > self.maxScore:
                self.maxScore = score
        print 'Load data from %s done! \n maxScore: %s, minScore: %s' % (self.trainPath, self.maxScore, self.minScore)
        
    def init(self):
        """
        I take the advise: 0.1 * rand(0, 1) / sqrt(dim) to initilize all the users and items' factors.
        """
        self.userFeatures = np.random.random((self.userIDMapper.getNumIDs(), self.numFeatures)) * (0.1 / np.sqrt(self.numFeatures))
        self.itemFeatures = np.random.random((self.itemIDMapper.getNumIDs(), self.numFeatures)) * (0.1 / np.sqrt(self.numFeatures))
        
    def train(self):
        """
        Train with Stochastic Gradient Decent(SGD)
        """
        for iteration in xrange(self.iterNum):
            startTime = time.clock()
            for userID in self.userData.keys():
                uId = self.userIDMapper.getId(userID, addNew=False)
                itemIDs = list(self.userData[userID].keys())
                np.random.shuffle(itemIDs)
                for itemID in itemIDs:
                    iId = self.itemIDMapper.getId(itemID, addNew=False)
                    estimatedPref = np.dot(self.userFeatures[uId], self.itemFeatures[iId])
                    err = self.userData[userID][itemID] - estimatedPref
                    self.updateFactors(uId, iId, err)
            endTime = time.clock()
            print 'Iterations: %s, Use: %s seconds' % (iteration+1, endTime-startTime)
            
    def updateFactors(self, uId, iId, err):
        userFeature = self.userFeatures[uId]
        itemFeature = self.itemFeatures[iId]
        self.userFeatures[uId] += self.learnRate * (err * itemFeature - self.regU * userFeature)
        self.itemFeatures[iId] += self.learnRate * (err * userFeature - self.regI * itemFeature)
        
    def predict(self, userID, itemID):
        uId = self.userIDMapper.getId(userID, addNew=False)
        iId = self.itemIDMapper.getId(itemID, addNew=False)
        if uId == -1:
            print 'User ID: %s, exists in the test set but not in the train set' % userID
            return 0
        if iId == -1:
            print 'Item ID %s, exists in the test set but not in the train set' % itemID
            return 0
        estimatedPref = np.dot(self.userFeatures[uId], self.itemFeatures[iId])
        if estimatedPref > self.maxScore:
            estimatedPref = self.maxScore
        if estimatedPref < self.minScore:
            estimatedPref = self.minScore
        return estimatedPref
    
if __name__ == '__main__':
    trainPath = 'D:/DataSet/ml-1m/ml-1m/trainSet.txt'
    testPath = 'D:/DataSet/ml-1m/ml-1m/testSet.txt'
    outPath = 'D:/DataSet/ml-1m/ml-1m/PMF_Features.txt'
    numFs = [10, 20, 30, 40, 50, 100]
    for numF in numFs:    
        pmf = PMF(trainPath)
        kparams = {'learnRate':0.01, 'regU':0.09, 'regI':0.04, 'numFeatures':numF, 'iterNum':15}
        pmf.setParams(**kparams)
        pmf.loadData(',')
        pmf.init()
        pmf.train()
        rmse = evaluate(pmf, testPath, ',')
        writeResult(outPath, '%s,%s'%(numF, rmse))
    print 'All work done!'
    
                
        
        