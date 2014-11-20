#-*-coding:utf-8-*-
import numpy as np
from recommender_utils import *

class SVD():
    def __init__(self, filePath):
        self.userData = {}
        self.filePath = filePath
    def setParams(self, numFeatures, numIter, learnRate, regVal):
        """
        numFeatures: number of features for both user and item's factor matrix
        numIter: number of iteration
        learnRate: the learn rate
        regVal: the regression value
        """
        self.numFeatures = numFeatures
        self.numIter = numIter
        self.learnRate = learnRate
        self.regVal = regVal
    
    def init(self):
        print 'Initilizing...'
        self.userIDMapper = IDMapper()
        self.itemIDMapper = IDMapper()
        for line in open(self.filePath):
            userID, itemID, score = map(int, line.strip().split('\t')[:3])
            self.userIDMapper.getId(userID)
            self.itemIDMapper.getId(itemID)
            self.userData.setdefault(userID, {})
            self.userData[userID][itemID] = score
        self.userFeatures = np.random.random_sample((self.userIDMapper.getNumIDs(), self.numFeatures))
        self.itemFeatures = np.random.random_sample((self.itemIDMapper.getNumIDs(), self.numFeatures))
        print 'Initilized!'
    
    def train(self):
        """
        Learning with SGD
        """
        for iterCnt in xrange(self.numIter):
            #userIDs = list(self.userData.keys())
            #np.random.shuffle(userIDs)
            for userID in self.userData:
                uId = self.userIDMapper.getId(userID, addNew=False)
                itemIDs = list(self.userData[userID].keys())
                np.random.shuffle(itemIDs)
                for itemID in itemIDs:
                    iId = self.itemIDMapper.getId(itemID, addNew=False)
                    predictValue = np.dot(self.userFeatures[uId], self.itemFeatures[iId])
                    err = self.userData[userID][itemID] - predictValue
                    self.updateFeatures(uId, iId, err)
            print 'Iteration count: %s' % (iterCnt + 1)
    
    def updateFeatures(self, uId, iId, err):
        Pu = self.userFeatures[uId]
        Qi = self.itemFeatures[iId]
        self.userFeatures[uId] += self.learnRate * (err * Qi - self.regVal * Pu)
        self.itemFeatures[iId] += self.learnRate * (err * Pu - self.regVal * Qi)
        
    def predict(self, userID, itemID):
        uID = self.userIDMapper.getId(userID,addNew=False)
        iID = self.itemIDMapper.getId(itemID,addNew=False)
        if uID == -1:
            print 'UserID: %s not in the train set' % userID
            return 0
        if itemID == -1:
            print 'ItemID: %s not in the train set' % itemID
            return 0
        predictValue = np.dot(self.userFeatures[uID], self.itemFeatures[iID])
        if predictValue < 1:
            predictValue = 1
        if predictValue > 5:
            predictValue = 5
        return predictValue

if __name__ == '__main__':
    trainPath = 'D:/DataSet/ml-100k/u1.base'
    testPath = 'D:/DataSet/ml-100k/u1.test'
    svd = SVD(trainPath)
    svd.setParams(30, 15, 0.01, 0.1)
    svd.init()
    svd.train()
    rmse = evaluate(svd, testPath, '\t')
    print 'The RMSE is: %s' % rmse
        