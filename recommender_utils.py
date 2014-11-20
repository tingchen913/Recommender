import numpy as np
def evaluate(predictor, testPath, delimiter):
    se = 0.0
    n = 0
    for line in open(testPath):
        lineArr = line.strip().split(delimiter)
        userID, itemID, score = map(int, lineArr[:3])
        estimatedPref = predictor.predict(userID, itemID)
        if estimatedPref:
            se += np.power(estimatedPref - score, 2)
            n += 1
            #print 'square error: %s, nonzero count: %s' % (se, n)
    rmse = np.sqrt(se / n)
    print 'Evaluate done!'
    return rmse

def writeResult(outPath, line):
    outFile = open(outPath, 'a+')
    outFile.write(line + '\n')
    outFile.close()
	
class IDMapper():
    """
    Mapping raw IDs to array rows' indices, 
    with this class the recommender is able to process abitary raw users and items' IDs.
    """
    def __init__(self):
        self.rawID2ID = {}
        self.rawIDs = []      
    def getId(self, rawID, addNew=True):
        """
        return the row id according to the raw user or item's ID.
        I use a flag-addNew here to control wether to add new ID into this mapper if the new ID is not exists.
        """
        ID = self.rawID2ID.get(rawID, -1)
        if ID == -1 and addNew:
            ID = len(self.rawIDs)
            self.rawID2ID[rawID]=ID
            self.rawIDs.append(rawID)
        return ID
    
    def getRawID(self, ID):
        return self.rawIDs[ID]
    
    def getNumIDs(self):
        """
        return the number of IDs comprised in this mapper.
        """
        return len(self.rawIDs)
    