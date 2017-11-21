import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def column(matrix, i):
    return [row[i] for row in matrix]


def IG(D, index, value):
    """Compute the Information Gain of a split on attribute index at value
    for dataset D.
    
    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the Information Gain for the given split
    """
    # This is to get columns
    # D[D[:, index] <= value], D[D[:, index] > value], index)


    colXofData = column(D[0], index)
    columnofNums = D[1]
    classesZero = []
    classesOne = []
    dataZero = []
    dataOne = []

    OriginaltrueCount = 0
    OriginalfalseCount = 0
    for x in range(0, len(D[1])):
        if (D[1][x] == 0):
            OriginaltrueCount += 1
        elif (D[1][x] == 1):
            OriginalfalseCount += 1

    op1 = OriginaltrueCount / (OriginaltrueCount + OriginalfalseCount)
    op2 = OriginalfalseCount / (OriginaltrueCount + OriginalfalseCount)
    XY = - (op1 * np.log2(op1) + ((op2) * np.log2((op2))))
    print("This is the Entropy of the entire dataset: {}".format(XY))

    # This is used for the split-entropy
    for x in range(0, len(columnofNums)):
        if(colXofData[x] >= value):
            dataZero.append(colXofData[x])
            classesZero.append(0)
        elif(colXofData[x] < value):
            dataOne.append(colXofData[x])
            classesOne.append(1)


    trueCount = len(dataZero)
    falseCount = len(dataOne)
    datasetNP = np.array(colXofData)
    classesNP = np.array(columnofNums)
    Yes = 0
    No = 0
    Yes = len(classesZero)
    No = len(classesOne)

    # Calculate entropy of dataset
        # p1 represents Dy
        # p2 represents Dn
    p1 = trueCount / (trueCount + falseCount)
    p2 = falseCount / (trueCount + falseCount)

    # Calculate Split-Entropy for dataset
    DYDN = - (((Yes)/(Yes+No))*p1*np.log2(p1) + ((No)/(Yes+No)) * p2*np.log2(p2))
    #print("This is the Split-Entropy: {}".format(DYDN))

    #Calculate Information Gain
    informationGain = XY - DYDN
    print("This is the Information Gain: {}".format(informationGain))
    return informationGain


def G(D, index, value):
    """Compute the Gini index of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the Gini index for the given split
    """

    trueCount = 0
    falseCount = 0
    for x in range(0, len(D[1])):
        if (D[1][x] == 0):
            trueCount += 1
        elif (D[1][x] == 1):
            falseCount += 1

    p1 = trueCount / (trueCount + falseCount)
    p2 = falseCount / (trueCount + falseCount)

    colXofData = column(D[0], index)
    columnofNums = D[1]
    classesZero = []
    classesOne = []
    dataZero = []
    dataOne = []

    for x in range(0, len(columnofNums)):
        if(colXofData[x] >= value):
            dataZero.append(colXofData[x])
            classesZero.append(0)
        elif(colXofData[x] < value):
            dataOne.append(colXofData[x])
            classesOne.append(1)

    trueCount = len(dataZero)
    falseCount = len(dataOne)

    Yes = 0
    No = 0
    Yes = len(classesZero)
    No = len(classesOne)
    yesPercent = Yes/(Yes+No)
    noPercent = No/(Yes+No)

    GiniDY = (yesPercent)*(1 - (yesPercent*yesPercent))
    GiniDN = (noPercent)*(1 - (noPercent*noPercent))
    GiniIndex = GiniDY + GiniDN
    print("This is the Gini Index: {}".format(GiniIndex))
    return GiniIndex

def CART(D, index, value):
    """Compute the CART measure of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the CART measure for the given split
    """
    trueCount = 0
    falseCount = 0
    for x in range(0, len(D[1])):
        if (D[1][x] == 0):
            trueCount += 1
        elif (D[1][x] == 1):
            falseCount += 1

    p1 = trueCount / (trueCount + falseCount)
    p2 = falseCount / (trueCount + falseCount)

    colXofData = column(D[0], index)
    columnofNums = D[1]
    classesZero = []
    classesOne = []
    dataZero = []
    dataOne = []

    for x in range(0, len(columnofNums)):
        if (colXofData[x] >= value):
            dataZero.append(colXofData[x])
            classesZero.append(0)
        elif (colXofData[x] < value):
            dataOne.append(colXofData[x])
            classesOne.append(1)

    trueCount = len(dataZero)
    falseCount = len(dataOne)
    Yes = 0
    No = 0

    Yes = len(classesZero)
    No = len(classesOne)
    yesPercent = Yes / (Yes + No)
    noPercent = No / (Yes + No)

    CARTMeasure = 2*(yesPercent*noPercent)*((abs(p1 - p2) + (abs(p1-p2))))
    print("This is the CART: {}".format(CARTMeasure))
    return CART
def bestSplit(D, criterion):
    """Computes the best split for dataset D using the specified criterion

    Args:
        D: A dataset, tuple (X, y) where X is the data, y the classes
        criterion: one of "IG", "GINI", "CART"

    Returns:
        A tuple (i, value) where i is the index of the attribute to split at value
    """

    #functions are first class objects in python, so let's refer to our desired criterion by a single name
    if(criterion == "IG"):
        infoGainList = []
        indexlist = []
        valuelist = []
        columnofNums = D[1]

        for index in range(0, 10):
            colXofData = column(D[0], index)
            classesZero = []
            classesOne = []
            dataZero = []
            dataOne = []
            OriginaltrueCount = 0
            OriginalfalseCount = 0
            for x in range(0, len(D[1])):
                if (D[1][x] == 0):
                    OriginaltrueCount += 1
                elif (D[1][x] == 1):
                    OriginalfalseCount += 1
            for value in range(int(min(colXofData)), int(max(colXofData))):
                op1 = OriginaltrueCount / (OriginaltrueCount + OriginalfalseCount)
                op2 = OriginalfalseCount / (OriginaltrueCount + OriginalfalseCount)
                XY = - (op1 * np.log2(op1) + ((op2) * np.log2((op2))))
                #print("This is the Entropy of the entire dataset: {}".format(XY))

                # This is used for the split-entropy
                for x in range(0, len(colXofData)):
                    if (colXofData[x] >= value):
                        dataZero.append(colXofData[x])
                        classesZero.append(0)
                    elif (colXofData[x] < value):
                        dataOne.append(colXofData[x])
                        classesOne.append(1)

                trueCount = len(dataZero)
                falseCount = len(dataOne)
                Yes = 0
                No = 0
                Yes = len(classesZero)
                No = len(classesOne)
                if (Yes > 0 and No > 0):
                    #print(Yes)
                    #print(No)
                    # Calculate entropy of dataset
                    # p1 represents Dy
                    # p2 represents Dn
                    p1 = trueCount / (trueCount + falseCount)
                    p2 = falseCount / (trueCount + falseCount)

                    # Calculate Split-Entropy for dataset
                    DYDN = - (((Yes) / (Yes + No)) * p1 * np.log2(p1) + ((No) / (Yes + No)) * p2 * np.log2(p2))
                    #print("This is the Split-Entropy: {}".format(DYDN))

                    # Calculate Information Gain
                    informationGain = XY - DYDN
                    infoGainList.append(informationGain)
                    indexlist.append(index)
                    valuelist.append(value)
                    #print("This is the Information Gain: {}".format(informationGain))
                    #print("This is the index: {}".format(index))
                    #print("This is the value: {}".format(value))
        infoGain = infoGainList.index(max(infoGainList))
        #print(infoGainList[infoGain])
        i = indexlist[infoGain]
        value = valuelist[infoGain]
        return (i, value)


    if(criterion == "GINI"):
        trueCount = 0
        falseCount = 0
        giniList = []
        indexlist = []
        valuelist = []
        for x in range(0, len(D[1])):
            if (D[1][x] == 0):
                trueCount += 1
            elif (D[1][x] == 1):
                falseCount += 1
            p1 = trueCount / (trueCount + falseCount)
            p2 = falseCount / (trueCount + falseCount)

        for index in range(0, 10):
            colXofData = column(D[0], index)
            columnofNums = D[1]
            classesZero = []
            classesOne = []
            dataZero = []
            dataOne = []

            for value in range(int(min(colXofData)), int(max(colXofData))):
                if (colXofData[x] >= value):
                    dataZero.append(colXofData[x])
                    classesZero.append(0)
                elif (colXofData[x] < value):
                    dataOne.append(colXofData[x])
                    classesOne.append(1)

                trueCount = len(dataZero)
                falseCount = len(dataOne)
                Yes = 0
                No = 0
                Yes = len(classesZero)
                No = len(classesOne)
                if Yes > 0 and No > 0:
                    yesPercent = Yes / (Yes + No)
                    noPercent = No / (Yes + No)

                    GiniDY = (yesPercent) * (1 - (yesPercent * yesPercent))
                    GiniDN = (noPercent) * (1 - (noPercent * noPercent))
                    GiniIndex = GiniDY + GiniDN
                    giniList.append(GiniIndex)
                    indexlist.append(index)
                    valuelist.append(value)
                    #print("This is the Gini Index: {}".format(GiniIndex))
                    #print("This is the index: {}".format(index))
                    #print("This is the value: {}".format(value))
        Gini = giniList.index(min(giniList))
        i = indexlist[Gini]
        value = valuelist[Gini]
        return (i, value)

    if(criterion == "CART"):
        trueCount = 0
        falseCount = 0
        CARTList = []
        indexlist = []
        valuelist = []
        for x in range(0, len(D[1])):
            if (D[1][x] == 0):
                trueCount += 1
            elif (D[1][x] == 1):
                falseCount += 1

        p1 = trueCount / (trueCount + falseCount)
        p2 = falseCount / (trueCount + falseCount)
        for index in range(0, 10):
            colXofData = column(D[0], index)
            columnofNums = D[1]
            classesZero = []
            classesOne = []
            dataZero = []
            dataOne = []

            for value in range(int(min(colXofData)), int(max(colXofData))):
                if (colXofData[x] >= value):
                    dataZero.append(colXofData[x])
                    classesZero.append(0)
                elif (colXofData[x] < value):
                    dataOne.append(colXofData[x])
                    classesOne.append(1)
                trueCount = len(dataZero)
                falseCount = len(dataOne)
                truePercent = trueCount/(trueCount+falseCount)
                falsePercent = falseCount/(trueCount+falseCount)
                Yes = 0
                No = 0
                Yes = len(classesZero)
                No = len(classesOne)
                if Yes > 0 and No > 0:
                    yesPercent = Yes / (Yes + No)
                    noPercent = No / (Yes + No)
                    CARTMeasure = 2 * (truePercent * falsePercent) * ((abs(yesPercent - noPercent) + (abs(yesPercent - noPercent))))
                    #print("This is the CART: {}".format(CARTMeasure))
                    CARTList.append(CARTMeasure)
                    indexlist.append(index)
                    valuelist.append(value)
        Cartnum = CARTList.index(max(CARTList))
        #print(CARTList[Cartnum])
        i = indexlist[Cartnum]
        value = valuelist[Cartnum]
        return (i, value)

def load(filename):
    """Loads filename as a dataset. Assumes the last column is classes, and 
    observations are organized as rows.

    Args:
        filename: file to read

    Returns:
        A tuple D=(X,y), where X is a list or numpy ndarray of observation attributes
        where X[i] comes from the i-th row in filename; y is a list or ndarray of 
        the classes of the observations, in the same order
    """
    with open(filename) as f:
        dataset = f.readlines()
    dataset = [x.strip().split() for x in dataset]
    dataset = [[float((float(j))) for j in i] for i in dataset]
    for x in range(0, len(dataset)):
        dataset[x][0] = int(dataset[x][0])
        dataset[x][1] = int(dataset[x][1])
        dataset[x][2] = int(dataset[x][2])
        dataset[x][3] = int(dataset[x][3])
        dataset[x][4] = int(dataset[x][4])
        dataset[x][5] = int(dataset[x][5])
        #dataset[x][6] = int(dataset[x][6])
        dataset[x][7] = int(dataset[x][7])
        dataset[x][8] = int(dataset[x][8])
        dataset[x][9] = int(dataset[x][9])
        dataset[x][10] = int(dataset[x][10])

    classes = []
    for x in range(0, len(dataset)):
        classes.append(dataset[x][10])
        dataset[x] = dataset[x][:-1]
    X = (dataset, classes)
    return X

def classifyIG(train, test):
    """Builds a single-split decision tree using the Information Gain criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """
    X = bestSplit(train, "IG")
    bestSplitColTrain = X[0]
    bestSplitValueTrain = X[1]
    allValuesTrain = train[0] # list of lists
    allClassesTrain = train[1]

    allClassesTest = test[1]

    data = []
    classes = []

    colXofData = column(test[0], bestSplitColTrain)
    for x in range(0, len(allClassesTest)):
        if(colXofData[x] >= bestSplitValueTrain):
            data.append(colXofData[x])
            classes.append(0)
        elif(colXofData[x] < bestSplitValueTrain):
            data.append(colXofData[x])
            classes.append(1)
    return(classes)


def classifyG(train, test):
    """Builds a single-split decision tree using the GINI criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """
    X = bestSplit(train, "GINI")
    bestSplitColTrain = X[0]
    bestSplitValueTrain = X[1]
    allValuesTrain = train[0]  # list of lists
    allClassesTrain = train[1]

    allClassesTest = test[1]

    data = []
    classes = []

    colXofData = column(test[0], bestSplitColTrain)
    for x in range(0, len(allClassesTest)):
        if (colXofData[x] >= bestSplitValueTrain):
            data.append(colXofData[x])
            classes.append(0)
        elif (colXofData[x] < bestSplitValueTrain):
            data.append(colXofData[x])
            classes.append(1)
    return(classes)


def classifyCART(train, test):
    """Builds a single-split decision tree using the CART criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """
    X = bestSplit(train, "CART")
    bestSplitColTrain = X[0]
    bestSplitValueTrain = X[1]
    allValuesTrain = train[0]  # list of lists
    allClassesTrain = train[1]

    allClassesTest = test[1]

    data = []
    classes = []

    colXofData = column(test[0], bestSplitColTrain)
    for x in range(0, len(allClassesTest)):
        if (colXofData[x] >= bestSplitValueTrain):
            data.append(colXofData[x])
            classes.append(0)
        elif (colXofData[x] < bestSplitValueTrain):
            data.append(colXofData[x])
            classes.append(1)
    return (classes)

def main():
    """This portion of the program will run when run only when main() is called.
    This is good practice in python, which doesn't have a general entry point 
    unlike C, Java, etc. 
    This way, when you <import HW2>, no code is run - only the functions you
    explicitly call.
    """
    file = load('train.txt')
    fileTest = load('test.txt')
    IG(file, 1, 21)
    G(file, 8, 6)
    CART(file, 5, 8)
    print("The Information Gain's best split is: {}".format(bestSplit(file, "IG")))
    print("The Gini Index's best split is: {}".format(bestSplit(file, "GINI")))
    print("The CART's best split is: {}".format(bestSplit(file, "CART")))

    print("This is the original Test classification: {}".format(fileTest[1]))
    print("classifyIG:\t\t\t\t\t\t\t\t  {}".format(classifyIG(file, fileTest)))
    print("classifyG:\t\t\t\t\t\t\t\t  {}".format(classifyG(file, fileTest)))
    print("classifyCART:\t\t\t\t\t\t\t  {}".format(classifyCART(file, fileTest)))
    print("classifyIG predicted 8/10 classes correctly")
    print("classifyG predicted 3/10 classes correctly")
    print("classifyCART predicted 3/10 classes correctly")
    exit()

if __name__=="__main__":
    """__name__=="__main__" when the python script is run directly, not when it 
    is imported. When this program is run from the command line (or an IDE), the 
    following will happen; if you <import HW2>, nothing happens unless you call
    a function.
    """
    main()
