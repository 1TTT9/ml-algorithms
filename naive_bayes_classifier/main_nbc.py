import string
from numpy import zeros

postStrings  =  [
    "My dog has flea problems, help please.",
    "Maybe not, take him to do park, stupid.",
    "My dalmation is so cute, I love him.",
    "Stop posting stupid worthless garbage.",
    "Mr Licks ate my steak. How to stop him?",
    "Quit buying worthless dog food, stupid."
]


def loadDatasets():
    global postStrings

    ex = string.punctuation
    map_tbl = string.maketrans('','')
    """
      Usage: String.translate(table [,deletechars]) -> string
      Statement:
        Return a copy of the string S, where all characters occurring
        in the optional argument deletechars are removed, and the
        remaining characters have been mapped through the given
        translation table, which must be a string of length 256.
    """
    postLists = []
    for post in postStrings:
	postLists.append( post.lower().translate(map_tbl, ex).split() )

    postClassifier = [ 0,1,0,1,0,1]

    return postLists, postClassifier


def createFeatureSet(dataset):
    vocabset = set([])
    for post in dataset:
	vocabset = vocabset | set(post)
    return list(vocabset)

def setFeatureList( features, post ):
    tmp = [0]* len(features)
    for w in post:
	if w in features:
	    tmp[ features.index(w) ] = 1
    return tmp

def setFeaturesToPL(features, trainedPL):
    retFPL = []
    for post in trainedPL:
	retFPL.append(setFeatureList(features, post))
    return retFPL    


def trainNBC(featuresPL, trainedC):
    """
       NEED TO BE MODIFIED
    """
    print '\n'.join(["    *** [WARNING] ***", 
		     "\tTwo problems hiden behind the classifier:",
		     "\t1) P(w|c{i}) reduced to 0 if P(w{k}|c{i}) is 0",
		     "\t2) similarly, P(w|c{i}) become tiny since all P<1",
		     "    *****************"
		 ])

    numOfTrained = len(trainedC)
    numOfFeatures = len(featuresPL[0])
    p_ab = sum(trainedC)/float(numOfTrained)
    p1 = zeros(numOfFeatures); p0 = zeros(numOfFeatures)
    p1Denom = 0.; p0Denom = 0.
    
    for i in range(numOfTrained):
	if trainedC[i] == 1:
	    p1 += featuresPL[i]
	    p1Denom += sum(featuresPL[i])
	else:
	    p0 += featuresPL[i]
	    p0Denom += sum(featuresPL[i])

    v_p1 = p1/p1Denom
    v_p0 = p0/p0Denom
    return v_p1, v_p0, p_ab

def main():
    trainedPL, trainedC = loadDatasets()
    
    features = createFeatureSet(trainedPL)
    #print features

    featuresPL = setFeaturesToPL(features, trainedPL)
    #for f in featuresPL: print f

    """
      NBC: P(c{i}|w) = ( P(w|c{i}) * P(c{i}) )/P(w)
      where P(c{i}) means the probability the class{i} happened in all posts,
            P(w|c{i}) = P(w{1}|c{i})*P(w{2}|c{i})*...*P(w{n}|c{i})
            P(w) is neglictable here in convenience of calculation
    """
    v_p1, v_p0, p_ab = trainNBC(featuresPL, trainedC)
    print p_ab


if __name__ == '__main__':
    main()

