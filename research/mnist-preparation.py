import numpy as np


def loadMNIST( folder ):
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile( folder + '/' + 'train-images-idx3-ubyte', 
                            dtype = 'ubyte' )

    magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )
    data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )

    labels = np.fromfile( folder + '/' + 'train-labels-idx1-ubyte',
                            dtype = 'ubyte' )[2 * intType.itemsize:]

    return data, labels

trainingImages, trainingLabels = loadMNIST( "../train_data" )
testImages, testLabels = loadMNIST( "../train_data" )
# test
print(trainingImages)
print(trainingLabels)

def toHotEncoding( classification ):
    # emulates the functionality of tf.keras.utils.to_categorical( y )
    hotEncoding = np.zeros( [ len( classification ),
                              np.max( classification ) + 1 ] )
    hotEncoding[ np.arange( len( hotEncoding ) ), classification ] = 1
    return hotEncoding

trainingLabels = toHotEncoding( trainingLabels )
testLabels = toHotEncoding( testLabels )
# test
print (trainingLabels)
print(testLabels)
