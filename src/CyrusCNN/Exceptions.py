#
#
#@author cyrus singer
#
#

#
#Netork.Exceptions
#
#This class holds custom Exceptions used for error handeling with the Perceptron object at Network.Perceptron.Perceptron
#

class unspecifiedActivation (Exception):
    pass

class unknownActivationFunction(Exception):
    def __init__(self,badVal):
        self.badValue=badVal

class badPath(Exception):
    def __init__(self,badPath):
        self.badPath=badPath

class missingFile(Exception):
    def __init__(self,path,file):
        self.path=path
        self.fileName=file

class fileMissingData(Exception):
    def __init__(self,file):
        self.filePath=file

class invalidStride(Exception):
    def __init__(self,stride):
        self.stride=stride

class invalidStrideLength(invalidStride):
    pass

class invalidStrideType(invalidStride):
    pass