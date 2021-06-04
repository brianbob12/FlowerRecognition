#
#
#@author cyrus singer
#


#
#Netork.Exceptions
#
#This class holds custom Exceptions used for error handeling 
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

class invalidLayerPlacement(Exception):
    def __init__(self,previousLayerFlat,requiresFlat,requiresNonFlat):
        self.previousLayerFlat=previousLayerFlat
        self.requiresFlat=requiresFlat
        self.requiresNonFlat=requiresNonFlat