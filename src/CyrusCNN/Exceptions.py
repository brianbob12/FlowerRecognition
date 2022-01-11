#
#
#@author cyrus singer
#


#
#Network.Exceptions
#
#This class holds custom Exceptions used for error handeling 
class unspecifiedActivation (Exception):
    pass

class unknownActivationFunction(Exception):
    def __init__(self,badVal):
        self.badValue=badVal

class invalidPath(Exception):
    def __init__(self,badPath):
        self.badPath=badPath

class missingFileForImport(Exception):
    def __init__(self,path,fileName):
        self.path=path
        self.fileName=fileName

class missingDirectoryForImport(Exception):
    def __init__(self,path):
        self.path=path

class invalidDataInFile(Exception):
    def __init__(self,pathOfFile,perameter,value):
        self.pathOfFile=pathOfFile
        self.perameter=perameter#string descriptor
        self.value=value

class invalidByteFile(Exception):
    def __init__(self,pathOfFile):
        self.pathOfFile=pathOfFile

class fileMissingData(Exception):
    def __init__(self,file):
        self.filePath=file

class invalidLayerPlacement(Exception):
    def __init__(self,previousLayerFlat,requiresFlat,requiresNonFlat):
        self.previousLayerFlat=previousLayerFlat
        self.requiresFlat=requiresFlat
        self.requiresNonFlat=requiresNonFlat