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
        
class invalidNodeConnection(Exception):
    def __init__(self,inputShape,requiredShape):
        self.inputShape=inputShape
        self.requiredShape=requiredShape

    def __str__(self):
        out="INVALID NODE CONNECTION\tRequired shape of "
        out+=str(self.requiredShape)
        out+=" but received shape of "
        out+=str(self.inputShape)
        return out

class notEnoughNodeConnections(Exception):
    def __init__(self,numberOfConnectionsReceived,requiredConnections):
        self.numberOfConnectionsReceived=numberOfConnectionsReceived
        self.requiredConnections=requiredConnections

    def __str__(self):
        out="NOT ENOUGH CONNECTIONS\tRequired "
        out+=str(self.requiredConnections)
        out+=" connections but only has "
        out+=str(self.numberOfConnectionsReceived)
        return out

class operationWithUnbuiltNode(Exception):
    def __init__(self,operation):
        self.operation=operation

    def __str__(self):
        out="OPERATION WITH UNBUILT LAYER\tOperation:"
        out+=self.operation
        return out
