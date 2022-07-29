#
#
#@author cyrus singer
#


#
#Network.Exceptions
#
#This class holds custom Exceptions used for error handeling 
from typing import Optional

from Elinvar.NN.Nodes import Node


class unspecifiedActivation (Exception):
    pass

class unknownActivationFunction(Exception):
    def __init__(self,badVal):
        self.badValue=badVal

class invalidPath(Exception):
    def __init__(self,badPath):
        self.badPath=badPath

class missingFileForImport(Exception):
    def __init__(self,path:str,fileName:str):
        self.path:str=path
        self.fileName:str=fileName

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

#error for when a node is given the wrong number of inputs
class invalidNodeInputs(Exception):
    def __str__(self):
        return "INVALID NODE INPUTS"

#error for when a node is given the wrong number of outputs
class invalidNumberOfNodeInputs(invalidNodeInputs):
    def __init__(self,numberOfInputsReceived:int,numberOfInputsRequired:int):
        self.numberOfInputsReceived=numberOfInputsReceived
        self.numberOfInputsRequired=numberOfInputsRequired

    def __str__(self):
        out="INVALID NODE INPUTS\tRequired "
        out+=str(self.numberOfInputsRequired)
        out+=" inputs but only has "
        out+=str(self.numberOfInputsReceived)
        return out

#error for when a node is given inputs of the wrong shape
class invalidNodeInputShape(invalidNodeInputs):
    def __init__(self,inputShape,requiredShape):
        self.inputShape=inputShape
        self.requiredShape=requiredShape

    def __str__(self):
        out="INVALID NODE INPUTS\tRequired shape of "
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
    def __init__(self,nodeID:int,operation:str,nodeName:Optional[str]=None):
        self.operation:str=operation
        self.nodeID:int=nodeID
        self.nodeName:Optional[str]=nodeName

    def __str__(self):
        out="OPERATION WITH UNBUILT LAYER\tOperation:"
        out+=self.operation
        out+=f"\nNodeID:{self.nodeID}"
        if self.nodeName!=None:
            out+=f"\nNode name: {self.nodeName}"
        return out

class nodeNotSetup(Exception):
    def __init__(self,attemptedOperation):
        self.attemptedOperation=attemptedOperation
        pass
    def __str__(self):
        out="LAYER NOT SET UP\n"
        out+="Node must be set up before attempting "
        out+=self.attemptedOperation
        return out

class UnregisteredNode(Exception):
    def __init__(self):
        pass
    
    def __str__(self) -> str:
        out="UNREGISTERED NODE\n"
        return out
