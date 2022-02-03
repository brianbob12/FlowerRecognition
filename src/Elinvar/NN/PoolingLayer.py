from importlib import invalidate_caches
from os import access
import tensorflow as tf
import numpy as np
from .Exceptions import invalidDataInFile, invalidPath, missingDirectoryForImport, missingFileForImport

#non-trainable layer#
#maxpooling
class PoolingLayer:
  def __init__(self):
    pass
  def newLayer(self,size,stride):
    self.size=size
    self.stride=stride
    
  #input has shape[None,a,a,3]
  def execute(self,inputs):
    #out=[]
    #for i in inputs:
    #  out.append(tf.nn.max_pool(inputs,self.size,self.stride,"VALID"))
    #return out
    return tf.nn.max_pool2d(inputs,[self.size,self.size],self.stride,"VALID")

  #has no trainable variables
  #this function is here so that all layers can hava a .getTrainableVariables
  def getTrainableVariables(self):
    return []

  def exportLayer(self,superdir,subdir):
    from os import mkdir

    accessPath=superdir+"\\"+subdir

    #create directory
    try:
      mkdir(accessPath)
    except FileExistsError:
      pass
    except Exception as e:
      raise(invalidPath(accessPath))

    #save hyper.txt
    #has data:size,stride
    with open(accessPath+"\\hyper.txt","w") as f:
      f.write(str(self.size)+"\n")
      f.write(str(self.stride)+"\n")

  def importLayer(self,superdir,subdir):
    from os import path

    accessPath=superdir+"\\"+subdir
    #check if directory exists
    if not path.exists(accessPath):
      raise(missingDirectoryForImport(accessPath))

    #import from hyper.txt
    try:
      with open(accessPath+"\\hyper.txt","r") as f:
        fileLines=f.readlines()
        #strip line breaks
        fileLines=[i[:-1] for i in fileLines]
        try:
          self.size=int(fileLines[0])
        except ValueError as e:
          raise(invalidDataInFile(accessPath+"\\hyper.txt","size",fileLines[0]))
        try:
          self.stride=int(fileLines[1])
        except ValueError as e:
          raise(invalidDataInFile(accessPath+"\\hyper.txt","stride",fileLines[1]))
    except IOError:
      raise(missingFileForImport(accessPath+"\\hyper.txt"))