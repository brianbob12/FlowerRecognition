import tensorflow as tf
import numpy as np

from .Exceptions import *

class ConvolutionLayer:
  
  def __init__(self):
    pass

    #filterSize is interger
    #stride is a single int
  def newLayer(self,kernelSize,stride):
    weightInitSTDDEV=0.1
    self.kernelSize=kernelSize
    self.kernel=tf.Variable(tf.random.truncated_normal(shape=[kernelSize,kernelSize,3,3],stddev=weightInitSTDDEV))
    self.strideSize=stride 
    self.strides=[1,stride,stride,1]
  
  #inputs have shape [None,a,a,3] tf.float32
  def execute(self,inputs):
    return tf.nn.conv2d(inputs,self.kernel,self.strides,"VALID")

  #return a list of the trainable variables
  def getTrainableVariables(self):
    return [self.kernel]

  def exportLayer(self,superdir,subdir):
    import struct
    from os import mkdir
    
    accessPath=superdir+"\\"+subdir

    #first step is to create a directory for the network if one does not already exist
    try:
      mkdir(accessPath)
    except FileExistsError:
      pass
    except Exception as e:
      raise(invalidPath(accessPath))    

    #save hyper.txt
    #contains: fliterSize, strides 
    with open(accessPath+"\\hyper.txt","w") as f:
      f.write(str(self.kernelSize)+"\n")
      f.write(str(self.strideSize)+"\n") 
    
    #save mat.kernel
    kernelFloats=[]
    with open(accessPath+"\\mat.kernel","wb") as f:
      for i in range(self.kernelSize):
        for j in range(self.kernelSize):
          for k in range(3):
            for l in range(3):
              kernelFloats.append(float(self.kernel[i][j][k][l]))
      f.write(bytearray(struct.pack(str(len(kernelFloats))+"f",*kernelFloats)))

  def importLayer(self,superdir,subdir):
    from os import path

    accessPath=superdir+"\\"+subdir
  
    #check if directory exists
    if not path.exists(accessPath):
      raise(missingDirectoryForImport(accessPath))

    #import from hyper.txt
    try:
      with open(accessPath+"\\hyper.txt","r") as f:
        fileLines=f.realdines()
        try:
          self.kernelSize=int(fileLines[0])
        except ValueError as e:
          raise(invalidDataInFile(accessPath+"\\hyper.txt","kernelSize",fileLines[0]))
        try:
          self.strideSize=int(fileLines[1]) 
        except ValueError as e:
          raise(invalidDataInFile(accessPath+"\\hyper.txt","size",fileLines[1])) 
    except IOError:
      raise(missingFileForImport(accessPath+"\\hyper.txt"))

  
    #import kernel
    import struct

    try:
      with open(accessPath+"//mat.kernel","rb") as f:
        raw=f.read()#type of bytes  
        try:
          inp=struct.unpack(str(self.kernelSize*self.kernelSize*3*3)+"f",raw)#list of float32s
        except struct.error as e:
          raise(invalidByteFile(accessPath+"//mat.kernel"))
      
        kernel=[] 
        try:
          for i in range(self.kernelSize):
            kernel.append([])
            for j in range(self.kernel):
              kernel[i].append([])
              for k in range(3):
                kernel[i][j].append([])
                for l in range(3):
                  kernel[i][j][k].append(inp[i*self.kernelSize*3*3+j*3*3+k*3+l])
        except Exception as e:
          raise(invalidByteFile(accessPath+"//mat.kernel"))

        self.kernel=tf.Variable(kernel)

    except IOError:
      raise(missingFileForImport(accessPath,"mat.kernel"))