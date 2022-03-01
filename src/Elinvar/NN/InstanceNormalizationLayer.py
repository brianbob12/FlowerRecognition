
from statistics import mean
from tensorflow import reduce_mean
from .Exceptions import *

#performs instance normilization
#normilizes values within each channel
#must take inputs of shape [batch,chanells,height,width]

class InstanceNormalizationLayer():

  def __init__(self):
    pass

  def newLayer(self,mean,stddev):
    self.stddev=stddev
    self.mean=mean  

  #outputs data with specified stddev and mean
  #TODO this HAS to run on the GPU, far too slow
  def execute(self,inputs):
    means=reduce_mean(inputs,[-2,-3])
    meanOfSquares=reduce_mean(inputs**2,[-2,-3])
    standardDeviations=meanOfSquares-means**2
    out=(inputs-means)/standardDeviations
    return out
  
  def getTrainableVariables(self):
    return([])

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
      f.write(str(self.mean)+"\n")
      f.write(str(self.stddev)+"\n")

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
          self.mean=int(fileLines[0])
        except ValueError as e:
          raise(invalidDataInFile(accessPath+"\\hyper.txt","mean",fileLines[0]))
        try:
          self.stddev=int(fileLines[1])
        except ValueError as e:
          raise(invalidDataInFile(accessPath+"\\hyper.txt","stddev",fileLines[1]))
    except IOError:
      raise(missingFileForImport(accessPath+"\\hyper.txt"))
 