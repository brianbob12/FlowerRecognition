from Elinvar.TM import TrainingEpisode
from .Module import Module


class Export(Module):
  def __init__(self,exportAll:bool=None,minCrossVal:float=None):
    self.exportAll:bool=exportAll
    self.minCrossVal:float=minCrossVal
    self.saveDirecotry:str=None

  def startOfQue(self, saveDirectory: str):
    self.saveDirecotry=saveDirectory

  def endOfEpisode(self, trainingEpisode: TrainingEpisode,lastCrossValError:float):
    def save():
      print(f"Saving network from episode {trainingEpisode.name} to {self.saveDirecotry}...")
      trainingEpisode.exportNetwork(self.saveDirecotry)
      print("Done saveing")

    if self.exportAll:
      save()
    else:
      if self.minCrossVal !=None:
        if self.minCrossVal<self.lastCrossVal:
          save()

    self.lastCrossVal=None