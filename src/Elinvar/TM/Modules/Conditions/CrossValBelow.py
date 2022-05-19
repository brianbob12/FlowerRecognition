from Elinvar.TM import TrainingEpisode
from .Condition import Condition

class CrossValBelow(Condition):
  __slots__=("target")
  def __init__(self,target:float):
      super().__init__()
      self.target:float=target
  
  def startOfEpisode(self, trainingEpisode: TrainingEpisode, episodeIndex: int):
    self.setValue(False)
      
  def endOfCrossVal(self, trainingEpisode: TrainingEpisode, index: int, crossValError: float):
    if self.met:
      if crossValError>=self.target:
        self.setValue(False)
    else:
      if crossValError<self.target:
        self.setValue(True)