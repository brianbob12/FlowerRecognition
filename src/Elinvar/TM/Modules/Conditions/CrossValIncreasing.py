from audioop import cross
from typing import Optional

from requests import NullHandler

from Elinvar.TM import TrainingEpisode

from .Condition import Condition

class CrossValIncreasing(Condition):
  __slots__=("target","lastCrossVal","counter")
  def __init__(self,target:int=1):
    self.target:int=target
    self.lastCrossVal:Optional[float]=None
    self.counter:int=0
  
  def startOfEpisode(self, trainingEpisode: TrainingEpisode, episodeIndex: int):
    self.setValue(False)
    self.lastCrossVal=None
    self.counter=0
  
  def endOfCrossVal(self, trainingEpisode: TrainingEpisode, index: int, crossValError: float):
    if self.lastCrossVal==None:
      pass
    else:
      if(self.lastCrossVal<crossValError):
        self.counter+=1
        if self.counter==self.target:
          self.setValue(True)
      else:
        self.counter=0
        self.setValue(False)
    self.lastCrossVal=crossValError