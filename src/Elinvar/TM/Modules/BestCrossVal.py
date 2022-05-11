from Elinvar.TM import TrainingEpisode
from .Module import Module

class BestCrossVal(Module):
  def __init__(self):
    self.bestCrossVal=None
    self.bestIndex=None
    self.bestTrainingEpisode=None
    self.currentIndex=None

  def startOfEpisode(self, trainingEpisode: TrainingEpisode, episodeIndex: int):
    self.currentIndex=episodeIndex

  def endOfEpisode(self, trainingEpisode: TrainingEpisode, lastCrossValError:float):
    if self.bestTrainingEpisode==trainingEpisode:
      return

    if self.bestCrossVal==None:
      self.bestCrossVal=lastCrossValError
      self.bestTrainingEpisode=trainingEpisode
      self.bestIndex=self.currentIndex
    elif lastCrossValError<self.bestCrossVal:
      self.bestCrossVal=lastCrossValError
      self.bestTrainingEpisode=trainingEpisode
      self.bestIndex=self.currentIndex