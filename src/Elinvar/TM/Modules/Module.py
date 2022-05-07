from Elinvar.TM.TrainingEpisode import TrainingEpisode

class Module:
  def __init__(self):
    pass

  def startOfQue(self,saveDirectory:str):
    pass

  def startOfEpisode(self,trainingEpisode:TrainingEpisode,episodeIndex:int):
    pass

  def endOfIteration(self,trainingEpisode:TrainingEpisode,trainingError:float,iterationTime:foat):
    pass

  def endOfCrossVal(self,trainingEpisode:TrainingEpisode,crossValError:float):
    pass

  def endOfEpisode(self,trainingEpisode:TrainingEpisode):
    pass

  def endOfQue(self):
    pass
    

