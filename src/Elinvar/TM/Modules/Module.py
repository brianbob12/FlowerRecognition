
from Elinvar.TM import TrainingEpisode

class Module():

  def __init__(self):
    pass

  def startOfQue(self,saveDirectory:str):
    pass

  def startOfEpisode(self,trainingEpisode:TrainingEpisode,episodeIndex:int):
    pass

  def endOfIteration(self,trainingEpisode:TrainingEpisode,index:int,trainingError:float,iterationTime:float):
    pass

  def endOfCrossVal(self,trainingEpisode:TrainingEpisode,index:int,crossValError:float):
    pass

  def endOfEpisode(self,trainingEpisode:TrainingEpisode,lastCrossValError:float):
    pass

  def endOfQue(self):
    pass
    

