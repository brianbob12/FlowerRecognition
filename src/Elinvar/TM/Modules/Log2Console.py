from Elinvar.TM import TrainingEpisode
from Elinvar.TM.Modules.Module import Module

class Log2Console(Module):
  def __init__(self):
    self.tableHeadings:list[str]=["Iteration","Training Error","Cross Val Error","Iteration Time"]
    self.rowFormat="{:>20}"*(len(self.tableHeadings))
    self.lastTrainingErrorStr:str=""
    self.lastIterationTimeStr:str=""
    self.crossValidated=False

  def startOfQue(self,saveDirectory:str):
    print("Starting Que")

  def startOfEpisode(self,trainingEpisode:TrainingEpisode,episodeIndex:int):
    print(f"Starting TrainingEpisode #{episodeIndex} {trainingEpisode.name}")
    print("-"*20*len(self.tableHeadings))
    print(self.rowFormat.format(*self.tableHeadings))

  def endOfIteration(self,trainingEpisode:TrainingEpisode,index:int,trainingError:float,iterationTime:float):
    if not self.crossValidated:
      #go to new line
      print()

    self.crossValidated=False
    te=format(trainingError,".4f")
    it=format(iterationTime,".4f")
    self.lastTrainingErrorStr=te
    self.lastIterationTimeStr=it
    print(self.rowFormat.format(index,te,"",it),end="")

  #this always runs after endOfIteration but does not run every time after
  def endOfCrossVal(self,trainingEpisode:TrainingEpisode,index:int,crossValError:float):
    cv=format(crossValError,".4f")
    self.crossValidated=True
    print("\r"+self.rowFormat.format(index,self.lastTrainingErrorStr,cv,self.lastIterationTimeStr))

  def endOfEpisode(self,trainingEpisode:TrainingEpisode,lastCrossValError:float):
    print("-"*20*len(self.tableHeadings))

  def endOfQue(self):
    print("End of que")
    