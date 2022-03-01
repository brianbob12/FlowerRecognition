
#a parent class holding error functions

class ErrorFunction:
  #multiple labels is a boolean
  #true for error functions that run for many labels each iteration
  #false for error functions that only run on one label per iteration
  def __init__(self,multipleLabels):
    self.multipleLabels=multipleLabels

  #there will be an execute function for all ErrorFunctions