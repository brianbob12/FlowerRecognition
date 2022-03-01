from .ErrorFunction import ErrorFunction
from tensorflow import reduce_mean

#currently no support for single label
class MSE(ErrorFunction):
  def __init__(self,multipleLabels):
    super().__init__(multipleLabels)

  #differentiable function that returns error
  #must pass numpy arrays
  def execute(self,guess,y):
    return reduce_mean((guess-y)**2)