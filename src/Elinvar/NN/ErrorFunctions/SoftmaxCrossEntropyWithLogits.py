from re import M
from .ErrorFunction import ErrorFunction

#requires Tensorflow
from tensorflow.nn import softmax_cross_entropy_with_logits
from tensorflow import reduce_mean

class SoftmaxCrossEntropyWithLogits(ErrorFunction):
  def __init__(self):
    super().__init__(True)

  @staticmethod
  def execute(guess,y):
    return reduce_mean(softmax_cross_entropy_with_logits(y,guess))