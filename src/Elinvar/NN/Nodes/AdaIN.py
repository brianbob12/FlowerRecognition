#Adaptive Instance Normalization Layer
from .InstanceNormalizationNode import InstanceNormalizationNode
#TODO (obviously)

class AdaIN(InstanceNormalizationNode):
  def execute(self, inputs):
      return super().execute(inputs)