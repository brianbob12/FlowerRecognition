#Adaptive Instance Normalization Layer
from .InstanceNormalizationNode import InstanceNormalizationNode

class AdaINStyleTransfer(InstanceNormalizationNode):
  def execute(self, inputs):
      return super().execute(inputs)