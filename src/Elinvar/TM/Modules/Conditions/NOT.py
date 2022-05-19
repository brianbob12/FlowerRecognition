from .Condition import Condition

class NOT(Condition):
  def __init__(self,condition:Condition):
    super().__init__()
    self.dependencies=[condition]

    def compute():
      self.setValue(not self.dependencies[0].met)

    condition.onChange.append(lambda value:compute())

    compute()