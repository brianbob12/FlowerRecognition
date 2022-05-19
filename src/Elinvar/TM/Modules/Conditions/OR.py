from .Condition import Condition

class OR(Condition):
  def __init__(self,*args:Condition):
    super().__init__()
    self.dependencies=[i for i in args]

    def compute():
      inputs=[i.met for i in self.dependencies]
      self.setValue(any(inputs))

    for condition in self.dependencies:
      condition.onChange.append(lambda value:compute())

    compute()