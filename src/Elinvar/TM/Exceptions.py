#exceptions to make
#no episode end conditions set
class missingTrainingProtocol(Exception):
  def __init__(self):
    pass
  def __str__(self):
    out="MISSING TRAINING PROTOCOL"
    out+="TrainingProtocol cannot be None. Must be of type TrainingProtocol."
    return out