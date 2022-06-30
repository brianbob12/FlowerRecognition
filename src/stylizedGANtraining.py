import Elinvar
import tensorflow as tf
import numpy as np
from PIL import Image

from dataRetrieval import getBatch, getFiles
from NetworkDesigns import newDiscriminator, newGenerator

#series A trains the discriminator, series B trains the generator

generatorNetwork=newGenerator()
discriminatorNetwork=newDiscriminator()