#%%
from CyrusCNN.CNN import CNN as CNN

#%%
myCNN=CNN(30)

#%%
myCNN.addConvolutionLayer(3,1)
myCNN.addPoolingLayer(3,1)
myCNN.addFlattenLayer()
myCNN.addDenseLayer(10,"relu")
myCNN.addDenseLayer(10,"relu")
# %%
import tensorflow as tf

#%%
print(tf.config.list_physical_devices("GPU"))
print(tf.config.list_physical_devices("CPU"))
#
#%%
n=1
x = tf.random.truncated_normal([n,30,30,3])
print(myCNN.evaluate(x))
# %%
