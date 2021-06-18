#%%
from CyrusCNN.CNN import CNN as CNN

#%%
myCNN=CNN(256,True)

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
#%%

#%%
n=3
x = tf.random.truncated_normal([n,256,256,3])
print(myCNN.evaluate(x))
# %%
