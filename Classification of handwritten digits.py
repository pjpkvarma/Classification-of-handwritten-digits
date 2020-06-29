#!/usr/bin/env python
# coding: utf-8

# # Simple network for classification of handwritten digits

# i'm using mnist dataset

# In[1]:


#import keras
import keras


# In[2]:


#import mnist dataset
from keras.datasets import mnist


# In[3]:


#load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[4]:


#the images are encoded as Numpy arrays and labels are an array of digits
#train images shape
train_images.shape


# In[5]:


test_images.shape


# In[6]:


from keras import models, layers


# In[7]:


net = models.Sequential()
net.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
net.add(layers.Dense(10, activation='softmax'))


# In[9]:


net.summary()


# In[10]:


net.compile(optimizer='rmsprop',
           loss='categorical_crossentropy',
           metrics=['accuracy'])


# In[11]:


#preparing the image data
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255


# In[12]:


from keras.utils import to_categorical


# In[13]:


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[14]:


net.fit(train_images, train_labels, epochs=5, batch_size=128)


# In[15]:


test_loss, test_accuracy = net.evaluate(test_images, test_labels)
test_accuracy


# In[ ]:




