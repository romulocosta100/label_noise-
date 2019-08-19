#!/usr/bin/env python
# coding: utf-8

# In[153]:


import tensorflow as tf
import random
import numpy as np
import pandas as pd


# In[154]:


mnist = tf.keras.datasets.mnist


# In[155]:


#Load
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))



#Convert the samples from integers to floating-point numbers:
x_train, x_test = x_train / 255.0, x_test / 255.0


# In[156]:


def def_model():
    model = tf.keras.models.Sequential([
        
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        ])

    model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[157]:


def corrupt_a_fraction_of_the_labels(fraction,labels):
    
    new_labels = []
    
    for labels in labels:
        
        if(random.randint(0,100) <= fraction):
            new_labels.append(random.randint(0,9))
        else:
            new_labels.append(labels)
    
    return np.asarray(new_labels)


# In[ ]:





# In[ ]:


df = pd.DataFrame(columns=['fraction','acc_test','acc_train_corrupted','acc_train_uncorrupted'])

for fraction in range(0,110,10):
    
    print("fraction:",fraction)
    y_train_corrupt = corrupt_a_fraction_of_the_labels(fraction,y_train)
    
    model = def_model()
    model.fit(x_train, y_train_corrupt, epochs=200,verbose=1)
    
    
    _, acc_test = model.evaluate(x_test, y_test)
    _, acc_train_corrupted = model.evaluate(x_train, y_train_corrupt)
    _, acc_train_uncorrupted = model.evaluate(x_train, y_train)
    
    print("fraction:",fraction)
    print("acc_test:",acc_test)
    print("acc_train_corrupted:",acc_train_corrupted)
    print("acc_train_uncorrupted:",acc_train_uncorrupted)
    
    df = df.append({'fraction': fraction,'acc_test':acc_test,'acc_train_corrupted':acc_train_corrupted,'acc_train_uncorrupted':acc_train_uncorrupted }, ignore_index=True)


# In[ ]:


df.to_csv("2_Trained_model_after_many_iterations.csv")


# In[ ]:


df.plot.line(x='fraction',figsize=(20,10),grid=True,style=["-o","-s","-^"], ms=10)


# In[ ]:





# In[ ]:




