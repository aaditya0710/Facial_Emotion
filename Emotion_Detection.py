#!/usr/bin/env python
# coding: utf-8

# In[4]:


from zipfile import ZipFile
zf = ZipFile('fer2013.csv.zip', 'r')
zf.extractall('JUPYTER_DATA')
zf.close()


# In[26]:


import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt


# In[78]:


data = pd.read_csv("JUPYTER_DATA/fer2013.csv")
data.head()


# In[64]:


del data['Usage']


# In[65]:


x = data['pixels']
y = data['emotion']


# In[67]:


emotion_map = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


# In[79]:


data['pixels'] = data['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])
data_X = np.array(data['pixels'].tolist(), dtype='float32').reshape(-1,48,48,1)/255.0  


# In[84]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(data_X,data_Y,test_size=0.2)


# In[93]:


xtrain.shape


# In[81]:


data_Y = to_categorical(data['emotion'], 7)


# In[31]:


from keras.utils import to_categorical


# In[38]:


from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Flatten,Dropout


# In[110]:


classifier = Sequential()
classifier.add(Conv2D(64,(3,3),input_shape=(48,48,3),activation='relu'))
classifier.add(MaxPooling2D(2,2))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(2,2))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(16,(3,3),activation='relu'))
classifier.add(MaxPooling2D(2,2))
classifier.add(Dropout(0.2))
classifier.add(Flatten())
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(7,activation='softmax'))


# In[88]:


data_X.shape


# In[111]:


classifier.summary()
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[96]:


classifier.fit(xtrain,ytrain,epochs=25)


# In[97]:


classifier.evaluate(xtest,ytest)


# In[170]:


import numpy as np
from keras.preprocessing import image
img = image.load_img("cry.jpg",target_size=(48,48))
img = image.img_to_array(img)
img = np.expand_dims(img,axis=0)
result = classifier.predict(img)


# In[163]:


result = result.tolist()
pred = max(result[0])
ind = result[0].index(pred)
emotion_map[ind]


# In[ ]:




