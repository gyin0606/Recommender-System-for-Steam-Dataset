#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json

path = open("ratings.csv","w+")
path.write("UserID,itemID,Rating\n")
k = 0
with open("australian_users_items.json") as f:
    for i in f:
        k += 1
        i = eval(i.strip())
        for j in i['items']:
            if j["playtime_forever"] >0:
                path.write(str(i["user_id"])+","+str(j["item_id"])+","+str(j["playtime_forever"])+"\n")
        if k>1000:
            break
path.close()


# In[4]:


# 每个 游戏均分等级
import pandas as pd
data = pd.read_csv("ratings.csv")
data.head()


# In[6]:


len(data),len(set(data.itemID))


# In[15]:


temp = data.groupby("itemID")["Rating"].agg(max)

data = pd.merge(data,pd.DataFrame(temp).reset_index(level=[0]),on="itemID")
data.head()


# In[19]:


data["Rating"] = pd.qcut(data.Rating_x/data.Rating_y,q=5)
data.head()


# In[22]:


lable2id = {}
for i in sorted(set(data.Rating)):
    lable2id[i] = len(lable2id)+1
lable2id


# In[23]:


data["Rating"] = data["Rating"].apply(lambda x:lable2id[x])


# In[24]:


data[["UserID","itemID","Rating"]].to_csv("data/ratings1.csv",index=False)


# In[ ]:




