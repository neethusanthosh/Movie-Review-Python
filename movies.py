#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# #Reading the data

# In[7]:


movies=pd.read_csv("D:/decoder lectures/casestudy/MovieData.csv")

movies.head()


# #inspect dataframe

# In[8]:


movies.shape


# In[9]:


movies.info()


# In[10]:


movies.describe()


# #2.Data Analysis

# 2.1  Reducing digits

# In[11]:


movies["Gross"]=movies["Gross"]/1000000
movies["budget"]=movies["budget"]/1000000
movies


# #2.2 Profit

# In[12]:


movies["profit"]=movies["Gross"]- movies["budget"]
movies


# In[13]:


# Sort the dataframe with the 'profit' column as reference
movies=movies.sort_values(by="profit",ascending=False)


# In[14]:


# Top 10 profitable movies by using position based indexing. Specify the rows till 10 (0-9)

movies.iloc[:10,:]


# In[15]:


# profit vs budget

sns.jointplot("budget","profit",movies)
plt.show()#Plot profit vs budget


# In[16]:


movies.columns


# In[17]:


#movies with negative profit

movies[movies["profit"]<0]


# #Rating and critics

# In[18]:


# Change the scale of MetaCritic

movies["MetaCritic"]=movies["MetaCritic"]/10


# In[19]:


# Find the average ratings

movies["Avg_rating"]=(movies["IMDb_rating"]+movies["MetaCritic"])/2


# In[20]:


movies


# In[21]:


#Sort in descending order of average rating
df=movies[["Title","IMDb_rating","MetaCritic","Avg_rating"]]
df=df.loc[abs(df["IMDb_rating"]-df["MetaCritic"]<0.5)]


# In[28]:


# Find the movies with metacritic-IMDb_rating < 0.5 and also with the average rating of >8
df=df.sort_values(by="Avg_rating",ascending=False)
UniversalAcclaim=df.loc[df["Avg_rating"]>=8]
UniversalAcclaim


# In[ ]:


#most popular trios


# In[29]:


group=movies.pivot_table(values=["actor_1_facebook_likes","actor_2_facebook_likes","actor_3_facebook_likes"],
                  aggfunc="sum",index=["actor_1_name","actor_2_name","actor_3_name"])
group


# In[30]:


group["Total likes"]=group["actor_1_facebook_likes"]+group["actor_2_facebook_likes"]+group["actor_3_facebook_likes"]
group


# In[31]:


group.sort_values(by="Total likes",ascending=False,inplace=True)
group


# In[32]:


group.reset_index(inplace=True)
group


# In[33]:


group=group.iloc[0:5,:]
group


# In[34]:


# most populat trio2


# In[35]:


sorted([1,5,2])


# In[36]:


j=0
for i in group["Total likes"]:
    temp=sorted([group.loc[j,"actor_1_facebook_likes"],group.loc[j,"actor_2_facebook_likes"],group.loc[j,"actor_3_facebook_likes"]])
    if temp[0]>= temp[1]/2 and temp[0]>=temp[2]/2 and temp[1]>=temp[2]/2:
        print(sorted([group.loc[j,"actor_1_name"],group.loc[j,"actor_2_name"],group.loc[j,"actor_3_name"]]))

    j=j+1


# In[37]:


#2.6 Runtime analysis


# In[38]:


# Runtime histogram/density plot
plt.hist(movies["Runtime"])
plt.show()


# In[39]:


#2.7 R rated movies


# In[40]:


# Write your code here

movies.loc[movies["content_rating"]=="R"].sort_values(by="CVotesU18",ascending=False)[["Title","CVotesU18"]].head(10)


# In[41]:


#3.Demographic analysis


# In[42]:


#3.1 combine  dataframe by genre


# In[43]:


# Create the dataframe df_by_genre
df_by_genre=movies.loc[:,"CVotes10":"VotesnUS"]
df_by_genre[["genre_1","genre_2","genre_3"]]=movies[["genre_1","genre_2","genre_3"]]

df_by_genre


# In[44]:


# Create a column cnt and initialize it to 1
df_by_genre["cnt"]=1


# In[45]:


df_by_genre[["genre_1","genre_2","genre_3"]]


# In[46]:


# Group the movies by individual genres
df_by_g1=df_by_genre.groupby("genre_1").aggregate(np.sum)
df_by_g2=df_by_genre.groupby("genre_2").aggregate(np.sum)
df_by_g3=df_by_genre.groupby("genre_3").aggregate(np.sum)

df_by_g1


# In[47]:


df_by_g2


# In[48]:


# Add the grouped data frames and store it in a new data frame
df_add=df_by_g1.add(df_by_g2,fill_value=0)
df_add=df_add.add(df_by_g3,fill_value=0)
df_add


# In[49]:


# Extract genres with atleast 10 occurences
genre_top_10=df_add.loc[df_add["cnt"]>10]

genre_top_10


# In[50]:


# Take the mean for every column by dividing with cnt 
genre_top_10.iloc[:,0:-1]=genre_top_10.iloc[:,0:-1].divide(genre_top_10["cnt"],axis=0)
genre_top_10


# In[51]:


# Rounding off the columns of Votes to two decimals
genre_top_10.loc[:,"VotesM":"VotesnUS"]=round(genre_top_10.loc[:,"VotesM":"VotesnUS"],2)


# In[52]:


# Converting CVotes to int type
genre_top_10[genre_top_10.loc[:,"CVotes10":"CVotesnUS"].columns]=genre_top_10[genre_top_10.loc[:,"CVotes10":"CVotesnUS"].columns].astype(int)


# In[53]:


genre_top_10


# In[54]:


#3.2 genre counts


# In[55]:


# Countplot for genres
sns.barplot(x=genre_top_10["cnt"],y=genre_top_10.index)
plt.show()


# In[56]:


#gender and genre


# In[57]:


# 1st set of heat maps for CVotes-related columns
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
ax=sns.heatmap(genre_top_10[["CVotesU18M","CVotes1829M","CVotes3044M","CVotes45AM"]],annot=True,cmap="coolwarm")
plt.subplot(1,2,2)

ax=sns.heatmap(genre_top_10[["CVotesU18F","CVotes1829F","CVotes3044F","CVotes45AF"]],annot=True,cmap="coolwarm")
plt.show()


# In[58]:


## 2st set of heat maps for CVotes-related columns
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
ax=sns.heatmap(genre_top_10[["VotesU18M","Votes1829M","Votes3044M","Votes45AM"]],annot=True,cmap="coolwarm")
plt.subplot(1,2,2)

ax=sns.heatmap(genre_top_10[["VotesU18F","Votes1829F","Votes3044F","CVotes45AF"]],annot=True,cmap="coolwarm")
plt.show()


# In[61]:


movies["Country"].value_counts()


# In[63]:


movies["IFUS"]=movies["Country"].copy()
movies.loc[movies["IFUS"]!="USA","IFUS"]="non-USA"


# In[64]:


movies["IFUS"].value_counts()


# In[65]:


#Box plot


# In[66]:


plt.subplot(1,2,1)
sns.boxplot(x="IFUS",y="CVotesUS",data=movies)
plt.subplot(1,2,2)
sns.boxplot(x="IFUS",y="CVotesnUS",data=movies)
plt.subplot(1,2,2)


# In[67]:


plt.subplot(1,2,1)
sns.boxplot(x="IFUS",y="VotesUS",data=movies)
plt.subplot(1,2,2)
sns.boxplot(x="IFUS",y="VotesnUS",data=movies)
plt.subplot(1,2,2)


# In[68]:


# top1000voters and genre


# In[69]:


genre_top_10=genre_top_10.sort_values('CVotes1000',ascending=False)


# In[70]:


genre_top_10['CVotes1000']


# In[71]:


#Bar Plot


# In[74]:


sns.barplot(genre_top_10.index,genre_top_10['CVotes1000'])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




