#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv('titanic_train.csv')


# In[5]:


df.head()


# In[6]:


print(df.shape)        # Rows and columns
print(df.columns)      # Column names
print(df.info())       # Data types and nulls
print(df.describe())   # Summary stats


# In[7]:


# Check for missing data
print(df.isnull().sum())

# Visualize missing values
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()


# In[9]:


# Categorical: Countplot
sns.countplot(x='Sex', data=df)
plt.title("Passenger Gender Distribution")
plt.show()


# In[10]:


# Numerical: Histogram
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.show()


# In[11]:


# Survival by gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Count by Gender")
plt.show()


# In[12]:


# Age vs Survived
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.show()


# In[13]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[14]:


#Most women survived, most men didn’t.

#Younger passengers had slightly better survival chances.

#Higher class passengers had higher survival rates.


# In[16]:


df_cleaned = df.dropna(subset=['Age', 'Embarked'])
df_cleaned.to_csv("titanic_cleaned.csv", index=False)

