#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import math
import scipy as scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


bene_data = pd.read_csv("C:/Users/sravy/Downloads/archive/Train_Beneficiarydata-1542865627584.csv")
ip_data = pd.read_csv("C:/Users/sravy/Downloads/archive/Train_Inpatientdata-1542865627584.csv")
op_data = pd.read_csv("C:/Users/sravy/Downloads/archive/Train_Outpatientdata-1542865627584.csv")


# In[3]:


bene_data.shape


# In[4]:


ip_data.shape


# In[5]:


op_data.shape


# In[6]:


# BENEFICIARY DATA EDA


# In[7]:


bene_data.columns


# In[8]:


bene_data.dtypes


# In[9]:


bene_data.head()


# In[10]:


#How many unique beneficiaries we have in our dataset

bene_data['BeneID'].nunique()


# In[11]:


#There are no duplicate beneficiary IDs


# In[12]:


#how many records we have at the Geneder level?

bene_data['Gender'].unique()


# In[13]:


bene_data['Gender'] = bene_data['Gender'].apply(lambda val: 0 if val == 2 else 1)


# In[14]:


with plt.style.context('seaborn'):
  plt.figure(figsize=(10,8))
  fig = bene_data['Gender'].value_counts().plot(kind='bar', color=['orange','blue'])
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{str(round((height*100)/bene_data.shape[0],2))+"%"}', (x + width/2, y + height*1.015), ha='center', fontsize=13.5)
  # Providing the labels and title to the graph
  plt.xlabel("Gender Code")
  plt.ylabel("Number or % share of patients\n")
  plt.grid(which='major', linestyle="--", color='lightgrey')
  plt.minorticks_on()
  plt.title("Distribution of BENEFICIARIES based on GENDER\n")


# In[15]:


#From the above plot we can observe that patients with Geneder 0: 57% and Gender 1: 42.91%


# In[16]:


# Calculate the age of beneficiary

bene_data['DOB'] = pd.to_datetime(bene_data['DOB'], format="%Y-%m-%d")


# In[17]:


bene_data['Patient_Age_Year'] = bene_data['DOB'].dt.year


# In[18]:


bene_age_year_df = pd.DataFrame(bene_data['Patient_Age_Year'].value_counts()).reset_index(drop=False)
bene_age_year_df.columns= ['year','num_of_beneficiaries']
bene_age_year_df = bene_age_year_df.sort_values(by='year')


# In[19]:


# Here, I'm displaying the distribution of BENEFICIARIES on the basis of their YEAR of Birth 
with plt.style.context('seaborn'):
  plt.figure(figsize=(21,9))
  fig = sns.barplot(data=bene_age_year_df, x='year', y='num_of_beneficiaries', palette='inferno')
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{str(round((height*100)/bene_data.shape[0],1))+"%"}', (x + width/2, y + height*1.025), ha='center', fontsize=13.5, rotation=90)
  # Providing the labels and title to the graph
  plt.xlabel("\nBeneficary YEAR of Birth")
  plt.xticks(rotation=90)
  plt.ylabel("Number or % share of patients\n")
  plt.minorticks_on()
  plt.grid(which='major', linestyle="--", color='lightgrey')
  plt.title("Distribution of BENEFICIARIES based on their YEAR of birth\n")


# In[20]:


# From the above plot we can observe that majority of beneficiares are from year 1919 to 1943. 
# More specifically percentage of beneficiaries are high along the years 1939-1943 where the lowest are along 1978-1983


# In[21]:


#Lets check if the beneficiary is dead or alive?

bene_data['Dead_or_Alive'] = bene_data['DOD'].apply(lambda val: 0 if val != val else 1)


# In[22]:


bene_data['Dead_or_Alive'].value_counts()


# In[23]:


# Here, I'm displaying the distribution of whether BENEFICIARY is ALIVE or NOT?
with plt.style.context('seaborn'):
  plt.figure(figsize=(10,8))
  fig = bene_data['Dead_or_Alive'].value_counts().plot(kind='bar', color=['lightblue','coral'])
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{str(round((height*100)/bene_data.shape[0],2))+"%"}', (x + width/2, y + height*1.015), ha='center', fontsize=13.5)
  # Providing the labels and title to the graph
  plt.xlabel("Alive or Dead Status?")
  plt.xticks(labels=["ALIVE","DEAD"], ticks=[0,1], rotation=20)
  plt.ylabel("Number or % share of patients\n")
  plt.grid(which='major', linestyle="--", color='lightgrey')
  plt.minorticks_on()
  plt.title("Distribution of BENEFICIARIES based on Alive or Dead Status\n")


# In[24]:


#From the above plot we can observe that 99% of beneficiaries are alive


# In[25]:


bene_data['DOD'] = pd.to_datetime(bene_data['DOD'])


# In[26]:


# Greatest Date of Death in the TRAIN set for beneficiaries
max_bene_DOD = max(bene_data['DOD'].unique()[1:])
max_bene_DOD


# In[27]:


bene_data['DOD'].fillna(value=max_bene_DOD, inplace=True)


# In[28]:


bene_data['AGE'] = np.round(((bene_data['DOD'] - bene_data['DOB']).dt.days)/365.0,1)


# In[29]:


bene_data.drop(labels=['DOD'],axis=1,inplace=True)


# In[30]:


bene_data.head()


# In[31]:


# Here, I'm displaying the distribution of AGE of Beneficiaries?
with plt.style.context('seaborn'):
  plt.figure(figsize=(10,8))
  bene_data['AGE'].plot(kind='hist', color='purple')
  # Providing the labels and title to the graph
  plt.xlabel("\nBeneficiaries Age in years")
  plt.ylabel("Frequency of patients\n")
  plt.grid(which='major', linestyle="--", color='lightgrey')
  plt.minorticks_on()
  plt.title("Distribution of BENEFICIARIES AGE")
  plt.legend();


# In[32]:


#From the above plot we can observe that majority of beneficiaries are b/w age 65 to 85.


# In[33]:


bene_data['AGE'].describe()


# In[34]:


#Lets see the number of beneficiaries on the basis of 'ChronicCond_Alzheimer'. And, the Annual IP & OP expenditures for such patients.


# In[35]:


with plt.style.context('seaborn-poster'):
  plt.figure(figsize=(12,8))
  fig = bene_data['ChronicCond_Alzheimer'].value_counts().plot(kind='bar', color=['palegreen','orange'])
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{str(round((height*100)/bene_data.shape[0],2))+"%"}', (x + width/2, y + height*1.01), ha='center', fontsize=13.5, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic ALZH Disease?")
  plt.xticks(ticks=[0,1], labels=['NO', 'YES'], fontsize=13, rotation=30)
  plt.ylabel("Number or % share of patients\n")
  plt.grid(which='major', linestyle="--", color='lightgrey')
  plt.minorticks_on()
  plt.title("Distribution of BENEFICIARIES on the basis of 'ChronicCond_Alzheimer'\n")

# 1 means +ve with Chronic ALZH Disease
# 2 means -ve with Chronic ALZH Disease
print(pd.DataFrame(bene_data['ChronicCond_Alzheimer'].value_counts()),"\n")


# In[36]:


#From the above plot we can observe that Beneficiaries that have Chronic Alzhimers are 33.22%.
#Beneficiaries with chronic alzhimers are almost half of with no Alzhimers


# In[37]:


# Here, I'm displaying the Total Annual Sum of Max IP Reimbursement for 'ChronicCond_Alzheimer'
with plt.style.context('seaborn-poster'):
  plt.figure(figsize=(12,8))
  fig = bene_data.groupby(['ChronicCond_Alzheimer'])['IPAnnualReimbursementAmt'].sum().plot(kind='bar', color=['orange','palegreen'])
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{str(round((height*100)/(bene_data["IPAnnualReimbursementAmt"].sum()),2))+"%"}', (x + width/2, y + height*1.01), ha='center', fontsize=13.5, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic ALZH Disease?")
  plt.xticks(ticks=[0,1], labels=['YES', 'NO'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum of Max IP Reimbursement \n")
  plt.grid(which='major', linestyle="--", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of Max IP Reimbursement : 'ChronicCond_Alzheimer'\n")

# 1 means +ve with Chronic ALZH Disease
# 2 means -ve with Chronic ALZH Disease
print(pd.DataFrame(bene_data.groupby(['ChronicCond_Alzheimer'])['IPAnnualReimbursementAmt'].sum()),"\n")


# In[38]:


#Reimbursement of admitted patients with chronic alzhimers is less than other IP


# In[39]:


# Here, I'm displaying the Total Annual Sum of Max OP Reimbursement for 'ChronicCond_Alzheimer'
with plt.style.context('seaborn-poster'):
  plt.figure(figsize=(12,8))
  fig = bene_data.groupby(['ChronicCond_Alzheimer'])['OPAnnualReimbursementAmt'].sum().plot(kind='bar', color=['orange','palegreen'])
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{str(round((height*100)/(bene_data["OPAnnualReimbursementAmt"].sum()),2))+"%"}', (x + width/2, y + height*1.01), ha='center', fontsize=13.5, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic ALZH Disease?")
  plt.xticks(ticks=[0,1], labels=['YES', 'NO'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum of Max OP Reimbursement \n")
  plt.grid(which='major', linestyle="--", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of Max OP Reimbursement : 'ChronicCond_Alzheimer'\n")

# 1 means +ve with Chronic ALZH Disease
# 2 means -ve with Chronic ALZH Disease
print(pd.DataFrame(bene_data.groupby(['ChronicCond_Alzheimer'])['OPAnnualReimbursementAmt'].sum()),"\n")


# In[40]:


#Reimbursement of patients with alzhimers who are not admitted is 16% less than the other group


# In[41]:


# Here, I'm displaying the Total Annual Sum of IP Co-payment for 'ChronicCond_Alzheimer'
with plt.style.context('seaborn-poster'):
  plt.figure(figsize=(12,8))
  fig = bene_data.groupby(['ChronicCond_Alzheimer'])['IPAnnualDeductibleAmt'].sum().plot(kind='bar', color=['orange','palegreen'])
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{str(round((height*100)/(bene_data["IPAnnualDeductibleAmt"].sum()),2))+"%"}', (x + width/2, y + height*1.01), ha='center', fontsize=13.5, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic ALZH Disease?")
  plt.xticks(ticks=[0,1], labels=['YES', 'NO'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum of IP Co-payment \n")
  plt.grid(which='major', linestyle="--", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of IP Co-payment paid by patient : 'ChronicCond_Alzheimer'\n")

# 1 means +ve with Chronic ALZH Disease
# 2 means -ve with Chronic ALZH Disease
print(pd.DataFrame(bene_data.groupby(['ChronicCond_Alzheimer'])['IPAnnualDeductibleAmt'].sum()),"\n")


# In[42]:


# Amount paid by the paitent who have been admitted with alzhimers is high when compared to the other group


# In[43]:


with plt.style.context('seaborn-poster'):
  plt.figure(figsize=(12,8))
  fig = bene_data.groupby(['ChronicCond_Alzheimer'])['OPAnnualDeductibleAmt'].sum().plot(kind='bar', color=['orange','palegreen'])
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{str(round((height*100)/(bene_data["OPAnnualDeductibleAmt"].sum()),2))+"%"}', (x + width/2, y + height*1.01), ha='center', fontsize=13.5, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic ALZH Disease?")
  plt.xticks(ticks=[0,1], labels=['YES', 'NO'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum of OP Co-payment \n")
  plt.grid(which='major', linestyle="--", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of OP Co-payment paid by patient : 'ChronicCond_Alzheimer'\n")

# 1 means +ve with Chronic ALZH Disease
# 2 means -ve with Chronic ALZH Disease
print(pd.DataFrame(bene_data.groupby(['ChronicCond_Alzheimer'])['OPAnnualDeductibleAmt'].sum()),"\n")


# In[44]:


#Amount paid by the Alzhimers patients who are not admitted is low than the other group


# In[45]:


# Number of beneficiaries with chronic or non-chronic conditions
pd.DataFrame(bene_data.groupby(['ChronicCond_Alzheimer'])['BeneID'].count())


# In[46]:


CC_ALZH_IP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_Alzheimer'])['IPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_Alzheimer'])['BeneID'].count())
CC_ALZH_IP_R.columns = ['AVG IP Reimbursement Amt']
CC_ALZH_IP_R


# In[47]:


CC_ALZH_OP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_Alzheimer'])['OPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_Alzheimer'])['BeneID'].count())
CC_ALZH_OP_R.columns = ['AVG OP Reimbursement Amt']
CC_ALZH_OP_R


# In[48]:


CC_ALZH_IP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_Alzheimer'])['IPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_Alzheimer'])['BeneID'].count())
CC_ALZH_IP_D.columns = ['AVG IP Co-payment Amt']
CC_ALZH_IP_D


# In[49]:


CC_ALZH_OP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_Alzheimer'])['OPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_Alzheimer'])['BeneID'].count())
CC_ALZH_OP_D.columns = ['AVG OP Co-payment Amt']
CC_ALZH_OP_D


# In[50]:


CC_ALZH_all_amts = pd.concat([CC_ALZH_IP_R, CC_ALZH_OP_R, CC_ALZH_IP_D, CC_ALZH_OP_D], axis=1)
CC_ALZH_all_amts


# In[ ]:





# In[51]:


with plt.style.context('seaborn-poster'):
  fig = CC_ALZH_all_amts.plot(kind='bar', colormap='rainbow')
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{round(height,0)}', (x + width/2, y + height*1.015), ha='center', fontsize=12, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic ALZH Disease?")
  plt.xticks(ticks=[0,1], labels=['YES', 'NO'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum \n")
  plt.grid(which='major', linestyle="-.", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of various amounts : 'ChronicCond_Alzheimer'\n")

# 1 means +ve with Chronic ALZH Disease
# 2 means -ve with Chronic ALZH Disease
print(CC_ALZH_all_amts,"\n")


# In[52]:


# Number of beneficiaries with chronic or no-chronic conditions
pd.DataFrame(bene_data.groupby(['ChronicCond_Heartfailure'])['BeneID'].count())


# In[53]:


CC_HF_IP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_Heartfailure'])['IPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_Heartfailure'])['BeneID'].count())
CC_HF_IP_R.columns = ['AVG IP Reimbursement Amt']
CC_HF_IP_R


# In[54]:


CC_HF_OP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_Heartfailure'])['OPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_Heartfailure'])['BeneID'].count())
CC_HF_OP_R.columns = ['AVG OP Reimbursement Amt']
CC_HF_OP_R


# In[55]:


CC_HF_IP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_Heartfailure'])['IPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_Heartfailure'])['BeneID'].count())
CC_HF_IP_D.columns = ['AVG IP Co-payment Amt']
CC_HF_IP_D


# In[56]:


CC_HF_OP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_Heartfailure'])['OPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_Heartfailure'])['BeneID'].count())
CC_HF_OP_D.columns = ['AVG OP Co-payment Amt']
CC_HF_OP_D


# In[57]:


CC_HF_all_amts = pd.concat([CC_HF_IP_R, CC_HF_OP_R, CC_HF_IP_D, CC_HF_OP_D], axis=1)
CC_HF_all_amts


# In[58]:


with plt.style.context('seaborn-poster'):
  fig = CC_HF_all_amts.plot(kind='bar', colormap='rainbow')
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{round(height,0)}', (x + width/2, y + height*1.015), ha='center', fontsize=12, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic HF Disease?")
  plt.xticks(ticks=[0,1], labels=['YES', 'NO'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum \n")
  plt.grid(which='major', linestyle="-.", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of various amounts : 'ChronicCond_Heartfailure'\n")

# 1 means +ve with Chronic HF Disease
# 2 means -ve with Chronic HF Disease
print(CC_HF_all_amts,"\n")


# In[59]:


# Number of beneficiaries with chronic or no-chronic conditions
pd.DataFrame(bene_data.groupby(['ChronicCond_KidneyDisease'])['BeneID'].count())


# In[60]:


CC_KD_IP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_KidneyDisease'])['IPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_KidneyDisease'])['BeneID'].count())
CC_KD_IP_R.columns = ['AVG IP Reimbursement Amt']
CC_KD_IP_R


# In[61]:


CC_KD_OP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_KidneyDisease'])['OPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_KidneyDisease'])['BeneID'].count())
CC_KD_OP_R.columns = ['AVG OP Reimbursement Amt']
CC_KD_OP_R


# In[62]:


CC_KD_IP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_KidneyDisease'])['IPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_KidneyDisease'])['BeneID'].count())
CC_KD_IP_D.columns = ['AVG IP Co-payment Amt']
CC_KD_IP_D


# In[63]:


CC_KD_OP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_KidneyDisease'])['OPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_KidneyDisease'])['BeneID'].count())
CC_KD_OP_D.columns = ['AVG OP Co-payment Amt']
CC_KD_OP_D


# In[64]:


CC_KD_all_amts = pd.concat([CC_KD_IP_R, CC_KD_OP_R, CC_KD_IP_D, CC_KD_OP_D], axis=1)
CC_KD_all_amts


# In[65]:


# Here, I'm displaying the Total Annual Sum of IP Co-payment for 'ChronicCond_KidneyDisease'
with plt.style.context('seaborn-poster'):
  fig = CC_KD_all_amts.plot(kind='bar', colormap='rainbow')
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{round(height,0)}', (x + width/2, y + height*1.015), ha='center', fontsize=12, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic KD Disease?")
  plt.xticks(ticks=[0,1], labels=['YES', 'NO'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum \n")
  plt.grid(which='major', linestyle="-.", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of various amounts : 'ChronicCond_KidneyDisease'\n")

# 1 means +ve with Chronic KD Disease
# 2 means -ve with Chronic KD Disease
print(CC_KD_all_amts,"\n")


# In[66]:


# Number of beneficiaries with chronic or no-chronic conditions
pd.DataFrame(bene_data.groupby(['ChronicCond_Cancer'])['BeneID'].count())


# In[67]:


CC_CN_IP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_Cancer'])['IPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_Cancer'])['BeneID'].count())
CC_CN_IP_R.columns = ['AVG IP Reimbursement Amt']
CC_CN_IP_R


# In[68]:


CC_CN_OP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_Cancer'])['OPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_Cancer'])['BeneID'].count())
CC_CN_OP_R.columns = ['AVG OP Reimbursement Amt']
CC_CN_OP_R


# In[69]:


CC_CN_IP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_Cancer'])['IPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_Cancer'])['BeneID'].count())
CC_CN_IP_D.columns = ['AVG IP Co-payment Amt']
CC_CN_IP_D


# In[70]:


CC_CN_OP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_Cancer'])['OPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_Cancer'])['BeneID'].count())
CC_CN_OP_D.columns = ['AVG OP Co-payment Amt']
CC_CN_OP_D


# In[71]:


CC_CN_all_amts = pd.concat([CC_CN_IP_R, CC_CN_OP_R, CC_CN_IP_D, CC_CN_OP_D], axis=1)
CC_CN_all_amts


# In[72]:


# Here, I'm displaying the Total Annual Sum of IP Co-payment for 'ChronicCond_Cancer'
with plt.style.context('seaborn-poster'):
  fig = CC_CN_all_amts.plot(kind='bar', colormap='rainbow')
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{round(height,0)}', (x + width/2, y + height*1.015), ha='center', fontsize=12, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic CN Disease?")
  plt.xticks(ticks=[0,1], labels=['YES', 'NO'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum \n")
  plt.grid(which='major', linestyle="-.", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of various amounts : 'ChronicCond_Cancer'\n")

# 1 means +ve with Chronic CN Disease
# 2 means -ve with Chronic CN Disease
print(CC_CN_all_amts,"\n")


# In[73]:


# Number of beneficiaries with chronic or no-chronic conditions
pd.DataFrame(bene_data.groupby(['ChronicCond_ObstrPulmonary'])['BeneID'].count())


# In[74]:


CC_PL_IP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_ObstrPulmonary'])['IPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_ObstrPulmonary'])['BeneID'].count())
CC_PL_IP_R.columns = ['AVG IP Reimbursement Amt']
CC_PL_IP_R


# In[75]:


CC_PL_OP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_ObstrPulmonary'])['OPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_ObstrPulmonary'])['BeneID'].count())
CC_PL_OP_R.columns = ['AVG OP Reimbursement Amt']
CC_PL_OP_R


# In[76]:


CC_PL_IP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_ObstrPulmonary'])['IPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_ObstrPulmonary'])['BeneID'].count())
CC_PL_IP_D.columns = ['AVG IP Co-payment Amt']
CC_PL_IP_D


# In[77]:


CC_PL_OP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_ObstrPulmonary'])['OPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_ObstrPulmonary'])['BeneID'].count())
CC_PL_OP_D.columns = ['AVG OP Co-payment Amt']
CC_PL_OP_D


# In[78]:


CC_PL_all_amts = pd.concat([CC_PL_IP_R, CC_PL_OP_R, CC_PL_IP_D, CC_PL_OP_D], axis=1)
CC_PL_all_amts


# In[79]:


# Here, I'm displaying the Total Annual Sum of IP Co-payment for 'ChronicCond_ObstrPulmonary'
with plt.style.context('seaborn-poster'):
  fig = CC_PL_all_amts.plot(kind='bar', colormap='rainbow')
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{round(height,0)}', (x + width/2, y + height*1.015), ha='center', fontsize=12, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic PL Disease?")
  plt.xticks(ticks=[0,1], labels=['YES', 'NO'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum \n")
  plt.grid(which='major', linestyle="-.", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of various amounts : 'ChronicCond_ObstrPulmonary'\n")

# 1 means +ve with Chronic PL Disease
# 2 means -ve with Chronic PL Disease


# In[80]:


# Number of beneficiaries with chronic or no-chronic conditions
pd.DataFrame(bene_data.groupby(['ChronicCond_Depression'])['BeneID'].count())


# In[81]:


CC_DP_IP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_Depression'])['IPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_Depression'])['BeneID'].count())
CC_DP_IP_R.columns = ['AVG IP Reimbursement Amt']
CC_DP_IP_R


# In[82]:


CC_DP_OP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_Depression'])['OPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_Depression'])['BeneID'].count())
CC_DP_OP_R.columns = ['AVG OP Reimbursement Amt']
CC_DP_OP_R


# In[83]:


CC_DP_IP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_Depression'])['IPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_Depression'])['BeneID'].count())
CC_DP_IP_D.columns = ['AVG IP Co-payment Amt']
CC_DP_IP_D


# In[84]:


CC_DP_OP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_Depression'])['OPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_Depression'])['BeneID'].count())
CC_DP_OP_D.columns = ['AVG OP Co-payment Amt']
CC_DP_OP_D


# In[85]:


CC_DP_all_amts = pd.concat([CC_DP_IP_R, CC_DP_OP_R, CC_DP_IP_D, CC_DP_OP_D], axis=1)
CC_DP_all_amts


# In[86]:


# Here, I'm displaying the Total Annual Sum of IP Co-payment for 'ChronicCond_Depression'
with plt.style.context('seaborn-poster'):
  fig = CC_DP_all_amts.plot(kind='bar', colormap='rainbow')
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{round(height,0)}', (x + width/2, y + height*1.015), ha='center', fontsize=12, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic DP Disease?")
  plt.xticks(ticks=[0,1], labels=['YES', 'NO'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum \n")
  plt.grid(which='major', linestyle="-.", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of various amounts : 'ChronicCond_Depression'\n")


# In[87]:


# Number of beneficiaries with chronic or no-chronic conditions
pd.DataFrame(bene_data.groupby(['ChronicCond_Diabetes'])['BeneID'].count())


# In[88]:


CC_DB_IP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_Diabetes'])['IPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_Diabetes'])['BeneID'].count())
CC_DB_IP_R.columns = ['AVG IP Reimbursement Amt']
CC_DB_IP_R


# In[89]:


CC_DB_OP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_Diabetes'])['OPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_Diabetes'])['BeneID'].count())
CC_DB_OP_R.columns = ['AVG OP Reimbursement Amt']
CC_DB_OP_R


# In[90]:


CC_DB_IP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_Diabetes'])['IPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_Diabetes'])['BeneID'].count())
CC_DB_IP_D.columns = ['AVG IP Co-payment Amt']
CC_DB_IP_D


# In[91]:


CC_DB_OP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_Diabetes'])['OPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_Diabetes'])['BeneID'].count())
CC_DB_OP_D.columns = ['AVG OP Co-payment Amt']
CC_DB_OP_D


# In[ ]:





# In[92]:


CC_DB_all_amts = pd.concat([CC_DB_IP_R, CC_DB_OP_R, CC_DB_IP_D, CC_DB_OP_D], axis=1)
CC_DB_all_amts


# In[93]:


# Here, I'm displaying the Total Annual Sum of IP Co-payment for 'ChronicCond_Diabetes'
with plt.style.context('seaborn-poster'):
  fig = CC_DB_all_amts.plot(kind='bar', colormap='rainbow')
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{round(height,0)}', (x + width/2, y + height*1.015), ha='center', fontsize=12, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic DB Disease?")
  plt.xticks(ticks=[0,1], labels=['YES', 'NO'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum \n")
  plt.grid(which='major', linestyle="-.", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of various amounts : 'ChronicCond_Diabetes'\n")

# 1 means +ve with Chronic DB Disease
# 2 means -ve with Chronic DB Disease


# In[94]:


# Number of beneficiaries with chronic or no-chronic conditions
pd.DataFrame(bene_data.groupby(['ChronicCond_IschemicHeart'])['BeneID'].count())


# In[95]:


CC_IH_IP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_IschemicHeart'])['IPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_IschemicHeart'])['BeneID'].count())
CC_IH_IP_R.columns = ['AVG IP Reimbursement Amt']
CC_IH_IP_R


# In[96]:


CC_IH_OP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_IschemicHeart'])['OPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_IschemicHeart'])['BeneID'].count())
CC_IH_OP_R.columns = ['AVG OP Reimbursement Amt']
CC_IH_OP_R


# In[97]:


CC_IH_IP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_IschemicHeart'])['IPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_IschemicHeart'])['BeneID'].count())
CC_IH_IP_D.columns = ['AVG IP Co-payment Amt']
CC_IH_IP_D


# In[98]:


CC_IH_OP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_IschemicHeart'])['OPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_IschemicHeart'])['BeneID'].count())
CC_IH_OP_D.columns = ['AVG OP Co-payment Amt']
CC_IH_OP_D


# In[99]:


CC_IH_all_amts = pd.concat([CC_IH_IP_R, CC_IH_OP_R, CC_IH_IP_D, CC_IH_OP_D], axis=1)
CC_IH_all_amts


# In[100]:


# Here, I'm displaying the Total Annual Sum of IP Co-payment for 'ChronicCond_IschemicHeart'
with plt.style.context('seaborn-poster'):
  fig = CC_IH_all_amts.plot(kind='bar', colormap='rainbow')
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{round(height,0)}', (x + width/2, y + height*1.015), ha='center', fontsize=12, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic IH Disease?")
  plt.xticks(ticks=[0,1], labels=['YES', 'NO'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum \n")
  plt.grid(which='major', linestyle="-.", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of various amounts : 'ChronicCond_IschemicHeart'\n")

# 1 means +ve with Chronic IH Disease
# 2 means -ve with Chronic IH Disease


# In[101]:


# Number of beneficiaries with chronic or no-chronic conditions
pd.DataFrame(bene_data.groupby(['ChronicCond_Osteoporasis'])['BeneID'].count())


# In[102]:


CC_OS_IP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_Osteoporasis'])['IPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_Osteoporasis'])['BeneID'].count())
CC_OS_IP_R.columns = ['AVG IP Reimbursement Amt']
CC_OS_IP_R


# In[103]:


CC_OS_OP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_Osteoporasis'])['OPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_Osteoporasis'])['BeneID'].count())
CC_OS_OP_R.columns = ['AVG OP Reimbursement Amt']
CC_OS_OP_R


# In[104]:


CC_OS_IP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_Osteoporasis'])['IPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_Osteoporasis'])['BeneID'].count())
CC_OS_IP_D.columns = ['AVG IP Co-payment Amt']
CC_OS_IP_D


# In[105]:


CC_OS_OP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_Osteoporasis'])['OPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_Osteoporasis'])['BeneID'].count())
CC_OS_OP_D.columns = ['AVG OP Co-payment Amt']
CC_OS_OP_D


# In[106]:


CC_OS_all_amts = pd.concat([CC_OS_IP_R, CC_OS_OP_R, CC_OS_IP_D, CC_OS_OP_D], axis=1)
CC_OS_all_amts


# In[107]:


# Here, I'm displaying the Total Annual Sum of IP Co-payment for 'ChronicCond_Osteoporasis'
with plt.style.context('seaborn-poster'):
  fig = CC_OS_all_amts.plot(kind='bar', colormap='rainbow')
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{round(height,0)}', (x + width/2, y + height*1.015), ha='center', fontsize=12, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic OS Disease?")
  plt.xticks(ticks=[0,1], labels=['YES', 'NO'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum \n")
  plt.grid(which='major', linestyle="-.", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of various amounts : 'ChronicCond_Osteoporasis'\n")

# 1 means +ve with Chronic OS Disease
# 2 means -ve with Chronic OS Disease


# In[108]:


# Number of beneficiaries with chronic or no-chronic conditions
pd.DataFrame(bene_data.groupby(['ChronicCond_rheumatoidarthritis'])['BeneID'].count())


# In[109]:


CC_RH_IP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_rheumatoidarthritis'])['IPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_rheumatoidarthritis'])['BeneID'].count())
CC_RH_IP_R.columns = ['AVG IP Reimbursement Amt']
CC_RH_IP_R


# In[110]:


CC_RH_OP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_rheumatoidarthritis'])['OPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_rheumatoidarthritis'])['BeneID'].count())
CC_RH_OP_R.columns = ['AVG OP Reimbursement Amt']
CC_RH_OP_R


# In[111]:


CC_RH_IP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_rheumatoidarthritis'])['IPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_rheumatoidarthritis'])['BeneID'].count())
CC_RH_IP_D.columns = ['AVG IP Co-payment Amt']
CC_RH_IP_D


# In[112]:


CC_RH_OP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_rheumatoidarthritis'])['OPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_rheumatoidarthritis'])['BeneID'].count())
CC_RH_OP_D.columns = ['AVG OP Co-payment Amt']
CC_RH_OP_D


# In[113]:


CC_RH_all_amts = pd.concat([CC_RH_IP_R, CC_RH_OP_R, CC_RH_IP_D, CC_RH_OP_D], axis=1)
CC_RH_all_amts


# In[114]:


# Here, I'm displaying the Total Annual Sum of IP Co-payment for 'ChronicCond_rheumatoidarthritis'
with plt.style.context('seaborn-poster'):
  fig = CC_RH_all_amts.plot(kind='bar', colormap='rainbow')
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{round(height,0)}', (x + width/2, y + height*1.015), ha='center', fontsize=12, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic RH Disease?")
  plt.xticks(ticks=[0,1], labels=['YES', 'NO'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum \n")
  plt.grid(which='major', linestyle="-.", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of various amounts : 'ChronicCond_rheumatoidarthritis'\n")

# 1 means +ve with Chronic RH Disease
# 2 means -ve with Chronic RH Disease


# In[115]:


# Number of beneficiaries with chronic or no-chronic conditions
pd.DataFrame(bene_data.groupby(['ChronicCond_stroke'])['BeneID'].count())


# In[116]:


CC_ST_IP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_stroke'])['IPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_stroke'])['BeneID'].count())
CC_ST_IP_R.columns = ['AVG IP Reimbursement Amt']
CC_ST_IP_R


# In[117]:


CC_ST_OP_R = pd.DataFrame(bene_data.groupby(['ChronicCond_stroke'])['OPAnnualReimbursementAmt'].sum() / bene_data.groupby(['ChronicCond_stroke'])['BeneID'].count())
CC_ST_OP_R.columns = ['AVG OP Reimbursement Amt']
CC_ST_OP_R


# In[118]:


CC_ST_IP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_stroke'])['IPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_stroke'])['BeneID'].count())
CC_ST_IP_D.columns = ['AVG IP Co-payment Amt']
CC_ST_IP_D


# In[119]:


CC_ST_OP_D = pd.DataFrame(bene_data.groupby(['ChronicCond_stroke'])['OPAnnualDeductibleAmt'].sum() / bene_data.groupby(['ChronicCond_stroke'])['BeneID'].count())
CC_ST_OP_D.columns = ['AVG OP Co-payment Amt']
CC_ST_OP_D


# In[120]:


CC_ST_all_amts = pd.concat([CC_ST_IP_R, CC_ST_OP_R, CC_ST_IP_D, CC_ST_OP_D], axis=1)
CC_ST_all_amts


# In[121]:


# Here, I'm displaying the Total Annual Sum of IP Co-payment for 'ChronicCond_stroke'
with plt.style.context('seaborn-poster'):
  fig = CC_ST_all_amts.plot(kind='bar', colormap='rainbow')
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{round(height,0)}', (x + width/2, y + height*1.015), ha='center', fontsize=12, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic ST Disease?")
  plt.xticks(ticks=[0,1], labels=['YES', 'NO'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum \n")
  plt.grid(which='major', linestyle="-.", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of various amounts : 'ChronicCond_stroke'\n")

# 1 means +ve with Chronic ST Disease
# 2 means -ve with Chronic ST Disease


# In[122]:


# Number of beneficiaries with chronic or no-chronic conditions
pd.DataFrame(bene_data.groupby(['RenalDiseaseIndicator'])['BeneID'].count())


# In[123]:


RKD_IP_R = pd.DataFrame(bene_data.groupby(['RenalDiseaseIndicator'])['IPAnnualReimbursementAmt'].sum() / bene_data.groupby(['RenalDiseaseIndicator'])['BeneID'].count())
RKD_IP_R.columns = ['AVG IP Reimbursement Amt']
RKD_IP_R


# In[124]:


RKD_OP_R = pd.DataFrame(bene_data.groupby(['RenalDiseaseIndicator'])['OPAnnualReimbursementAmt'].sum() / bene_data.groupby(['RenalDiseaseIndicator'])['BeneID'].count())
RKD_OP_R.columns = ['AVG OP Reimbursement Amt']
RKD_OP_R


# In[125]:


RKD_IP_D = pd.DataFrame(bene_data.groupby(['RenalDiseaseIndicator'])['IPAnnualDeductibleAmt'].sum() / bene_data.groupby(['RenalDiseaseIndicator'])['BeneID'].count())
RKD_IP_D.columns = ['AVG IP Co-payment Amt']
RKD_IP_D


# In[126]:


RKD_OP_D = pd.DataFrame(bene_data.groupby(['RenalDiseaseIndicator'])['OPAnnualDeductibleAmt'].sum() / bene_data.groupby(['RenalDiseaseIndicator'])['BeneID'].count())
RKD_OP_D.columns = ['AVG OP Co-payment Amt']
RKD_OP_D


# In[127]:


RKD_all_amts = pd.concat([RKD_IP_R, RKD_OP_R, RKD_IP_D, RKD_OP_D], axis=1)
RKD_all_amts


# In[128]:


# Here, I'm displaying the Total Annual Sum of IP Co-payment for 'RenalDiseaseIndicator'
with plt.style.context('seaborn-poster'):
  fig = RKD_all_amts.plot(kind='bar', colormap='rainbow')
  # Using the "patches" function we will get the location of the rectangle bars from the graph.
  ## Then by using those location(width & height) values we will add the annotations
  for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{round(height,0)}', (x + width/2, y + height*1.015), ha='center', fontsize=12, rotation=0)
  # Providing the labels and title to the graph
  plt.xlabel("\nHaving Chronic Renal Kidney Disease?")
  plt.xticks(ticks=[0,1], labels=['NO', 'YES'], fontsize=13, rotation=30)
  plt.ylabel("Total Annual Sum \n")
  plt.grid(which='major', linestyle="-.", color='lightgrey')
  plt.minorticks_on()
  plt.title("Total Annual Sum of various amounts : 'RenalDiseaseIndicator'\n")

# Y means +ve with Renal Kidney Disease
# 0 means -ve with Renal Kidney Disease


# In[129]:


def cal_display_percentiles(x_col, y_col, title_lbl, x_filter_code):
    """
    Description : This function is created for calculating and generating the percentiles for pre-disease indicators.
    
    Input: It accepts below parameters:
        1. x_col : Disease indicator feature name.
        2. y_col : Feature like re-imbursement or deductible amount whose percentiles you want to generate.
        3. title_lbl : Label to be provided in the title of the plot.
        4. x_filter_code : Category code for which you want to generate the percentiles.
        
    Output: It returns the dataframe having percentiles and their respective values for the specific disease indicator feature.
    And, it displays the pointplot graph of the same.
    """
    percentiles = []
    percentiles_vals = []

    # Calculating & storing the various percentiles and their respective values
    for val in [0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.999,0.9999,0.99999,0.999999,1.0]:
        percentile = round(float(val*100),6)
        percentiles.append(percentile)

        percentile_val = round(bene_data[bene_data[x_col] == x_filter_code][y_col].quantile(val),1)
        percentiles_vals.append(percentile_val)

    # Creating the temp dataframe for displaying the results
    tmp_percentiles = pd.DataFrame([percentiles, percentiles_vals]).T
    tmp_percentiles.columns = ['Percentiles', 'Values']

    # Here, I'm displaying the Percentiles values for all disease code features
    with plt.style.context('seaborn-poster'):
        plt.figure(figsize=(15,7))
        sns.pointplot(data=tmp_percentiles, x='Percentiles', y='Values', markers="o", palette='spring')
        sns.pointplot(data=tmp_percentiles, x='Percentiles', y='Values', markers="", color='grey', linestyles="solid")
        # Providing the labels and title to the graph
        plt.xlabel("\nPercentiles")
        plt.xticks(rotation=90, size=12)
        plt.ylabel("Total Annual `{}` Sum \n".format(y_col))
        plt.grid(which='major', linestyle="-.", color='lightpink')
        plt.minorticks_on()
        plt.title("Percentile values of `{}` :: `{}`\n".format(y_col,title_lbl))
        
    return tmp_percentiles


# In[ ]:





# In[ ]:





# In[130]:


RKD_YES_IP_R_percentiles = cal_display_percentiles(x_col='RenalDiseaseIndicator', 
                                                   y_col='IPAnnualReimbursementAmt',
                                                   title_lbl="Renal Kidney Disease = YES",
                                                   x_filter_code='Y')


# In[ ]:





# In[ ]:





# In[131]:


RKD_NO_IP_R_percentiles = cal_display_percentiles(x_col='RenalDiseaseIndicator', 
                                                   y_col='IPAnnualReimbursementAmt',
                                                   title_lbl="Renal Kidney Disease = NO",
                                                   x_filter_code='0')


# In[132]:


RKD_YES_OP_R_percentiles = cal_display_percentiles(x_col='RenalDiseaseIndicator', 
                                                   y_col='OPAnnualReimbursementAmt',
                                                   title_lbl="Renal Kidney Disease = YES",
                                                   x_filter_code='Y')


# In[133]:


RKD_NO_OP_R_percentiles = cal_display_percentiles(x_col='RenalDiseaseIndicator', 
                                                   y_col='OPAnnualReimbursementAmt',
                                                   title_lbl="Renal Kidney Disease = NO",
                                                   x_filter_code='0')


# In[134]:


CC_ST_YES_IP_R_percentiles = cal_display_percentiles(x_col='ChronicCond_stroke', 
                                                     y_col='IPAnnualReimbursementAmt',
                                                     title_lbl="ChronicCond_stroke = YES",
                                                     x_filter_code=1)


# In[135]:


CC_ST_NO_IP_R_percentiles = cal_display_percentiles(x_col='ChronicCond_stroke', 
                                                     y_col='IPAnnualReimbursementAmt',
                                                     title_lbl="ChronicCond_stroke = NO",
                                                     x_filter_code=2)


# In[136]:


CC_ST_YES_OP_R_percentiles = cal_display_percentiles(x_col='ChronicCond_stroke', 
                                                     y_col='OPAnnualReimbursementAmt',
                                                     title_lbl="ChronicCond_stroke = YES",
                                                     x_filter_code=1)


# In[137]:


CC_ST_NO_OP_R_percentiles = cal_display_percentiles(x_col='ChronicCond_stroke', 
                                                     y_col='OPAnnualReimbursementAmt',
                                                     title_lbl="ChronicCond_stroke = NO",
                                                     x_filter_code=2)


# In[138]:


#EDA In-patients & Out-Patients


# In[139]:


#In-patients


# In[140]:


ip_data.shape


# In[141]:


ip_data.head()


# In[142]:


ip_data.dtypes


# In[143]:


#Out-patients


# In[144]:


op_data.shape


# In[145]:


op_data.head()


# In[146]:


op_data.dtypes


# In[147]:


#Beneficiaries who are admitted and not admitted


# In[148]:


ip_bene_unq = set(ip_data['BeneID'])
op_bene_unq = set(op_data['BeneID'])

len(ip_bene_unq), len(op_bene_unq)


# In[149]:


#Number of patients who are only admitted


# In[150]:


only_in_patients = ip_bene_unq.intersection(op_bene_unq)
len(only_in_patients)


# In[151]:


#number patients who are not admitted at all


# In[152]:


only_out_patients = op_bene_unq.difference(ip_bene_unq)
len(only_out_patients)


# In[153]:


patients_counts = pd.DataFrame([len(only_in_patients), len(only_out_patients)]).T
patients_counts.columns = ['Only In-patients', 'Only Out-patients']
patients_counts


# In[154]:


tot_patients = len(only_in_patients) + len(only_out_patients)
tot_patients


# In[155]:


# Here, I'm displaying the number of only in-patients and out-patients
with plt.style.context('seaborn-poster'):
    fig = patients_counts.plot(kind='bar',colormap='copper')
    # Using the "patches" function we will get the location of the rectangle bars from the graph.
    ## Then by using those location(width & height) values we will add the annotations
    for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/tot_patients,2))+"%"}', (x + width/2, y + height*1.015), ha='center', fontsize=13.5)
    # Providing the labels and title to the graph
    plt.xticks(labels=["Patients Counts"], ticks=[0], rotation=10)
    plt.ylabel("Number or % share of patients\n")
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.minorticks_on()
    plt.title("Number of only In or Out patients\n")
    plt.plot();


# In[156]:


#from the above we can observe that 80% of the patients were treated without admission


# In[157]:


#Claim Duration Days


# In[158]:


ip_data['ClaimStartDt'] = pd.to_datetime(ip_data['ClaimStartDt'], format="%Y-%m-%d")
ip_data['ClaimEndDt'] = pd.to_datetime(ip_data['ClaimEndDt'], format="%Y-%m-%d")


# In[159]:


ip_data['Claim_Duration'] = (ip_data['ClaimEndDt'] - ip_data['ClaimStartDt']).dt.days


# In[160]:


# Here, I'm displaying the number of only in-patients and out-patients
with plt.style.context('seaborn-poster'):
    plt.figure(figsize=(12,8))
    ip_data['Claim_Duration'].plot(kind='hist', colormap="Accent");
    # Providing the labels and title to the graph
    plt.xlabel("Claim Duration(in days)")
    plt.minorticks_on()
    plt.title("Distribution of Claim Duration Days\n")
    plt.plot();


# In[161]:


#From the above we can observe that Maximun number of claims were filed within 7 days


# In[162]:


#What is the relationship b/w Amount of Insurance Claim Reimbursed v/s Claim Clearance Days?


# In[163]:


unq_claim_duration_days = ip_data['Claim_Duration'].unique()
unq_claim_duration_days


# In[164]:


tot_claims_filed_for_specific_days = pd.DataFrame(ip_data.groupby(['Claim_Duration'])['BeneID'].count())
tot_claims_filed_for_specific_days


# In[165]:


tot_insc_amount_for_claim_durations = pd.DataFrame(ip_data.groupby(['Claim_Duration'])['InscClaimAmtReimbursed'].sum())
tot_insc_amount_for_claim_durations


# In[166]:


claim_clearance_amts = pd.merge(left=tot_claims_filed_for_specific_days, right=tot_insc_amount_for_claim_durations,
                                how='inner',
                                left_on=tot_claims_filed_for_specific_days.index,
                                right_on=tot_insc_amount_for_claim_durations.index)

claim_clearance_amts.columns = ['Claim_durations_in_days', 'Total_claims', 'All_Claims_Total_Amount']
claim_clearance_amts.head()


# In[167]:


claim_clearance_amts['Avg_Claim_Insc_Amount'] = np.round(claim_clearance_amts['All_Claims_Total_Amount']/claim_clearance_amts['Total_claims'],2)


# In[168]:


claim_clearance_amts.head()


# In[169]:


with plt.style.context('seaborn'):
    plt.figure(figsize=(16,16))
    sns.pointplot(data=claim_clearance_amts, x='Claim_durations_in_days', y='Total_claims', 
                  color='k', markers="^", linestyles="")
    sns.pointplot(data=claim_clearance_amts, x='Claim_durations_in_days', y='Total_claims', 
                  color='coral', markers="", linestyles="-")
     
    # Providing the labels and title to the graph
    plt.xticks(rotation=90)
    plt.xlabel("\nClaims Durations(in days)")
    plt.ylabel("Total Claims\n")
    plt.yticks(np.arange(0,7500,200))
    plt.grid(which='major', linestyle="--", color='White')
    plt.minorticks_on()
    plt.title('\nTrend of "Total Filed Claims" for every duration(in days)')
    plt.plot();


# In[170]:


#From the above we can observe that maximum number of claims were filed for 3 days and very few were filed for greater than 15 days
#There is a spike for 35 days claim period
#And 600 were filed for 0 days i.e claim started and ended on the same day


# In[171]:


with plt.style.context('seaborn'):
    plt.figure(figsize=(16,7))
    sns.pointplot(data=claim_clearance_amts, x='Claim_durations_in_days', y='Avg_Claim_Insc_Amount', 
                  color='k', markers="^", linestyles="")
    sns.pointplot(data=claim_clearance_amts, x='Claim_durations_in_days', y='Avg_Claim_Insc_Amount', 
                  color='coral', markers="", linestyles="-")
    # Providing the labels and title to the graph
    plt.xticks(rotation=90)
    plt.xlabel("\nVarious Claims duration (in days)")
    plt.ylabel(" ")
    plt.grid(which='major', linestyle="--", color='white')
    plt.minorticks_on()
    plt.title('\nTrend of "Avg Re-imbursed Claim Amount"')
    plt.plot();


# In[172]:


#From the above we can observe that reimbursement increases as the clain duration days increases. however we already know very few were filed for more than 15 days
#We can also observe that claims duration 30-35 has highest reimbursement amounts


# In[173]:


with plt.style.context('seaborn'):
    plt.figure(figsize=(16,7))
    sns.pointplot(data=claim_clearance_amts, x='Claim_durations_in_days', y='All_Claims_Total_Amount', color='blue')
    # Providing the labels and title to the graph
    plt.xticks(rotation=90)
    plt.xlabel("\nClaims Durations (in days)")
    plt.ylabel("Total Re-imbursed Claim Amount")
    plt.grid(which='major', linestyle="--", color='white')
    plt.minorticks_on()
    plt.title("\nTrend of `Total Re-imbursed Claim Amount` for each filed duration(in days)")
    plt.plot();


# In[174]:


#From above we can observe that total reimbursement amount is highest for 3 days claims
#And we can clearly see a spike for 30-35 days claims which can be a potential frauds


# In[175]:


#relationship b/w DeductibleAmtPaid and Re-imbursed Amount


# In[176]:


no_of_claim_with_no_copay = ip_data[ip_data['DeductibleAmtPaid'].isna()].shape[0]
no_of_claim_with_no_copay


# In[177]:


no_of_claim_with_copay = ip_data[~ip_data['DeductibleAmtPaid'].isna()].shape[0]
no_of_claim_with_copay


# In[178]:


percent_of_no_copay_claims = round((no_of_claim_with_no_copay / (no_of_claim_with_copay + no_of_claim_with_no_copay)) * 100,1)
print("### Percentage of claims with no co-payment or deductible --> {}% ###".format(percent_of_no_copay_claims))


# In[179]:


re_imbursed_amt_for_no_copay = ip_data[ip_data['DeductibleAmtPaid'].isna()]['InscClaimAmtReimbursed'].sum()
re_imbursed_amt_for_no_copay


# In[180]:


re_imbursed_amt_with_some_copay = ip_data[~ip_data['DeductibleAmtPaid'].isna()]['InscClaimAmtReimbursed'].sum()
re_imbursed_amt_with_some_copay


# In[181]:


tot_sum_of_claims_with_copay = re_imbursed_amt_with_some_copay / (re_imbursed_amt_with_some_copay + re_imbursed_amt_for_no_copay)
tot_sum_of_claims_with_no_copay = re_imbursed_amt_for_no_copay / (re_imbursed_amt_with_some_copay + re_imbursed_amt_for_no_copay)


# In[182]:


percent_of_tot_sum_no_copay_claims_amt = round(tot_sum_of_claims_with_no_copay * 100,1)
print("### Percentage of Total Re-imbursed Amount for claims with no co-payment or deductible --> {}% ###".      format(percent_of_tot_sum_no_copay_claims_amt))


# In[183]:


#The above table tells us that there are a 2% of total claims for which there is no co-payment


# In[184]:


##EDA on Out-patients data


# In[185]:


op_data.shape


# In[186]:


#Adding claim clearance days


# In[187]:


op_data['ClaimStartDt'] = pd.to_datetime(op_data['ClaimStartDt'], format="%Y-%m-%d")
op_data['ClaimEndDt'] = pd.to_datetime(op_data['ClaimEndDt'], format="%Y-%m-%d")


# In[188]:


op_data['Claim_Duration'] = (op_data['ClaimEndDt'] - op_data['ClaimStartDt']).dt.days


# In[189]:


with plt.style.context('seaborn-poster'):
    plt.figure(figsize=(12,8))
    op_data['Claim_Duration'].plot(kind='hist', colormap="viridis");
    # Providing the labels and title to the graph
    plt.xlabel("Claim Duration(in days)")
    plt.minorticks_on()
    plt.title("Distribution of Claim Duration Days\n")
    plt.plot();


# In[190]:


#From the above we can deduce that majority of claims were filed for less than or equal to 2 days


# In[191]:


##relationship b/w Claim Duration and Re-imbursed Amount?


# In[192]:


with plt.style.context("seaborn-poster"):
    sns.stripplot(x="Claim_Duration", y="InscClaimAmtReimbursed", data=op_data, palette="plasma")
    # Providing the labels and title to the graph
    plt.xlabel("Claim Durations (in days)")
    plt.ylabel("Claim Re-imbursed Amount\n")
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.minorticks_on()
    plt.title("Various re-imbursed amounts for different claim durations\n")
    plt.plot();


# In[193]:


#From the above we can deduce that most of the reimbursement amounts are less than 20000, very few are more than 100000


# In[194]:


#relationship b/w Amount of Insurance Claim Reimbursed v/s Claim Clearance Days?


# In[195]:


unq_claim_duration_days = op_data['Claim_Duration'].unique()
unq_claim_duration_days


# In[196]:


tot_claims_filed_for_specific_days = pd.DataFrame(op_data.groupby(['Claim_Duration'])['ClaimID'].count())
tot_claims_filed_for_specific_days


# In[197]:


tot_insc_amount_for_claim_durations = pd.DataFrame(op_data.groupby(['Claim_Duration'])['InscClaimAmtReimbursed'].sum())
tot_insc_amount_for_claim_durations


# In[198]:


claim_clearance_amts = pd.merge(left=tot_claims_filed_for_specific_days, right=tot_insc_amount_for_claim_durations,
                                how='inner',
                                left_on=tot_claims_filed_for_specific_days.index,
                                right_on=tot_insc_amount_for_claim_durations.index)

claim_clearance_amts.columns = ['Claim_durations_in_days', 'Total_claims', 'All_Claims_Total_Amount']
claim_clearance_amts.head()


# In[199]:


claim_clearance_amts['Avg_Claim_Insc_Amount'] = np.round(claim_clearance_amts['All_Claims_Total_Amount']/claim_clearance_amts['Total_claims'],2)


# In[200]:


claim_clearance_amts.head()


# In[201]:


with plt.style.context('seaborn'):
    plt.figure(figsize=(16,10))
    fig = sns.barplot(data=claim_clearance_amts, x='Claim_durations_in_days', y='Total_claims', palette='plasma')     
    # Using the "patches" function we will get the location of the rectangle bars from the graph.
    ## Then by using those location(width & height) values we will add the annotations
    for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/claim_clearance_amts["Total_claims"].sum(),2))+"%"}', (x + width/2, y + height*1.02), ha='center', fontsize=9, rotation=0)
    
    # Providing the labels and title to the graph
    plt.xticks(rotation=90)
    plt.xlabel("\nClaims Durations(in days)")
    plt.ylabel("Total Claims\n")
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.minorticks_on()
    plt.title('\nTrend of "Total Filed Claims" for every duration(in days)')
    plt.plot();


# In[202]:


#From the above we can observe that most number of claims were filed for 0 days
#But we can see a spike near 20 days


# In[203]:


with plt.style.context('seaborn'):
    plt.figure(figsize=(14,7))
    sns.pointplot(data=claim_clearance_amts, x='Claim_durations_in_days', y='Avg_Claim_Insc_Amount', 
                  color='blue', markers="^", linestyles="")
    sns.pointplot(data=claim_clearance_amts, x='Claim_durations_in_days', y='Avg_Claim_Insc_Amount', 
                  color='coral', markers="", linestyles="-")
    # Providing the labels and title to the graph
    plt.xticks(rotation=90)
    plt.xlabel("\nVarious Claims duration (in days)")
    plt.ylabel(" ")
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.minorticks_on()
    plt.title('\nTrend of "Avg Re-imbursed Claim Amount"')
    plt.plot();


# In[204]:


#We can observe that average reimbursement amount is same throughout the durations except for 21 and 23days


# In[205]:


with plt.style.context('seaborn'):
    plt.figure(figsize=(16,7))
    sns.pointplot(data=claim_clearance_amts, x='Claim_durations_in_days', y='All_Claims_Total_Amount', color='green')
    # Providing the labels and title to the graph
    plt.xticks(rotation=90)
    plt.xlabel("\nClaims Durations (in days)")
    plt.ylabel("Total Re-imbursed Claim Amount")
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.minorticks_on()
    plt.title("\nTrend of `Total Re-imbursed Claim Amount` for each filed duration(in days)")
    plt.plot();


# In[206]:


#Reimbursement claim amount is highest for 0 days and there no much different throught other durations but again from 19 to 21 days there's a spike in amount
#This can imply potential sign fraud


# In[207]:


#EDA on Providers data


# In[208]:


prov_data = pd.read_csv("C:/Users/sravy/Downloads/archive/Train-1542865627584.csv")


# In[209]:


prov_data.head()


# In[210]:


prov_data.shape


# In[211]:


#Checking fraud and Non-Fraud counts


# In[212]:


print("### The unique number of providers are {}. ###".format(prov_data.shape[0]))


# In[213]:


with plt.style.context('seaborn-poster'):
    fig = prov_data["PotentialFraud"].value_counts().plot(kind='bar', color=['green','orange'])
    # Using the "patches" function we will get the location of the rectangle bars from the graph.
    ## Then by using those location(width & height) values we will add the annotations
    for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/prov_data.shape[0],2))+"%"}', (x + width/2, y + height*1.015), ha='center', fontsize=13.5)
    # Providing the labels and title to the graph
    plt.xlabel("Provider Fraud or Not?")
    plt.ylabel("Number or % share of providers\n")
    plt.yticks(np.arange(0,5100,500))
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.minorticks_on()
    plt.title("Distribution of Fraud & Non-fraud providers\n")
    plt.plot();


# In[214]:


#From the above we can say that 90% of the providers are not fraudsters


# In[215]:


#Adding Ip dataset


# In[216]:


ip_data["Admitted?"] = 1


# In[217]:


ip_data.head()


# In[218]:


#Adding op dataset


# In[219]:


op_data["Admitted?"] = 0


# In[220]:


op_data.head()


# In[221]:


#Merging dataset


# In[222]:


common_cols = [col for col in ip_data.columns if col in op_data.columns]
len(common_cols)


# In[223]:


train_ip_op_df = pd.merge(left=ip_data, right=op_data, left_on=common_cols, right_on=common_cols, how="outer")
train_ip_op_df.shape


# In[224]:


train_ip_op_df.head()


# In[225]:


#merging Ip op with bene


# In[226]:


train_ip_op_bene_df = pd.merge(left=train_ip_op_df, right=bene_data, left_on='BeneID', right_on='BeneID',how='inner')
train_ip_op_bene_df.shape


# In[227]:


#Merging data with Providers


# In[228]:


train_iobp_df = pd.merge(left=train_ip_op_bene_df, right=prov_data, left_on='Provider', right_on='Provider',how='inner')
train_iobp_df.shape


# In[229]:


train_iobp_df["Provider"].nunique()


# In[230]:


train_iobp_df["ClaimID"].nunique()


# In[231]:


# One provider may have been involved in more than one claim. So, does all the claims filed by a potentially fraud provider are all frauds?¶
# - This cannot holds True for all the providers because if one provider has filed say 50 claims then we can't say that all the claims for that provider are fraudulent. 
#     - There may exists a pattern that out of 50 claims a provider files 1 or 2 fraudulent claims. 


# In[232]:


#Therefore, it is a big assumption to make that all the claims filed by a potentially fraud provider are fraudulent.


# In[233]:


prvs_claims_df = pd.DataFrame(train_iobp_df.groupby(['Provider'])['ClaimID'].count()).reset_index()
prvs_claims_tgt_lbls_df = pd.merge(left=prvs_claims_df, right=prov_data, on='Provider', how='inner')
prvs_claims_tgt_lbls_df


# In[234]:


#As shown in the above table, PRV51005 has filed 1165 claims so after joining the datasets all of these will be marked as Fraud.


# In[235]:


print(pd.DataFrame(train_iobp_df['PotentialFraud'].value_counts()), "\n")

with plt.style.context('seaborn-poster'):
    fig = train_iobp_df['PotentialFraud'].value_counts().plot(kind='bar', color=['green','orange'])
    # Using the "patches" function we will get the location of the rectangle bars from the graph.
    ## Then by using those location(width & height) values we will add the annotations
    for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/train_iobp_df.shape[0],2))+"%"}', (x + width/2, y + height*1.015), ha='center', fontsize=13.5)
    # Providing the labels and title to the graph
    plt.xlabel("Fraud or Not?")
    plt.ylabel("Number (or %) of claims\n")
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.minorticks_on()
    plt.title("Distribution of Fraud & Non-fraud claims\n")
    plt.plot();


# In[236]:


# The above plot shows us that, 62% of claims are Non-Fraud and 32% of them are Fraudulent.
# By looking at the percentages we may say that there is a class-imbalance problem but after looking at the number of records it doesn't seem to be a severe class-imbalance problem.
# So, I'll try some class balancing techniques only after training a baseline model w/o any synthetic or class weighting techniques.


# In[237]:


##Top-25 Providers with maximum number of fraudulent cases


# In[239]:


tmp = pd.DataFrame(train_iobp_df.groupby(['Provider','PotentialFraud'])['BeneID'].count()).reset_index()
tmp.columns = ['Provider', 'Fraud?', 'Num_of_cases']
tot_fraud_cases = tmp[tmp['Fraud?'] == 'Yes']['Num_of_cases'].sum()
tot_non_fraud_cases = tmp[tmp['Fraud?'] == 'No']['Num_of_cases'].sum()
tmp['Cases'] = tmp['Fraud?'].apply(lambda val: tot_non_fraud_cases if val == "No" else tot_fraud_cases)
tmp['Percentage'] = round(((tmp['Num_of_cases'] / tmp['Cases']) * 100),2)

tmp.head()


# In[240]:


tmp_only_frauds = tmp[tmp['Fraud?'] == 'Yes'].sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)


# In[241]:


print(tmp_only_frauds[['Provider','Num_of_cases','Percentage']].head(25), "\n")

with plt.style.context('seaborn'):
    plt.figure(figsize=(14,8))
    fig = sns.barplot(data=tmp_only_frauds.iloc[0:25], x="Provider", y="Num_of_cases", palette='Accent')
    # Using the "patches" function we will get the location of the rectangle bars from the graph.
    ## Then by using those location(width & height) values we will add the annotations
    for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/tot_fraud_cases,2))+"%"}', (x + width/2, y + height*1.025), ha='center', fontsize=13.5, rotation=90)
    
    # Providing the labels and title to the graph
    plt.xlabel("\nTop Fraudulent Providers")
    plt.xticks(rotation=90, fontsize=12)
    plt.ylabel("Number (or % share) of Cases\n")
    plt.minorticks_on()
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.title("Top-25 Providers with most number of fraudulent cases\n")
    plt.plot();


# In[242]:


#We can observe that PRV51459 providers has the highest number of fraudalent cases


# In[243]:


## Top-25 Providers with maximum number of non-fraudulent cases


# In[244]:


tmp_only_non_frauds = tmp[tmp['Fraud?'] == 'No'].sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)


# In[245]:


print(tmp_only_non_frauds[['Provider','Num_of_cases','Percentage']].head(25), "\n")

with plt.style.context('seaborn'):
    plt.figure(figsize=(14,8))
    fig = sns.barplot(data=tmp_only_non_frauds.iloc[0:25], x="Provider", y="Num_of_cases", palette='Accent')
    # Using the "patches" function we will get the location of the rectangle bars from the graph.
    ## Then by using those location(width & height) values we will add the annotations
    for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/tot_non_fraud_cases,2))+"%"}', (x + width/2, y + height*1.025), ha='center', fontsize=13.5, rotation=90)
    
    # Providing the labels and title to the graph
    plt.xlabel("\nTop Non-Fraudulent Providers")
    plt.xticks(rotation=90, fontsize=12)
    plt.ylabel("Number (or % share) of Cases\n")
    plt.minorticks_on()
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.title("Top-25 Providers with most number of non-fraudulent cases\n")
    plt.plot();


# In[246]:


#We can observe that provider PRV53750 has the most non fraudalents cases but there's much difference with the rest of the providers


# In[247]:


## Top-25 Attending Physicians with maximum number of fraudulent cases


# In[248]:


tmp = pd.DataFrame(train_iobp_df.groupby(['AttendingPhysician','PotentialFraud'])['BeneID'].count()).reset_index()
tmp.columns = ['AttendingPhysician', 'Fraud?', 'Num_of_cases']
tot_fraud_cases = tmp[tmp['Fraud?'] == 'Yes']['Num_of_cases'].sum()
tot_non_fraud_cases = tmp[tmp['Fraud?'] == 'No']['Num_of_cases'].sum()
tmp['Cases'] = tmp['Fraud?'].apply(lambda val: tot_non_fraud_cases if val == "No" else tot_fraud_cases)
tmp['Percentage'] = round(((tmp['Num_of_cases'] / tmp['Cases']) * 100),2)

tmp.head()


# In[249]:


tmp_only_frauds = tmp[tmp['Fraud?'] == 'Yes'].sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)


# In[250]:


print(tmp_only_frauds[['AttendingPhysician','Num_of_cases','Percentage']].head(25), "\n")

with plt.style.context('seaborn'):
    plt.figure(figsize=(14,8))
    fig = sns.barplot(data=tmp_only_frauds.iloc[0:25], x="AttendingPhysician", y="Num_of_cases", palette='Accent')
    # Using the "patches" function we will get the location of the rectangle bars from the graph.
    ## Then by using those location(width & height) values we will add the annotations
    for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/tot_fraud_cases,2))+"%"}', (x + width/2, y + height*1.025), ha='center', fontsize=13.5, rotation=90)
    
    # Providing the labels and title to the graph
    plt.xlabel("\nTop Fraudulent AttendingPhysician")
    plt.xticks(rotation=90, fontsize=12)
    plt.ylabel("Number (or % share) of Cases\n")
    plt.minorticks_on()
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.title("Top-25 AttendingPhysician with most number of fraudulent cases\n")
    plt.plot();


# In[251]:


#We can observe that attending physician PHY330576 have the highest fraudalent cases, others seen to have very low.


# In[252]:


## Top-25 Attenting Physicians with maximum number of non-fraudulent cases


# In[253]:


tmp_only_non_frauds = tmp[tmp['Fraud?'] == 'No'].sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)


# In[254]:


print(tmp_only_non_frauds[['AttendingPhysician','Num_of_cases','Percentage']].head(25), "\n")

with plt.style.context('seaborn'):
    plt.figure(figsize=(14,8))
    fig = sns.barplot(data=tmp_only_non_frauds.iloc[0:25], x="AttendingPhysician", y="Num_of_cases", palette='Accent')
    # Using the "patches" function we will get the location of the rectangle bars from the graph.
    ## Then by using those location(width & height) values we will add the annotations
    for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/tot_non_fraud_cases,2))+"%"}', (x + width/2, y + height*1.025), ha='center', fontsize=13.5, rotation=90)
    
    # Providing the labels and title to the graph
    plt.xlabel("\nTop Non-Fraudulent AttendingPhysician")
    plt.xticks(rotation=90, fontsize=12)
    plt.ylabel("Number (or % share) of Cases\n")
    plt.minorticks_on()
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.title("Top-25 AttendingPhysician with most number of non-fraudulent cases\n")
    plt.plot();


# In[255]:


#We can observe that attending physicians PHY351121 and PHY375943 have highest non-fraudalent cases when compared to others.


# In[256]:


## Top-25 Operating Physicians with maximum number of fraudulent cases


# In[257]:


tmp = pd.DataFrame(train_iobp_df.groupby(['OperatingPhysician','PotentialFraud'])['BeneID'].count()).reset_index()
tmp.columns = ['OperatingPhysician', 'Fraud?', 'Num_of_cases']
tot_fraud_cases = tmp[tmp['Fraud?'] == 'Yes']['Num_of_cases'].sum()
tot_non_fraud_cases = tmp[tmp['Fraud?'] == 'No']['Num_of_cases'].sum()
tmp['Cases'] = tmp['Fraud?'].apply(lambda val: tot_non_fraud_cases if val == "No" else tot_fraud_cases)
tmp['Percentage'] = round(((tmp['Num_of_cases'] / tmp['Cases']) * 100),2)

tmp.head()


# In[258]:


tmp_only_frauds = tmp[tmp['Fraud?'] == 'Yes'].sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)


# In[259]:


print(tmp_only_frauds[['OperatingPhysician','Num_of_cases','Percentage']].head(25), "\n")

with plt.style.context('seaborn'):
    plt.figure(figsize=(14,8))
    fig = sns.barplot(data=tmp_only_frauds.iloc[0:25], x="OperatingPhysician", y="Num_of_cases", palette='Accent')
    # Using the "patches" function we will get the location of the rectangle bars from the graph.
    ## Then by using those location(width & height) values we will add the annotations
    for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/tot_fraud_cases,2))+"%"}', (x + width/2, y + height*1.025), ha='center', fontsize=13.5, rotation=90)
    
    # Providing the labels and title to the graph
    plt.xlabel("\nTop Fraudulent OperatingPhysician")
    plt.xticks(rotation=90, fontsize=12)
    plt.ylabel("Number (or % share) of Cases\n")
    plt.minorticks_on()
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.title("Top-25 OperatingPhysician with most number of fraudulent cases\n")
    plt.plot();


# In[260]:


#We can observe that operating Physician PHY330576 has the highest number of fraudalent cases


# In[261]:


## Top-25 Operating Physicians with maximum number of non-fraudulent cases


# In[262]:


tmp_only_non_frauds = tmp[tmp['Fraud?'] == 'No'].sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)


# In[263]:


print(tmp_only_non_frauds[['OperatingPhysician','Num_of_cases','Percentage']].head(25), "\n")

with plt.style.context('seaborn'):
    plt.figure(figsize=(14,8))
    fig = sns.barplot(data=tmp_only_non_frauds.iloc[0:25], x="OperatingPhysician", y="Num_of_cases", palette='Accent')
    # Using the "patches" function we will get the location of the rectangle bars from the graph.
    ## Then by using those location(width & height) values we will add the annotations
    for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/tot_non_fraud_cases,2))+"%"}', (x + width/2, y + height*1.025), ha='center', fontsize=13.5, rotation=90)
    
    # Providing the labels and title to the graph
    plt.xlabel("\nTop Non-Fraudulent OperatingPhysician")
    plt.xticks(rotation=90, fontsize=12)
    plt.ylabel("Number (or % share) of Cases\n")
    plt.minorticks_on()
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.title("Top-25 OperatingPhysician with most number of non-fraudulent cases\n")
    plt.plot();


# In[264]:


#We can observe that operating Physicians PHY387900 and PHY351121 has the highest number of non-fraudalent cases but there's no much difference with others.


# In[265]:


## Top-25 Other Physicians with maximum number of fraudulent cases


# In[266]:


tmp = pd.DataFrame(train_iobp_df.groupby(['OtherPhysician','PotentialFraud'])['BeneID'].count()).reset_index()
tmp.columns = ['OtherPhysician', 'Fraud?', 'Num_of_cases']
tot_fraud_cases = tmp[tmp['Fraud?'] == 'Yes']['Num_of_cases'].sum()
tot_non_fraud_cases = tmp[tmp['Fraud?'] == 'No']['Num_of_cases'].sum()
tmp['Cases'] = tmp['Fraud?'].apply(lambda val: tot_non_fraud_cases if val == "No" else tot_fraud_cases)
tmp['Percentage'] = round(((tmp['Num_of_cases'] / tmp['Cases']) * 100),2)

tmp.head()


# In[267]:


tmp_only_frauds = tmp[tmp['Fraud?'] == 'Yes'].sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)


# In[268]:


print(tmp_only_frauds[['OtherPhysician','Num_of_cases','Percentage']].head(25), "\n")

with plt.style.context('seaborn'):
    plt.figure(figsize=(14,8))
    fig = sns.barplot(data=tmp_only_frauds.iloc[0:25], x="OtherPhysician", y="Num_of_cases", palette='Accent')
    # Using the "patches" function we will get the location of the rectangle bars from the graph.
    ## Then by using those location(width & height) values we will add the annotations
    for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/tot_fraud_cases,2))+"%"}', (x + width/2, y + height*1.025), ha='center', fontsize=13.5, rotation=90)
    
    # Providing the labels and title to the graph
    plt.xlabel("\nTop Fraudulent OtherPhysician")
    plt.xticks(rotation=90, fontsize=12)
    plt.ylabel("Number (or % share) of Cases\n")
    plt.minorticks_on()
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.title("Top-25 OtherPhysician with most number of fraudulent cases\n")
    plt.plot();


# In[269]:


#We can observe physician PHY412132 has the highest fraudalent cases. Other physicians PHY341578. PHY338032 & PHY337425 has the little less frauds than PHY412132 but higher when compared with the rest.


# In[270]:


## Top-25 ClmAdmitDiagnosisCode with maximum number of fraudulent cases


# In[271]:


tmp = pd.DataFrame(train_iobp_df.groupby(['ClmAdmitDiagnosisCode','PotentialFraud'])['BeneID'].count()).reset_index()
tmp.columns = ['ClmAdmitDiagnosisCode', 'Fraud?', 'Num_of_cases']
tot_fraud_cases = tmp[tmp['Fraud?'] == 'Yes']['Num_of_cases'].sum()
tot_non_fraud_cases = tmp[tmp['Fraud?'] == 'No']['Num_of_cases'].sum()
tmp['Cases'] = tmp['Fraud?'].apply(lambda val: tot_non_fraud_cases if val == "No" else tot_fraud_cases)
tmp['Percentage'] = round(((tmp['Num_of_cases'] / tmp['Cases']) * 100),2)

tmp.head()


# In[272]:


tmp_only_frauds = tmp[tmp['Fraud?'] == 'Yes'].sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)


# In[273]:


print(tmp_only_frauds[['ClmAdmitDiagnosisCode','Num_of_cases','Percentage']].head(25), "\n")

with plt.style.context('seaborn'):
    plt.figure(figsize=(14,8))
    fig = sns.barplot(data=tmp_only_frauds.iloc[0:25], x="ClmAdmitDiagnosisCode", y="Num_of_cases", palette='Accent')
    # Using the "patches" function we will get the location of the rectangle bars from the graph.
    ## Then by using those location(width & height) values we will add the annotations
    for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/tot_fraud_cases,2))+"%"}', (x + width/2, y + height*1.025), ha='center', fontsize=13.5, rotation=90)
    
    # Providing the labels and title to the graph
    plt.xlabel("\nTop Fraudulent ClmAdmitDiagnosisCode")
    plt.xticks(rotation=90, fontsize=12)
    plt.ylabel("Number (or % share) of Cases\n")
    plt.minorticks_on()
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.title("Top-25 ClmAdmitDiagnosisCode with most number of fraudulent cases\n")
    plt.plot();


# In[274]:


#We can observe 42731, V7612 & 78650 claims has the highest fraudalent cases


# In[275]:


## Checking if States have relationship with maximum number of fraudalent cases


# In[286]:


tmp = pd.DataFrame(train_iobp_df.groupby(['State','PotentialFraud'])['BeneID'].count()).reset_index()
tmp.columns = ['State', 'Fraud?', 'Num_of_cases']
tot_fraud_cases = tmp[tmp['Fraud?'] == 'Yes']['Num_of_cases'].sum()
tot_non_fraud_cases = tmp[tmp['Fraud?'] == 'No']['Num_of_cases'].sum()
tmp['Cases'] = tmp['Fraud?'].apply(lambda val: tot_non_fraud_cases if val == "No" else tot_fraud_cases)
tmp['Percentage'] = round(((tmp['Num_of_cases'] / tmp['Cases']) * 100),2)

tmp.head()


# In[287]:


tmp_only_frauds = tmp[tmp['Fraud?'] == 'Yes'].sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)


# In[288]:


print(tmp_only_frauds[['State','Num_of_cases','Percentage']].head(25), "\n")

with plt.style.context('seaborn'):
    plt.figure(figsize=(14,8))
    fig = sns.barplot(data=tmp_only_frauds.iloc[0:25], x="State", y="Num_of_cases", palette='Accent')
    # Using the "patches" function we will get the location of the rectangle bars from the graph.
    ## Then by using those location(width & height) values we will add the annotations
    for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/tot_fraud_cases,2))+"%"}', (x + width/2, y + height*1.025), ha='center', fontsize=13.5, rotation=90)
    
    # Providing the labels and title to the graph
    plt.xlabel("\nTop Fraudulent State Codes")
    plt.xticks(rotation=0, fontsize=12)
    plt.ylabel("Number (or % share) of Cases\n")
    plt.minorticks_on()
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.title("Top-25 State Codes with most number of fraudulent cases\n")
    plt.plot();


# In[ ]:


#Maximun frauds happend in states 5 - 14%, 10 - 8%, 33 - 8%


# In[289]:


## Checking Top 25 countries that have fraudalent cases


# In[290]:


tmp = pd.DataFrame(train_iobp_df.groupby(['County','PotentialFraud'])['BeneID'].count()).reset_index()
tmp.columns = ['County', 'Fraud?', 'Num_of_cases']
tot_fraud_cases = tmp[tmp['Fraud?'] == 'Yes']['Num_of_cases'].sum()
tot_non_fraud_cases = tmp[tmp['Fraud?'] == 'No']['Num_of_cases'].sum()
tmp['Cases'] = tmp['Fraud?'].apply(lambda val: tot_non_fraud_cases if val == "No" else tot_fraud_cases)
tmp['Percentage'] = round(((tmp['Num_of_cases'] / tmp['Cases']) * 100),2)

tmp.head()


# In[291]:


tmp_only_frauds = tmp[tmp['Fraud?'] == 'Yes'].sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)


# In[292]:


print(tmp_only_frauds[['County','Num_of_cases','Percentage']].head(25), "\n")

with plt.style.context('seaborn'):
    plt.figure(figsize=(15,10))
    fig = sns.barplot(data=tmp_only_frauds.iloc[0:25], x="County", y="Num_of_cases", palette='Accent')
    # Using the "patches" function we will get the location of the rectangle bars from the graph.
    ## Then by using those location(width & height) values we will add the annotations
    for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{str(round((height*100)/tot_fraud_cases,2))+"%"}', (x + width/2, y + height*1.025), ha='center', fontsize=13.5, rotation=90)
    
    # Providing the labels and title to the graph
    plt.xlabel("\nTop Fraudulent Country Codes")
    plt.xticks(rotation=90, fontsize=12)
    plt.ylabel("Number (or % share) of Cases\n")
    plt.minorticks_on()
    plt.grid(which='major', linestyle="--", color='lightgrey')
    plt.title("Top-25 Country Codes with most number of fraudulent cases\n")
    plt.plot();


# In[293]:


#Country codes 200 has 5% of fraudalent cases and 470 has 3.3% nearly to the highest number of cases


# In[ ]:




