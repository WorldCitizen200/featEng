#%%
import numpy as np
import pandas as pd
train = pd.read_csv('http://bit.ly/kaggletrain')
train.head()

# In[30]:
# # ONE HOT ENCODING --Maps female to 0 and male 1
sex = pd.get_dummies(train.Sex, prefix='Sex').iloc[:, 1:]
sex.head()

# In[56]:
# ONE HOT ENCODING --Maps Embarked_C to 1, Embarked_S to 1, and Embarked_Q to 1
embarked = pd.get_dummies(train.Embarked, prefix='Embarked', )
embarked.head()

# In[57]:
# Concatenates new features back on to orignal dataframe
train = pd.concat([train, sex, embarked], axis=1)
train.head()

# In[58]:
feature_cols = ['Sex_male', 'Embarked_C',  'Embarked_Q']

# selecting rows and columns from dataframe ##feature selection essentially
X = train.loc[:, feature_cols]  # we want every row within feature_cols
X.shape

#%%
# This is the response or TARGET vector
Y = train.Survived
Y.shape

# Now I want to do more feature engineering to add 2 new columns to X.shape so its shape is (891, 5) instead of (891, 3)
# feature one manipulates Name Series and feature two manipulates Age Series
# I need your help making using for loop to extract desired info

# this loop demos the two series I want to mainuplate
#%%
#
for index, row in train.iterrows():
    print(row["Name"], row["Age"])

# I'd say for name feature, we classify whether its Mr. Ms. Miss. Mrs. Master
# For age feature, we classify by every ten years of age with some sort of for loop

# GOOD LUCK! FUTURE BUSINESS TRANSACTIONS UP FOR GRABS FOR FIRST PPERSON TO HELP SOLVE MY PROBLEMS
