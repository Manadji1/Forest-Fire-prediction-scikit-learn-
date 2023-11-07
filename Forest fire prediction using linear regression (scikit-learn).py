#!/usr/bin/env python
# coding: utf-8

# # Forest Fire Prediction Using Linear Regression, scikit-learn

#    Variables 
#    1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9
#    2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9
#    3. month - month of the year: 'jan' to 'dec' 
#    4. day - day of the week: 'mon' to 'sun'
#    5. FFMC - FFMC index from the FWI system: 18.7 to 96.20
#    6. DMC - DMC index from the FWI system: 1.1 to 291.3 
#    7. DC - DC index from the FWI system: 7.9 to 860.6 
#    8. ISI - ISI index from the FWI system: 0.0 to 56.10
#    9. temp - temperature in Celsius degrees: 2.2 to 33.30
#    10. RH - relative humidity in %: 15.0 to 100
#    11. wind - wind speed in km/h: 0.40 to 9.40 
#    12. rain - outside rain in mm/m2 : 0.0 to 6.4 
#    13. area - the burned area of the forest (in ha): 0.00 to 1090.84 

# In[25]:


get_ipython().system('pip install install -U scikit-learn')


# In[26]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[27]:


data=pd.read_csv('forestfires.csv')


# In[28]:


data.info()


# In[29]:


data


# # Converting, month & day columns into integers 

# In[30]:


def ind (df,column,order):
    df=df.copy()
    df[column]=df[column].apply(lambda x: order.index(x))
    return df

    
    


# In[31]:


def pr(df):
    df=df.copy()
    df= ind (df,
                column='month', 
                order=[
                    'jan',
                    'feb',
                    'mar',
                    'apr',
                    'may',
                    'jun',
                    'jul',
                    'aug',
                    'sep',
                    'oct',
                    'nov',
                    'dec'] )
    df= ind (df,
                column='day', 
                order=[
                    'sun',
                    'mon',
                    'tue',
                    'wed',
                    'thu',
                    'fri',
                    'sat',
                         ] )
    return df
   

    
              


# In[32]:


processed_data=pr(data)
processed_data


# In[33]:


a=processed_data.drop(['area'], axis=1)

feature=['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain']


# # Evaluating R-square for each Independent variables

# In[34]:


for i in feature:
    lm=LinearRegression()
    c=processed_data[[i]]
    d=processed_data['area']
    lm.fit(c,d)
    score=lm.score(c,d)
    print('R-square of',i,'is:',score)

    
lm=LinearRegression()
c=processed_data[feature]
d=processed_data['area']
lm.fit(c,d)
score=lm.score(c,d)
#score=score.sort()
print('Initial R-square of model',i,'is:',score)

# value closer to one is ideal for the model



# In[35]:


processed_data.corr()['area'].sort_values()


# # Close look at the effect of ISI on the model  

# In[36]:


#from the r-suares above, rain and ISI have the lowest scores
#create individual linear regs for each id variable and apply the function to the corresponding column and retest model 
lm=LinearRegression()
x=processed_data[['ISI']]
y=processed_data['area']
lm.fit(x,y)
print('the y intercept is: ', lm.intercept_)
print('the slope is: ', lm.coef_ )
sns.regplot(x=x,y=y,data=processed_data)


# In[37]:


#applying linearfunction to the ISI values in the data
# y=0.11528731x+11.807208720691872
processed_data['ISI']=processed_data['ISI'].apply(lambda x : 0.11528731*x + 11.807208720691872)


# In[38]:


lm=LinearRegression()
c=processed_data[feature]
d=processed_data['area']
lm.fit(c,d)
score=lm.score(c,d)
print('R-square of modified ISI column',i,'is:',score)


# # Close look at the effect of rain on the model

# In[39]:


lm=LinearRegression()
x=processed_data[['rain']]
y=processed_data['area']
lm.fit(x,y)
print('the y intercept is: ', lm.intercept_)
print('the slope is: ', lm.coef_ )
sns.regplot(x=x,y=y,data=processed_data)


# In[40]:


processed_data['rain']=processed_data['rain'].apply(lambda x : -1.58424422*x + 12.881612253841306)
processed_data


# In[41]:


lm=LinearRegression()
c=processed_data[feature]
d=processed_data['area']
lm.fit(c,d)
score=lm.score(c,d)
print('R-square of modified rain on model is:',score)


# # Evaluating  DMC 

# In[42]:


# we'll next try to see how DMC can affect the r-square
lm=LinearRegression()
x=processed_data[['DMC']]
y=processed_data['area']
lm.fit(x,y)
print('the y intercept is: ', lm.intercept_)
print('the slope is: ', lm.coef_ )


# In[43]:


processed_data['DMC']=processed_data['DMC'].apply(lambda x : 0.07254905*x + 4.803608705221178)
processed_data


# In[44]:


lm=LinearRegression()
c=processed_data[feature]
d=processed_data['area']
lm.fit(c,d)
score=lm.score(c,d)
print('R-square of model of modified DMC is:',score)


# # Model Evaluation

# In[45]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# In[46]:


x_train, x_test, y_train, y_test = train_test_split(c, d, test_size=0.30, random_state=1)


# In[47]:


from sklearn.linear_model import Ridge


# In[48]:


r=Ridge(alpha=0.1)
r.fit(x_train,y_train)
print(r.score(x_test,y_test))

