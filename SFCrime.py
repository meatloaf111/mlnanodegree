#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
import xgboost 
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss, make_scorer
from sklearn.metrics import classification_report


# In[2]:


crime_df = pd.read_csv('Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv')


# In[3]:


crime_df.columns.values


# In[7]:


crime_df.info()


# In[11]:


type(crime_df['Time'])


# In[4]:


for col in ['Location','PdId','SF Find Neighborhoods', 'Current Police Districts',
       'Current Supervisor Districts', 'Analysis Neighborhoods',':@computed_region_yftq_j783', ':@computed_region_p5aj_wyqh', ':@computed_region_rxqg_mtj9', ':@computed_region_bh8s_q3mv', ':@computed_region_fyvs_ahh9',':@computed_region_9dfj_4gjx',':@computed_region_n4xg_c4py', ':@computed_region_4isq_27mq',':@computed_region_fcz8_est8', ':@computed_region_pigm_ib2e',':@computed_region_9jxd_iqea', ':@computed_region_6pnf_4xz7',':@computed_region_6ezc_tdp2', ':@computed_region_h4ep_8xdi',':@computed_region_nqbw_i6c3', ':@computed_region_2dwj_jsy4']:
    del crime_df[col]


# In[5]:


crime_df.head(3)


# In[6]:


print(crime_df['Category'])


# In[9]:


crime_df['Year'] = [int(dte.split("/")[2]) for dte in crime_df['Date']]


# In[15]:


crime_df['Hour'] = pd.to_datetime(crime_df['Time']).dt.hour


# # EDA

# In[8]:


rounding_factor = 4

# Create heatmap
from matplotlib.colors import LogNorm
x = np.round(crime_df['X'].head(10000),rounding_factor)
y = np.round(crime_df['Y'].head(10000),rounding_factor)
fig = plt.figure()
plt.suptitle('Reported Crime Heatmap')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
H, xedges, yedges, img = plt.hist2d(x, y, norm=LogNorm())
extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

plt.show()


# In[9]:






sns.countplot(x='Year',data=crime_df)
plt.title('Number of incidents per year')
plt.show()
#sns.plt.title('Number of cases by Year')

#crime_df['Week'] = crime_df['Date'].map(lambda x: x.week)
#crime_df['Hour'] = crime_df['Date'].map(lambda x: x.hour)
#crime_df['event']=1
#yearly_events = crime_df[['Year','event']].groupby(['Year']).count().reset_index()
#weekly_events_years = weekly_events.pivot(index='Week', columns='Year', values='event').fillna(method='ffill')
#%matplotlib inline
#ax = yearly_events.interpolate().plot(title='number of cases every 2 weeks', figsize=(10,6))
#plt.savefig('events_every_two_weeks.png')


# In[10]:


sns.countplot(x='DayOfWeek',data=crime_df)
plt.title('Number of cases by dayofweek')


# In[16]:


plt.figure(figsize=(20,10))
sns.countplot(x='Hour',data=crime_df)
plt.title('Number of cases by hour')


# # Data preprocess

# In[3]:


xy_scaler = preprocessing.StandardScaler() 

xy_scaler.fit(crime_df[["X","Y"]]) 

crime_df[["X","Y"]]=xy_scaler.transform(crime_df[["X","Y"]]) 


# In[4]:


le_crime = preprocessing.LabelEncoder()
crime = le_crime.fit_transform(crime_df.Category)
crime_df['Category'] = crime


# In[20]:


crime_classes = crime_df['Category'].unique()
crime_classes


# In[5]:


le_dow= preprocessing.LabelEncoder()
le_time = preprocessing.LabelEncoder()
crime_df['DayOfWeek'] = le_dow.fit_transform(crime_df.DayOfWeek)
crime_df['Time'] = le_time.fit_transform(crime_df.Time)


# In[10]:


crime_2015_2018 = crime_df[crime_df.Year > 2014]
#crime_2015_2017 = crime_2015_2018[crime_df.Year < 2018]


# In[9]:


crime_2015_2018.shape


# In[10]:


crime_2015_2018.head()


# In[11]:


training, testing = train_test_split(crime_2015_2018,test_size = 0.33, random_state=7)


# In[12]:


training.head()


# # Dataset Cleanup

# In[12]:


#training = training[['Category', 'DayOfWeek', 'Date', 'Time', 'X', 'Y']]
training = training[['Category', 'DayOfWeek',  'Time', 'X', 'Y']]
# Rename X,Y to Longitude, Latitude
training.columns = ['Category', 'DayOfWeek',  'Time', 'Longitude', 'Latitude']
training.head()


# In[13]:


testing = testing[['Category', 'DayOfWeek',  'Time', 'X', 'Y']]

# Rename X,Y to Longitude, Latitude
testing.columns = ['Category', 'DayOfWeek', 'Time', 'Longitude', 'Latitude']
testing.head()


# In[14]:


label = training['Category'].astype('category')

testlabel = testing['Category'].astype('category')

del training['Category']

del testing['Category']


# # LogisticRegression as benchmark model

# In[21]:


lr = LogisticRegression()
lr.fit(training,label)


# In[22]:


lrpredicted = lr.predict_proba(testing)
print(lrpredicted)


# In[23]:


log_loss(testlabel,lrpredicted)


# # XGB initial model

# In[24]:


xgb_model = XGBClassifier()
eval_set = [(training, label), (testing, testlabel)]
xgb_model.fit(training,label,eval_metric="mlogloss",eval_set=eval_set,verbose=True)


# In[36]:


predicted = xgb_model.predict_proba(testing)


# In[37]:


#print(predicted)


# In[25]:


log_loss(testlabel,predicted)


# In[53]:


results = xgb_model.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()


# In[42]:


print(xgb_model.feature_importances_)


# # Parameter tuning

# In[15]:


xgb_model = XGBClassifier()


# In[14]:


param_dist1 = {

               'learning_rate':[0.1,0.2,0.3]
}


# In[18]:


stratShuffleSplit = cross_validation.StratifiedShuffleSplit(label, train_size = 0.5, n_iter = 1)


# In[22]:


grid_search = GridSearchCV(xgb_model,

                            param_grid = param_dist1,

                            cv = stratShuffleSplit,

                            scoring={'neg_log_loss': make_scorer(log_loss, labels=crime_classes, greater_is_better=False,needs_proba=True)},
                  n_jobs=-1,
                  refit='neg_log_loss',

                            verbose=10)


# In[23]:


grid_search.fit(training, label)


# In[24]:


grid_search.best_params_


# In[16]:


param_dist2 = {

               'max_depth':[3,10],

               'min_child_weight':list(range(1,3,1))

}


# In[21]:


grid_search = GridSearchCV(xgb_model,

                            param_grid = param_dist2,

                            cv = stratShuffleSplit,

                            scoring={'neg_log_loss': make_scorer(log_loss, labels=crime_classes, greater_is_better=False,needs_proba=True)},
                  n_jobs=-1,
                  refit='neg_log_loss',

                            verbose=10)


# In[22]:


grid_search.fit(training, label)


# In[25]:


grid_search.best_params_, grid_search.best_score_


# In[22]:


param_dist3 = {'gamma':[0.5,1,2]}


# In[4]:


xgb_model = XGBClassifier()


# In[23]:


grid_search = GridSearchCV(xgb_model,

                            param_grid = param_dist3,

                            cv = stratShuffleSplit,

                            scoring={'neg_log_loss': make_scorer(log_loss, labels=crime_classes, greater_is_better=False,needs_proba=True)},
                  n_jobs=1,
                  refit='neg_log_loss',

                            verbose=10)


# In[24]:


grid_search.fit(training, label)


# In[29]:


grid_search.best_score_


# In[28]:


print('Best score: %0.3f' % grid_search.best_score_)
print('Best parameters set:')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(best_parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
    predictions = grid_search.predict(testing)
    print(classification_report(testlabel, predictions))


# # Final Model

# In[26]:


xgb_model = XGBClassifier(
                      learning_rate =  0.3,
                      max_depth = 10,

                      min_child_weight=2,

                      gamma = 1,
                      early_stopping_rounds=90
)


# In[27]:


eval_set = [(training, label), (testing, testlabel)]
xgb_model.fit(training,label,eval_metric="mlogloss",eval_set=eval_set,verbose=False)


# In[28]:


predicted = xgb_model.predict_proba(testing)


# In[29]:


log_loss(testlabel,predicted)


# In[30]:


results = xgb_model.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()


# In[31]:


print(xgb_model.feature_importances_)
plt.bar(range(len(xgb_model.feature_importances_)),xgb_model.feature_importances_)
plt.title('Feature Importances')
plt.show()


# In[ ]:




