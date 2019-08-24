# Machine Learning Engineer Nanodegree
## Capstone Proposal
Ken Adachi  
July 2nd, 2019

Updated:Aug 24th, 2019

## Proposal

### Domain Background

I want to apply Machine Learning technique for the purpose of crime prevention.

There are some academic papers or articles on this domain such as:

Learning, Predicting and Planning against Crime:Demonstration Based on Real Urban Crime Data
https://pdfs.semanticscholar.org/cacd/e031e470af4fe835bf50f14eb4c265e0f2a6.pdf

USING MACHINE LEARNING ALGORITHMS TO ANALYZE CRIME DATA
https://www.researchgate.net/publication/275220711_Using_Machine_Learning_Algorithms_to_Analyze_Crime_Data

AI for Crime Prevention and Detection – 5 Current Applications
https://emerj.com/ai-sector-overviews/ai-crime-prevention-5-current-applications/

I want to take Crime Opportunity Theory as a basis for this project.This theory suggests that the occurence of a crime depens not only on the presense of the motivated offender but also on the conditions of the environment in which that offender is situated.

Reference:
Community Safety Maps for Children in Japan: An Analysis from a Situational Crime Prevention Perspective
https://link.springer.com/article/10.1007/s11417-011-9113-z


I except that I can find a pattern in where actual crime happened.
So I want to analyze the histrocial crime incident data with the information on where it occured.

This might lead to prevent crime to happen in my neighborhood and proctec children in my community.

### Problem Statement

In order to avoid to be involved in encountering crime, I want to predict if the crime occur in the present location. There are several points to consider:

- Location

Crime Opportunity theory suggests that the "view" of the location is important for the ones who try to commit a crime to decide to do so.For example, if there is a street with many tall trees which makes it hard for everyone to be seen, there is a higher chance of the crime to be occured.

So if I can specify the "location" as a street level granurality , then that would precise.
Or if I can use street view image of the location leveraging map data such as google street view, then the result would be interesting.

However, given limitted resources and time, I'm thinking of specifying location as neigborhood in a way such as "the crime actually happend in within a few miles radius from the current point".

- Types of Crime

I would not classify what types of crimes occured or will occure.
The main purpose of this project is to identify the location that makes criminals to think the place is a good opportunity for them to commit a crime.
So, I just want to focus on analyizing the crime occured in that place or not.

- Timing of the crime

It is expected that the crime might occur on specific timing.
For example, crime targeted for the children will occur more when children got home from school.



### Datasets and Inputs

I'm going to use "Police Department Incident Reports: Historical 2003 to May 2018" data set provided by San Francisco City Government.

https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry

This dataset includes police incident reports filed by officers and by individuals through self-service online reporting for non-emergency cases through 2003 to 2018.The dataset has attributes such as when the incident reports filed (Date, time) and detail location of the incidents(latitude, longitude).

```python
crime_df = pd.read_csv('Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv')
```

```python
crime_df.columns.values
```




    array(['IncidntNum', 'Category', 'Descript', 'DayOfWeek', 'Date', 'Time',
           'PdDistrict', 'Resolution', 'Address', 'X', 'Y', 'Location', 'PdId',
           'SF Find Neighborhoods', 'Current Police Districts',
           'Current Supervisor Districts', 'Analysis Neighborhoods',
           ':@computed_region_yftq_j783', ':@computed_region_p5aj_wyqh',
           ':@computed_region_rxqg_mtj9', ':@computed_region_bh8s_q3mv',
           ':@computed_region_fyvs_ahh9', ':@computed_region_9dfj_4gjx',
           ':@computed_region_n4xg_c4py', ':@computed_region_4isq_27mq',
           ':@computed_region_fcz8_est8', ':@computed_region_pigm_ib2e',
           ':@computed_region_9jxd_iqea', ':@computed_region_6pnf_4xz7',
           ':@computed_region_6ezc_tdp2', ':@computed_region_h4ep_8xdi',
           ':@computed_region_nqbw_i6c3', ':@computed_region_2dwj_jsy4'], dtype=object)

```python
crime_df.shape
```




    (2215024, 33)

There are 33 features in this dataset with about 2 million data rows.
There is no description on the original dataset about the feature ranging from
'SF Find Neighborhoods' to ':@computed_region_2dwj_jsy4'. So I will omit those features and focus on following featuers:

'IncidntNum': Unique key value on each incident.

'Category':VEHICLE THEFT,NON-CRIMINAL etc.

'Descript':STOLEN MOTORCYCLE,PAROLE VIOLATION etc

'DayOfWeek':'Monday', 'Tuesday'...

'Date':DD/MM/YYYY

'Time':HH:mm

'PdDistrict':SOUTHERN,MISSION etc

'Resolution':ARREST,BOOKED etc

'Address':Street name of crime such as Block of TEHAMA ST

'X':Longitude

'Y':Latitude

'Location':Concat of X and Y

'PdId':Unique Identifier for use in update and insert operations

Category fields have following instances.
There looks skew in specific classes so this is not closely balanced.

```python
crime_df['Category'].unique()
```
```
array(['VEHICLE THEFT', 'NON-CRIMINAL', 'OTHER OFFENSES', 'ROBBERY','DRUG/NARCOTIC', 'LIQUOR LAWS', 'WARRANTS', 'PROSTITUTION','ASSAULT', 'LARCENY/THEFT', 'VANDALISM', 'STOLEN PROPERTY','KIDNAPPING', 'BURGLARY', 'SECONDARY CODES', 'DRUNKENNESS','SUSPICIOUS OCC', 'DRIVING UNDER THE INFLUENCE', 'WEAPON LAWS','FRAUD', 'TRESPASS', 'FAMILY OFFENSES', 'MISSING PERSON','SEX OFFENSES, FORCIBLE', 'RUNAWAY', 'DISORDERLY CONDUCT',
'FORGERY/COUNTERFEITING', 'GAMBLING', 'BRIBERY', 'EXTORTION',
'ARSON', 'EMBEZZLEMENT', 'PORNOGRAPHY/OBSCENE MAT', 'SUICIDE',
'SEX OFFENSES, NON FORCIBLE', 'BAD CHECKS', 'LOITERING',
'RECOVERED VEHICLE', 'TREA'], dtype=object)
```

```python
crime_df['Category'].value_counts()
```

```
LARCENY/THEFT                  480448
OTHER OFFENSES                 309358
NON-CRIMINAL                   238323
ASSAULT                        194694
VEHICLE THEFT                  126602
DRUG/NARCOTIC                  119628
VANDALISM                      116059
WARRANTS                       101379
BURGLARY                        91543
SUSPICIOUS OCC                  80444
MISSING PERSON                  64961
ROBBERY                         55867
FRAUD                           41542
SECONDARY CODES                 25831
FORGERY/COUNTERFEITING          23050
WEAPON LAWS                     22234
TRESPASS                        19449
PROSTITUTION                    16701
STOLEN PROPERTY                 11891
SEX OFFENSES, FORCIBLE          11742
DISORDERLY CONDUCT              10040
DRUNKENNESS                      9826
RECOVERED VEHICLE                8716
DRIVING UNDER THE INFLUENCE      5672
KIDNAPPING                       5346
RUNAWAY                          4440
LIQUOR LAWS                      4083
ARSON                            3931
EMBEZZLEMENT                     2988
LOITERING                        2430
SUICIDE                          1292
FAMILY OFFENSES                  1183
BAD CHECKS                        925
BRIBERY                           813
EXTORTION                         741
SEX OFFENSES, NON FORCIBLE        431
GAMBLING                          348
PORNOGRAPHY/OBSCENE MAT            59
TREA                               14
```

### Solution Statement

I want to build prediction model leveraging machine learning techniques.

In this case, I want to apply surpervised learning becuase the labeled data is available.

Based on sklearn algorithm cheat-sheet, I'm going to use classification methods to build the model.

https://scikit-learn.org/stable/tutorial/machine_learning_map/


### Benchmark Model

I will use very simple model such as logistic regression as a benchmark model.

### Evaluation Metrics

This is a classification problem.However since the Category class is not closely balanced, I would not use F measures as metrics.
I'm going to use log loss.

### Project Design

Summary of the project design is as follows:

- Problem defnition
  I want to predict the possibility if I'm going to encounter with crime in the current location.This will lead to prevent to encounter with crime and save the peopole of the local community.
  
- Choose the algorithm
  I want to use surpervised learning methods , and classification algorithm to build a prediction model.

  I'm going to use ensemble model leveraging XGBoost or LightGBM.

- Analyze the data
  Before going into the detail of machine learning, I will conduct data analytics or visulization to find when and where the crime occured.

- Pre Process
  Data set has a text field of 'Category'.This is a non-numerical feature, so I'm going to encode this to numerical value using pandas get_dummies() or LabelEncoder.

- Train and Tune
  Divide the data set into training data set and test dataset.
  Train the model using training data set and then evaluate it with test dataset.
  Tune hyper parameter to improve the score of the model.


