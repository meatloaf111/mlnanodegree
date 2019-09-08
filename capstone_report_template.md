# Machine Learning Engineer Nanodegree
## Capstone Project
Ken Adachi  
September xxst, 2019

## I. Definition

### Project Overview
In this project, I want to apply Machine Learning technique for the purpose of crime prevention.

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
In order to avoid to be involved in encountering crime, I want to develop a solution to predict if the crime occur in the present location. There are several points to consider:

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

In sum, I want to make the solution to accept the location and time and then predict if the crime will occur or not.

### Metrics
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_


## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
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



### Exploratory Visualization
- Location
Let's visualize the occurence of the crime by the location.

```python
rounding_factor = 4

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
```

![](C:\Users\keadachi\Downloads\locationheatmap.png)

- Occurence by Year
```python
crime_df['Year'] = [int(dte.split("/")[2]) for dte in crime_df['Date']]
sns.countplot(x='Year',data=crime_df)
```

![](C:\Users\keadachi\Downloads\incidentsperyear.png)


- Occurence by dayofweek
```python
sns.countplot(x='DayOfWeek',data=crime_df)
plt.title('Number of cases by dayofweek')
```
![](C:\Users\keadachi\Downloads\perdayofweek.png)

- Category
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

### Algorithms and Techniques

I want to build prediction model leveraging machine learning techniques.

In this case, I want to apply surpervised learning becuase the labeled data is available.

Based on sklearn algorithm cheat-sheet, I'm going to use classification methods to build the model.

https://scikit-learn.org/stable/tutorial/machine_learning_map/

I'm going to use ensemble model leveraging XGBoost or LightGBM.

In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
I will use very simple model such as logistic regression as a benchmark model.

In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
