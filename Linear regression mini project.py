#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Data Set

# ## Table of Contents
# 
# 1. [Defning Problem Statement](#section1)<br>
# 2. [Importing Libraries](#section2)
#     - 2.1 [Loading Dataset](#section201)<br>    
# 3. [Basic Data Exploration](#section3)
#     - 3.1 [Feature Engineering](#section301)<br>
#     - 3.2 [Removing Unwanted columns](#section302)<br>   
# 4. [Visual Exploratory Data Analysis](#section4)<br>
#     - 4.1 [Rating Distribution](#section401)<br>
#     - 4.2 [Categorical Column Distribution](#section402)<br>
#     - 4.3 [Continuous Variable Distribution](#section403)<br>
#     - 4.4 [Outlier Removal](#section404)<br>
#     - 4.5 [Distribution of the Outlier Removal](#section405)<br>
#     - 4.6 [Missing Value Treatment](#section406)<br>
#     - 4.7 [Feature Selection](#section407)<br>
#     - 4.8 [Correlation Heatmap](#section408)<br>
#     - 4.9 [Relationship Exploration using Box Plot](#section409)<br> 
# 5. [Selection of Target and Feature Columns for Machine Learning](#section5)<br>
#     - 5.1 [Data Pre-processing for Machine Larning](#section501)<br>
#     - 5.2 [Converting the nominal variable to numeric using get_dummies](#section502)<br>
#     - 5.3 [Splitting Test and Train data](#section503)<br>
#     - 5.4 [Standardization/Normalization of data](#section504)<br>
#     6. [Linear Regression Model](#section6)<br>
#     - 6.1 [Decission Tree Regressor](#section601)<br>
#     - 6.2 [Cross Validation](#section602)<br>
# 7. [Conclusion](#section7)<br>
#     

# # 1. Defining the problem statement:

# To Create a Predictive model which can predictthe future rating of restaurant
# - Target variable: Rating
# - Predictors: location, menu,cost,votes,etc.
# 
#   -Rating=1 Worst
#   
#   -Rating=5 Best

# # 2. Importing Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### 2.1. Loading the Data Set

# In[3]:


ZomatoData=pd.read_csv('ZomatoData.csv',encoding='latin')


# In[4]:


print('Shape before deleting duplicate values:',ZomatoData.shape)
ZomatoData=ZomatoData.drop_duplicates()
print('Shape after deleting duplicate values',ZomatoData.shape)


# # 3. Basic Data Exploration

# In[5]:


ZomatoData.head()


# In[6]:


ZomatoData.info()


# In[7]:


ZomatoData.describe(include='all')


# In[8]:


ZomatoData.nunique()


# ### 3.1 Feature Engineering

# In[9]:


def cuisine_counter(inpStr):
    NumCuisines=len(str(inpStr).split(','))
    return(NumCuisines)


# In[10]:


ZomatoData['Cuisine_Count']=ZomatoData['Cuisines'].apply(cuisine_counter)
ZomatoData.head()


# ### 3.2 Removing unwanted Columns from the data

# In[11]:


UselessColumns=['Restaurant ID', 'Restaurant Name',
                'City','Address','Locality', 
                'Locality Verbose','Cuisines']
ZomatoData= ZomatoData.drop(UselessColumns,axis=1)
ZomatoData.head()


# # 4 Visual Exploratory Data Analysis

# ### 4.1 Rating distribution 

# In[12]:


ZomatoData['Rating'].plot(kind= 'hist', color= 'purple')


# ### 4.2 Categorical Column Distrubtion

# In[13]:


def PlotBarCharts(inpData,ColsToPlot):
    fig, subPlot= plt.subplots(nrows=1,ncols=len(ColsToPlot),figsize=(20,5))
    fig.suptitle('Bar Chart of:'+ str(ColsToPlot))
    
    for colName, plotNumber in zip(ColsToPlot, range(len(ColsToPlot))):
        inpData.groupby(colName).size().plot(kind='bar',ax=subPlot[plotNumber])


# In[14]:


PlotBarCharts(inpData=ZomatoData, ColsToPlot=[
    'Country Code', 'Currency', 'Has Table booking', 'Has Online delivery', 'Is delivering now',
    'Switch to order menu','Price range'])


# ###  Interpretation
# In this data, "Country Code", "Currency", "is delivering now" and "Switch to order menu" are too skewed. There is just one bar which is dominating and other categories have very less rows or there is just one value only. Such columns are not correlated with the target variable because there is no information to learn. The algorithms cannot find any rule like when the value is this then the target variable is that.
# 
# ### Selected Categorical Variables: 
# Only three categorical variables are selected for further analysis.
# 
# 'Has Table booking', 'Has Online delivery', 'Price range'

# ### 4.3 Continuous Variable Distribution

# In[15]:


ZomatoData.hist(['Longitude', 'Latitude', 
                 'Votes', 'Average Cost for two'], figsize=(18,10), color='g')


# ### Interpretation
# Histograms shows us the data distribution for a single continuous variable.
# 
# The X-axis shows the range of values and Y-axis represent the number of values in that range. For example, in the above histogram of "Votes", there are around 9000 rows in data that has a vote value between 0 to 1000.
# 
# The ideal outcome for histogram is a bell curve or slightly skewed bell curve. If there is too much skewness, then outlier treatment should be done and the column should be re-examined, if that also does not solve the problem then only reject the column.
# 
# ### Selected Continuous Variables:
# 
# Longitude : Selected. The distribution is good.
# 
# Latitude: Selected. The distribution is good.
# 
# Votes: Selected. Outliers seen beyond 300000, need to treat them.
# 
# Average Cost for two: Selected. Outliers seen beyond 4000, need to treat them.

# ### 4.4 Outlier Removal

# In[16]:


ZomatoData['Votes'][ZomatoData['Votes']<4000].sort_values(ascending=False)


# Above result shows the nearest logical value is 3986, hence, replacing any value above 4000 with it.

# In[17]:


ZomatoData['Votes'][ZomatoData['Votes']>4000] = 3986


# In[18]:


ZomatoData['Average Cost for two'][ZomatoData['Average Cost for two']<50000].sort_values(ascending=False)


# In[19]:


ZomatoData['Average Cost for two'][ZomatoData['Average Cost for two']>50000] =8000


# ### 4.5 Distribution After Outlier Removal

# In[20]:


ZomatoData.hist(['Votes', 'Average Cost for two'], figsize=(18,5))


# ### 4.6 Missing values treatment

# In[21]:


ZomatoData.isnull().sum()


# ### 4.7 Feature Selection
# In this case study the Target variable is Continuous, hence below two scenarios will be present
# 
# Continuous Target Variable Vs Continuous Predictor
# 
# Continuous Target Variable Vs Categorical Predictor

# In[22]:


ContinuousCols=['Longitude', 'Latitude', 'Votes', 'Average Cost for two']

for predictor in ContinuousCols:
    ZomatoData.plot.scatter(x=predictor,y='Rating',figsize=(10,5),title=predictor+"Vs" + 'Rating',color= 'orange')


# ### 4.8 Correlation Heatmap

# In[23]:


# Calculating correlation matrix
ContinuousCols=['Rating','Longitude', 'Latitude', 'Votes', 'Average Cost for two']

# Creating the correlation matrix
CorrelationData=ZomatoData[ContinuousCols].corr()
CorrelationData


# In[24]:


plt.figure(figsize=(12,9))
sns.heatmap(CorrelationData, vmax=.8, linewidth=0.01, square= True,linecolor='black', annot=True, cmap='rainbow')
plt.title('Correlation Heatmap')


# ### Final selected Continuous columns:
# 
# 'Votes', 'Average Cost for two'

# ### 4.9 Relationship exploration: Categorical Vs Continuous -- Box Plots
# When the target variable is Continuous and the predictor variable is Categorical we analyze the relation using Boxplots and measure the strength of relation using Anova test

# In[25]:


CategoricalColsList=['Has Table booking', 'Has Online delivery', 'Price range']

import matplotlib.pyplot as plt
fig, PlotCanvas=plt.subplots(nrows=1, ncols=len(CategoricalColsList), figsize=(18,5))

# Creating box plots for each continuous predictor against the Target Variable "Rating"
for PredictorCol , i in zip(CategoricalColsList, range(len(CategoricalColsList))):
    ZomatoData.boxplot(column='Rating', by=PredictorCol, figsize=(5,5), vert=True, ax=PlotCanvas[i],color='red' )


# ### interpretation
# In this data, all three categorical predictors looks correlated with the Target variable.

# # 5 Feature Selection for Machine Learning

# In[26]:


SelectedColumns=['Votes','Average Cost for two','Has Table booking',
                 'Has Online delivery','Price range']

DataForML = ZomatoData[SelectedColumns]
DataForML.head()


# In[27]:


DataForML.to_pickle('DataForML.pkl')


# ### 5.1 Data Pre-processing for Machine Learning

# In[28]:


DataForML['Has Table booking'].replace({'Yes':1,'No':0},inplace=True)

DataForML['Has Online delivery'].replace({'Yes':1,'No':0},inplace=True)


# ### 5.2 Converting the nominal variable to numeric using get_dummies()

# In[29]:


# Treating all the nominal variables at once using dummy variables
DataForML_Numeric=pd.get_dummies(DataForML)

# Adding Target Variable to the data
DataForML_Numeric['Rating']=ZomatoData['Rating']

# Printing sample rows
DataForML_Numeric.head()


# In[30]:


DataForML_Numeric.columns


# ### 5.3 Splitting the data into Training and Testing sample

# In[31]:


TargetVariable='Rating'
Predictors=['Votes', 'Average Cost for two', 'Has Table booking',
       'Has Online delivery', 'Price range']

X=DataForML_Numeric[Predictors].values
y=DataForML_Numeric[TargetVariable].values

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428)


# ### 5.4 Standardization/Normalization of data

# In[32]:


### Sandardization of data ###
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Choose between standardization and MinMAx normalization

PredictorScaler=MinMaxScaler()

# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)

# Generating the standardized values of X
X=PredictorScalerFit.transform(X)

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[33]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# # 6 Linear Regresion

# In[34]:


# Multiple Linear Regression
from sklearn.linear_model import LinearRegression
RegModel = LinearRegression()

# Printing all the parameters of Linear regression
print(RegModel)

# Creating the model on Training Data
LREG=RegModel.fit(X_train,y_train)
prediction=LREG.predict(X_test)

# Taking the standardized values to original scale


from sklearn import metrics
# Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, LREG.predict(X_train)))

###########################################################################
print('\n##### Model Validation and Accuracy Calculations ##########')

# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
print(TestingDataResults[[TargetVariable,'Predicted'+TargetVariable]].head())

# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(
  TestingDataResults['Rating']-TestingDataResults['PredictedRating']))/TestingDataResults['Rating'])

MAPE=np.mean(TestingDataResults['APE'])
MedianMAPE=np.median(TestingDataResults['APE'])

Accuracy =100 - MAPE
MedianAccuracy=100- MedianMAPE
print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
print('Median Accuracy on test data:', MedianAccuracy)


# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


# ### 6.1 Decission Tree Regressor

# In[35]:


# Decision Trees (Multiple if-else statements!)
from sklearn.tree import DecisionTreeRegressor
RegModel = DecisionTreeRegressor(max_depth=6,criterion='mse')
# Good Range of Max_depth = 2 to 20

# Printing all the parameters of Decision Tree
print(RegModel)

# Creating the model on Training Data
DT=RegModel.fit(X_train,y_train)
prediction=DT.predict(X_test)

from sklearn import metrics
# Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, DT.predict(X_train)))

# Plotting the feature importance for Top 10 most important columns
get_ipython().run_line_magic('matplotlib', 'inline')
feature_importances = pd.Series(DT.feature_importances_, index=Predictors)
feature_importances.nlargest(10).plot(kind='barh')

###########################################################################
print('\n##### Model Validation and Accuracy Calculations ##########')

# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
print(TestingDataResults[[TargetVariable,'Predicted'+TargetVariable]].head())

# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(
  TestingDataResults['Rating']-TestingDataResults['PredictedRating']))/TestingDataResults['Rating'])

MAPE=np.mean(TestingDataResults['APE'])
MedianMAPE=np.median(TestingDataResults['APE'])

Accuracy =100 - MAPE
MedianAccuracy=100- MedianMAPE
print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
print('Median Accuracy on test data:', MedianAccuracy)


# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


# In[36]:


# Separate Target Variable and Predictor Variables
TargetVariable='Rating'

# Selecting the final set of predictors for the deployment
# Based on the variable importance charts of multiple algorithms above
Predictors=['Votes', 'Average Cost for two', 'Price range']

X=DataForML_Numeric[Predictors].values
y=DataForML_Numeric[TargetVariable].values

### Sandardization of data ###
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Choose either standardization or Normalization
# On this data Min Max Normalization produced better results

# Choose between standardization and MinMAx normalization
#PredictorScaler=StandardScaler()
PredictorScaler=MinMaxScaler()

# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)

# Generating the standardized values of X
X=PredictorScalerFit.transform(X)

print(X.shape)
print(y.shape)


# In[37]:


# choose from different tunable hyper parameters
# Decision Trees (Multiple if-else statements!)
from sklearn.tree import DecisionTreeRegressor
RegModel = DecisionTreeRegressor(max_depth=6,criterion='mse')

# Training the model on 100% Data available
FinalDecisionTreeModel=RegModel.fit(X,y)


# ### 6.2 Cross Validation

# In[38]:


# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(FinalDecisionTreeModel, X , y, cv=10, scoring=custom_Scoring)
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


# In[39]:


import pickle
import os

# Saving the Python objects as serialized files can be done using pickle library
# Here let us save the Final ZomatoRatingModel
with open('FinalDecisionTreeModel.pkl', 'wb') as fileWriteStream:
    pickle.dump(FinalDecisionTreeModel, fileWriteStream)
    # Don't forget to close the filestream!
    fileWriteStream.close()
    
print('pickle file of Predictive Model is saved at Location:',os.getcwd())


# In[40]:


# This Function can be called from any from any front end tool/website
def FunctionPredictResult(InputData):
    import pandas as pd
    Num_Inputs=InputData.shape[0]
    
    # Making sure the input data has same columns as it was used for training the model
    # Also, if standardization/normalization was done, then same must be done for new input
    
    # Appending the new data with the Training data
    DataForML=pd.read_pickle('DataForML.pkl')
    InputData=InputData.append(DataForML)
    
    # Generating dummy variables for rest of the nominal variables
    InputData=pd.get_dummies(InputData)
            
    # Maintaining the same order of columns as it was during the model training
    Predictors=['Votes', 'Average Cost for two', 'Price range']
    
    # Generating the input values to the model
    X=InputData[Predictors].values[0:Num_Inputs]
    
    # Generating the standardized values of X since it was done while model training also
    X=PredictorScalerFit.transform(X)
    
    # Loading the Function from pickle file
    import pickle
    with open('FinalDecisionTreeModel.pkl', 'rb') as fileReadStream:
        PredictionModel=pickle.load(fileReadStream)
        # Don't forget to close the filestream!
        fileReadStream.close()
            
    # Genrating Predictions
    Prediction=PredictionModel.predict(X)
    PredictionResult=pd.DataFrame(Prediction, columns=['Prediction'])
    return(PredictionResult)


# In[41]:


# Calling the function for new sample data
NewSampleData=pd.DataFrame(
data=[[314,1100,3],
     [591,1200,4]],
columns=['Votes', 'Average Cost for two', 'Price range'])

print(NewSampleData)

# Calling the Function for prediction
FunctionPredictResult(InputData= NewSampleData)


# # 7 Conclusion

# ### Linear Regression 
# - R2 Value: 0.30651463134648727
# - Mean Accuracy on test data: 56.82382371482229
# - Median Accuracy on test data: 74.07407407407408
#     
# Accuracy values for 10-fold Cross Validation:
# - Final Average Accuracy of the model: 53.69
# 
# ### Decision Tree Regressor
# - R2 Value: 0.9120831300688722
# - Mean Accuracy on test data: 91.61407688871982
# - Median Accuracy on test data: 93.75
#     
# Accuracy values for 10-fold Cross Validation:
# - Final Average Accuracy of the model: 92.86
# 
# 
# 
# 

# In[ ]:




