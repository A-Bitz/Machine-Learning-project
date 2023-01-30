#!/usr/bin/env python
# coding: utf-8

# # Loan Prediction
# 
# <b> Problem
# 
# * <font color=blue> *A Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a data set.*</font>
# 
# 

# # Data
# <b>Variable Descriptions:
# 
# * <font color=orange>| Variable | Description | |------------------- |------------------------------------------------ | | Loan_ID | Unique Loan ID | | Gender | Male/ Female | | Married | Applicant married (Y/N) | | Dependents | Number of dependents | | Education | Applicant Education (Graduate/ Under Graduate) | | Self_Employed | Self employed (Y/N) | | ApplicantIncome | Applicant income | | CoapplicantIncome | Coapplicant income | | LoanAmount | Loan amount in thousands | | Loan_Amount_Term | Term of loan in months | | Credit_History | credit history meets guidelines | | Property_Area | Urban/ Semi Urban/ Rural | | Loan_Status | Loan approved (Y/N) |</font>

# In[1]:


# Importing Library
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


# Reading the training dataset in a dataframe using Pandas
df = pd.read_csv("D:/Loan-Approval-Prediction/trainingloan.csv")

# Reading the test dataset in a dataframe using Pandas 
test = pd.read_csv("D:/Loan-Approval-Prediction/test.csv")


# In[2]:


# First 10 Rows of training Dataset

df.head(10)


# In[3]:


# Store total number of observation in training dataset
df_length =len(df)

# Store total number of columns in testing data set
test_col = len(test.columns)


# ## Understanding the various features (columns) of the dataset.

# In[4]:


# Summary of numerical variables for training data set

df.describe()


# <font color=blue> *1. For the non-numerical values (e.g. Property_Area, Credit_History etc.), we can look at frequency distribution to understand whether they make sense or not.* </font>

# In[5]:


# Get the unique values and their frequency of variable Property_Area

df['Property_Area'].value_counts()


# <font color=Red>2. Understanding Distribution of Numerical Variables
# 
# * ApplicantIncome
# - LoanAmount
#     </font>

# In[6]:


# Box Plot for understanding the distributions and to observe the outliers.

get_ipython().run_line_magic('matplotlib', 'inline')

# Histogram of variable ApplicantIncome

df['ApplicantIncome'].hist()


# In[7]:


# Box Plot for variable ApplicantIncome of training data set

df.boxplot(column='ApplicantIncome')


# <font color=Violet>3.The above Box Plot confirms the presence of a lot of outliers/extreme values. This can be attributed to the income disparity in the society. </font>

# In[8]:


# Box Plot for variable ApplicantIncome by variable Education of training data set

df.boxplot(column='ApplicantIncome', by = 'Education')


# <font color=green>4.We can see that there is no substantial different between the mean income of graduate and non-graduates. But there are a higher number of graduates with very high incomes, which are appearing to be the outliers</font>

# In[9]:


# Histogram of variable LoanAmount

df['LoanAmount'].hist(bins=50)


# In[10]:


# Box Plot for variable LoanAmount of training data set

df.boxplot(column='LoanAmount')


# In[11]:


# Box Plot for variable LoanAmount by variable Gender of training data set

df.boxplot(column='LoanAmount', by = 'Gender')


# <font color=purple>5.LoanAmount has missing as well as extreme values, while ApplicantIncome has a few extreme values.</font>

# ## Understanding Distribution of Categorical Variables

# In[12]:


# Loan approval rates in absolute numbers
loan_approval = df['Loan_Status'].value_counts()['Y']
print(loan_approval)


# <b>* 422 number of loans were approved.

# In[13]:


# Credit History and Loan Status
pd.crosstab(df ['Credit_History'], df ['Loan_Status'], margins=True)


# In[14]:


#Function to output percentage row wise in a cross table
def percentageConvert(ser):
    return ser/float(ser[-1])

# # Loan approval rate for customers having Credit_History (1)
#df['Y'] = pd.crosstab(df ["Credit_History"], df ["Loan_Status"], margins=True).apply(percentageConvert, axis=1)
#loan_approval_with_Credit_1 = df['Y'][1]
#print(loan_approval_with_Credit_1*100)


# * 79.58 % of the applicants whose loans were approved have Credit_History equals to 1.

# In[15]:


df.head()


# In[16]:


# Replace missing value of Self_Employed with more frequent category
df['Self_Employed'].fillna('No',inplace=True)


# ## Outliers of LoanAmount and Applicant Income

# In[17]:


# Add both ApplicantIncome and CoapplicantIncome to TotalIncome
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

# Looking at the distribtion of TotalIncome
df['LoanAmount'].hist(bins=20)


# <b> * The extreme values are practically possible, i.e. some people might apply for high value loans due to specific needs. So instead of treating them as outliers, let’s try a log transformation to nullify their effect:

# In[18]:


# Perform log transformation of TotalIncome to make it closer to normal
df['LoanAmount_log'] = np.log(df['LoanAmount'])

# Looking at the distribtion of TotalIncome_log
df['LoanAmount_log'].hist(bins=20)


# ## Data Preparation for Model Building
# 
# * sklearn requires all inputs to be numeric, we should convert all our categorical variables into numeric by encoding the categories. Before that we will fill all the missing values in the dataset.

# In[19]:


# Impute missing values for Gender
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)

# Impute missing values for Married
df['Married'].fillna(df['Married'].mode()[0],inplace=True)

# Impute missing values for Dependents
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)

# Impute missing values for Credit_History
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

# Convert all non-numeric values to number
cat=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']

for var in cat:
    le = preprocessing.LabelEncoder()
    df[var]=le.fit_transform(df[var].astype('str'))
df.dtypes


# ## Generic Classification Function

# In[29]:


#Import models from scikit learn module:
from sklearn import metrics
from sklearn.model_selection import KFold

def classification_model(model, data, predictors, outcome):
    #Fit the model:
    model.fit(data[predictors],data[outcome])
  
    #Make predictions on training set:
    predictions = model.predict(data[predictors])
  
    #Print accuracy
    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

    #Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0])
    error = []
    for train, test in kf.split(data):
        # Filter training data
        train_predictors = (data[predictors].iloc[train,:])
    
        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]
    
        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)
    
        #Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
    print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    #Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors],data[outcome])



# ## Model Building

# In[30]:


#Combining both train and test dataset

#Create a flag for Train and Test Data set
df['Type']='Train' 
test['Type']='Test'
fullData = pd.concat([df,test], axis=0)

#Look at the available missing values in the dataset
fullData.isnull().sum()


# In[31]:


#Identify categorical and continuous variables
ID_col = ['Loan_ID']
target_col = ["Loan_Status"]
cat_cols = ['Credit_History','Dependents','Gender','Married','Education','Property_Area','Self_Employed']


# In[32]:


#Imputing Missing values with mean for continuous variable
fullData['LoanAmount'].fillna(fullData['LoanAmount'].mean(), inplace=True)
fullData['LoanAmount_log'].fillna(fullData['LoanAmount_log'].mean(), inplace=True)
fullData['Loan_Amount_Term'].fillna(fullData['Loan_Amount_Term'].mean(), inplace=True)
fullData['ApplicantIncome'].fillna(fullData['ApplicantIncome'].mean(), inplace=True)
fullData['CoapplicantIncome'].fillna(fullData['CoapplicantIncome'].mean(), inplace=True)

#Imputing Missing values with mode for categorical variables
fullData['Gender'].fillna(fullData['Gender'].mode()[0], inplace=True)
fullData['Married'].fillna(fullData['Married'].mode()[0], inplace=True)
fullData['Dependents'].fillna(fullData['Dependents'].mode()[0], inplace=True)
fullData['Loan_Amount_Term'].fillna(fullData['Loan_Amount_Term'].mode()[0], inplace=True)
fullData['Credit_History'].fillna(fullData['Credit_History'].mode()[0], inplace=True)


# In[33]:


#Create a new column as Total Income

fullData['TotalIncome']=fullData['ApplicantIncome'] + fullData['CoapplicantIncome']

fullData['TotalIncome_log'] = np.log(fullData['TotalIncome'])

#Histogram for Total Income
fullData['TotalIncome_log'].hist(bins=20) 


# ## Logistic Regression Model
# 1.The chances of getting a loan will be higher for:
# 
# * Applicants having a credit history (we observed this in exploration.)
# - Applicants with higher applicant and co-applicant incomes
# * Applicants with higher education level
# - Properties in urban areas with high growth perspectives
# 
# So let’s make our model with ‘Credit_History’, 'Education' & 'Gender'

# In[34]:


#create label encoders for categorical features
for var in cat_cols:
    number = LabelEncoder()
    fullData[var] = number.fit_transform(fullData[var].astype('str'))

train_modified=fullData[fullData['Type']=='Train']
test_modified=fullData[fullData['Type']=='Test']
train_modified["Loan_Status"] = number.fit_transform(train_modified["Loan_Status"].astype('str'))


# In[35]:


from sklearn.linear_model import LogisticRegression


predictors_Logistic=['Credit_History','Education','Gender']

x_train = train_modified[list(predictors_Logistic)].values
y_train = train_modified["Loan_Status"].values

x_test=test_modified[list(predictors_Logistic)].values


# In[36]:


# Create logistic regression object
model = LogisticRegression()

# Train the model using the training sets
model.fit(x_train, y_train)

#Predict Output
predicted= model.predict(x_test)

#Reverse encoding for predicted outcome
predicted = number.inverse_transform(predicted)

#Store it to test dataset
test_modified['Loan_Status']=predicted

outcome_var = 'Loan_Status'

classification_model(model, df,predictors_Logistic,outcome_var)

test_modified.to_csv("D:\Loan-Approval-Prediction\Logistic_Prediction.csv",columns=['Loan_ID','Loan_Status'])


# In[ ]:




