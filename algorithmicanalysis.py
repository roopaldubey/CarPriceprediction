import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=pd.read_csv(r"C:\Users\ROOPAL\Downloads\cardata.csv\data.csv")
df.head()
df.count()

# All columns have 11914 values except 'Market Category' , 'Engine HP','Engine Fuel Type','Engine Cylinders'
df['Make'].nunique()
df['Model'].nunique()

# Some values have string characters as entry so to apply mathematical operation there is a need to convert it to numerical values
df['Market Category'].nunique()
df['Engine Fuel Type'].nunique()
df['Transmission Type'].nunique()

df['Transmission Type'].unique()

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
df['Transmission Type'] = labelencoder_X.fit_transform(df['Transmission Type'])
df['Transmission Type'].nunique()

# This converts the column values to numbers such as 0,1,2,3
df['Transmission Type'].unique()
le_name_mapping = dict(zip(labelencoder_X.classes_, 
                       labelencoder_X.transform(labelencoder_X.classes_))) 
mapping_dict={}
mapping_dict['Make']= le_name_mapping 
print(mapping_dict)

df.isnull().sum()


# It's important to convert the null values to some substantial information so that the prediction should be strengthened
df['Make'].unique()
df['Make'].max()
df['Make'].fillna("Volvo",inplace=True)


# filling null values with the most occuring value of the column
df['Make'] = labelencoder_X.fit_transform(df['Make'])
df['Make'].head()

le_name_mapping = dict(zip(labelencoder_X.classes_, 
                       labelencoder_X.transform(labelencoder_X.classes_))) 
 
mapping_dict={}
mapping_dict['Make']= le_name_mapping 
print(mapping_dict)

df['Make'].unique()

df['Engine HP'].fillna((df['Engine HP'].mean()), inplace=True)
df['Engine Fuel Type'].unique()
df['Engine Fuel Type'].unique()
df.replace(['premium unleaded (required)','premium unleaded (recommended)'],['premium_unleaded','premium_unleaded'],inplace=True)
df.replace(['regular unleaded'],['regular_unleaded'],inplace=True)
df.replace(['flex-fuel (unleaded/E85)'],['flex_fuel'],inplace=True)
df.replace([ 'flex-fuel (premium unleaded recommended/E85)','flex-fuel (premium unleaded required/E85)',
            'flex-fuel (unleaded/natural gas)'],['flex_fuel','flex_fuel','flex_fuel'],inplace=True)
df['Engine Fuel Type'].fillna("petrol",inplace=True)

df['Engine Fuel Type'] = labelencoder_X.fit_transform(df['Engine Fuel Type'])
df['Engine Fuel Type'].head()

le_name_mapping = dict(zip(labelencoder_X.classes_, labelencoder_X.transform(labelencoder_X.classes_))) 

mapping_dict={}
mapping_dict['Make']= le_name_mapping 
print(mapping_dict)

df['Engine Fuel Type'].isnull().sum()
df['Engine Fuel Type'].head()
df['Model'].max()
df['Model'].nunique()
df['Model'] = labelencoder_X.fit_transform(df['Model'])
df['Model'].head()
df['Model'].max()
le_name_mapping = dict(zip(labelencoder_X.classes_, labelencoder_X.transform(labelencoder_X.classes_))) 
mapping_dict={}
mapping_dict['Model']= le_name_mapping 
print(mapping_dict)

df['Engine Cylinders'].unique()
df['Engine Cylinders'].nunique()

df['Engine Cylinders'].fillna((df['Engine Cylinders'].mean()), inplace=True)
df['Engine Cylinders'].isnull().sum()
df.drop(labels='Market Category',axis=1,inplace=True)
df.drop(labels='Driven_Wheels',axis=1,inplace=True)
df.drop(labels='Number of Doors',axis=1,inplace=True)
df.head()
df['Vehicle Size'].unique()
df['Vehicle Size'] = labelencoder_X.fit_transform(df['Vehicle Size'])
le_name_mapping = dict(zip(labelencoder_X.classes_, 
                        labelencoder_X.transform(labelencoder_X.classes_))) 
mapping_dict={}
mapping_dict['Make']= le_name_mapping 
print(mapping_dict)


df['Vehicle Style'].unique()

df['Vehicle Style'] = labelencoder_X.fit_transform(df['Vehicle Style'])

le_name_mapping = dict(zip(labelencoder_X.classes_, 
                        labelencoder_X.transform(labelencoder_X.classes_))) 
mapping_dict={}
mapping_dict['Make']= le_name_mapping 
print(mapping_dict)

df['Year'] = 2019 -df['Year']
df.columns
X=df.iloc[:,:12]
X_val=df.iloc[:,:12]
y=df.iloc[:,12]
print(y.head())
print(X.head().dtypes)


print(X.columns)
print("----------")
X.dtypes
print(X.iloc[[1, 3, 5, 6], [1, 3]])
X.Model.head()
y.head()

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


# # Linear Regression
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
sns.pairplot(df, x_vars=['Make', 'Model', 'Year'], y_vars='MSRP', height=3, aspect=0.9)


# features that minimizes least square sum are taken as the features for linear regression

sns.pairplot(df, x_vars=['Engine Fuel Type', 'Engine HP',
       'Engine Cylinders'], y_vars='MSRP', height=3, aspect=0.9)

sns.pairplot(df, x_vars=['Transmission Type', 'Vehicle Size',
       'Vehicle Style'], y_vars='MSRP', height=3, aspect=0.9)

sns.pairplot(df, x_vars=['highway MPG', 'city mpg', 'Popularity'], y_vars='MSRP', height=3, aspect=0.9)


# In linear regression we do not require much of preprocessing 
# Simple linear regression is an approach for predicting a quantitative response using a single feature (or "predictor" or "input variable")


X_val.head()

max=0;
for column in X_val:
    print("-----------")
    print(column)

    feature_cols = [column]
    X_reg = df[feature_cols]
    y_reg = df.MSRP

    from sklearn.model_selection import train_test_split
    X_reg_train, X_reg_test, y_reg_train, y_reg_test =  train_test_split(X_reg,y_reg,test_size = 0.2, random_state= 0)

    # instantiate and fit
    lm = LinearRegression()
    lm.fit(X_reg_train, y_reg_train)
    print(" ")
    
    # print the coefficients
    print("The coefficients")
    print("intercept:")
    print(lm.intercept_)
    print("coefficient:")
    print(lm.coef_)
    y_reg_pred=lm.predict(X_reg_test)
   
    #Accuracy classification score.
    #In multilabel classification, this function computes subset accuracy: the set of labels predicted for a 
    #sample must exactly match the corresponding set of labels.
    #R2 model
    #The threshold for a good R-squared value depends widely on the domain
    #Therefore, it's most useful as a tool for comparing different models
    #R^2 (coefficient of determination) regression score function
    
    print(" ")
    print("The r2_score of the predicted value")
    print(metrics.r2_score(y_reg_test,y_reg_pred)*100)
    if metrics.r2_score(y_reg_test,y_reg_pred)*100 > max :
        max=metrics.r2_score(y_reg_test,y_reg_pred)*100
        max_column=column
        mse=metrics.mean_squared_error(y_reg_test,y_reg_pred)
        
        
    print("--------------------------------------------------------------------------------------------------------------------")

print(max_column)
print("r2_score")
print(max)
print("mse")
print(mse)


# The accuracy obtained by the linear regression model is 50% with df['Engine HP'] as the independent feature

X_reg = df[['Engine HP']]
y_reg = df[['MSRP']]

from sklearn.model_selection import train_test_split
X_reg_train, X_reg_test, y_reg_train, y_reg_test =  train_test_split(X_reg,y_reg,test_size = 0.2, random_state= 0)

# instantiate and fit
lm = LinearRegression()

lm.fit(X_reg_train, y_reg_train)
y_reg_pred=lm.predict(X_reg_test)

print("MSE", metrics.mean_squared_error(y_reg_test, y_reg_pred))

plt.scatter(X_reg_test,y_reg_pred)
plt.plot(X_reg_test, y_reg_pred, color='red')


# In Linear regression taking [Engine Hp] column as the feature we recieved the accuracy of the model as 50%.The accuracy increases if more number of features are used That is called Multiple Regression. 

X_reg=X_val[['Engine Fuel Type','Engine HP','Engine Cylinders']]
Y_reg=df.MSRP

X_reg = sc_X.fit_transform(X_reg)
from sklearn.model_selection import train_test_split
X_reg_train, X_reg_test, y_reg_train, y_reg_test =  train_test_split(X_reg,y_reg,test_size = 0.2, random_state= 0)

lm.fit(X_reg_train, y_reg_train)
y_reg_pred=lm.predict(X_reg_test)
print("r2_score")
print(metrics.r2_score(y_reg_test,y_reg_pred)*100)
print("mse")
print(metrics.mean_squared_error(y_reg_test,y_reg_pred))


# # PCA

# Accuracy is increased by 1% 
# This clearly means that the accuracy increases by taking more features into consideration,but as we have many features and most of them give the same accuracy which means that they can be reduced 

# Also the way to optimize the algorithm is to apply PCA and fasten the results

# PCA is effected by scale so there is a need to scale the features in the data before applying PCA.StandardScaler is used standardize the dataset’s features onto unit scale (mean = 0 and variance = 1) which is a requirement for the optimal performance of many machine learning algorithms

# The data was standardize prior in this code.
# The standardize data is:

# After dimensionality reduction, there usually isn’t a particular meaning assigned to each principal component. The new components are just the two main dimensions of variation.


X

from sklearn.decomposition import PCA

#There are 11 features let's reduce the dimension to 8

pca = PCA(n_components=8)

principalComponents = pca.fit_transform(X)

p_Df = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3',
                          'principal component 4','principal component 5','principal component 6','principal component 7',
                          'principal component 8'])

arr=pca.explained_variance_ratio_
arr

sum=0
for val in arr:
    val=(val*100)
    sum +=val;
    print(val)
sum


# These all components together give 93% of information rest of the information is in the other eigen vectors of the covariance matrix 
p_Df.count()
y.shape
from sklearn.model_selection import train_test_split
p_Df_train, p_Df_test, y_train, y_test =  train_test_split(X,y,test_size = 0.2, random_state= 0)

lm = LinearRegression()

lm.fit(p_Df_train , y_train)
y_pred=lm.predict(p_Df_test)
print("r2_score")
print(metrics.r2_score(y_test,y_pred))
print("mse")
print(metrics.mean_squared_error(y_test,y_pred))


# This means that its not only about increasing the features but its about increasing relevant features

# 
# Main limitation of Linear Regression is the assumption of linearity between the dependent variable and the independent 
# variables. In the real world, the data is rarely linearly separable. It assumes that there is a straight-line relationship 
# between the dependent and independent variables which is incorrect many times.

# Linear Regression is great tool to analyze the relationships among the variables but it isn’t recommended for most practical 
# applications because it over-simplifies real world problems by assuming linear relationship among the variables


# # Decision Tree
conda list
from sklearn.tree import DecisionTreeRegressor
import pickle

regressor = DecisionTreeRegressor(max_depth=6,random_state = 0) 
regressor1 = DecisionTreeRegressor(max_depth=6,min_samples_split=4,random_state = 0) 

X_val.shape

y.shape
y.isnull().sum()
from sklearn.model_selection import train_test_split
X_val_train,X_val_test,y_train,y_test=train_test_split(X_val,y,test_size=0.2)

X_val.columns

X_val_train['Model'].nunique()

regressor.fit(X_val_train,y_train)

X_val.columns

filename = "finalized_model.sav"

pickle.dump(regressor, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_val_test, y_test)
print(result)

regressor1.fit(p_Df_train,y_train)


# The above value is after implementation of PCA along with decision tree

y_pred = regressor.predict(X_val_test) 
print(y_pred)

regressor.max_depth

print("MSE", metrics.mean_squared_error(y_test, y_pred))
print("r2_score",metrics.r2_score(y_test, y_pred))

y_pred_pca = regressor1.predict(p_Df_test)

print("MSE", metrics.mean_squared_error(y_test, y_pred_pca))
print("r2_score",metrics.r2_score(y_test, y_pred_pca))


# This also means decision tree overfit

# R2  compares the fit of the chosen model with that of a horizontal straight line (the null hypothesis).
# If the chosen model fits worse than a horizontal line, then R2 is negative
# R2  is negative only when the chosen model does not follow the trend of the data, so fits worse than a horizontal line.

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(regressor1, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# Greedy algorithms cannot guarantee to return the globally optimal decision tree. This can be mitigated by training multiple
# trees, where the features and samples are randomly sampled with replacement.

# Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the data set prior
# to fitting with the decision tree.

# # SVM

S=df.drop(['MSRP'],axis=1)
t=df[['MSRP']]

SC=sc_X.fit_transform(S)
T=sc_X.fit_transform(t)

SC_train,SC_test,T_train,T_test=train_test_split(SC,T)

from sklearn.svm import SVR
regressor_svm = SVR(kernel = 'rbf',gamma='scale')
regressor_svm.fit(SC_train, T_train)

y_pred_svm=regressor_svm.predict(SC_test)

metrics.r2_score(T_test,y_pred_svm)

metrics.mean_squared_error(T_test,y_pred_svm)

principalComponents = pca.fit_transform(SC)

p_Df = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3',
                          'principal component 4','principal component 5','principal component 6','principal component 7',
                          'principal component 8'])
ndf=p_Df

arr=pca.explained_variance_ratio_
arr

sum=0
for val in arr:
    val=(val*100)
    sum +=val;
    print(val)
sum

ndf_train,ndf_test,T_train,T_test=train_test_split(ndf,T)

regressor_svm.fit(ndf_train, T_train)

y_pred_svm=regressor_svm.predict(ndf_test)

print(metrics.r2_score(T_test,y_pred_svm))
metrics.mean_squared_error(T_test,y_pred_svm)


# The value of mean square error is very low and better than Decision tree this clearly states that Decision Tree was under 
# overfitting but still the accuracy was best therefore we should for decision tree by avoiding its shortcoming

# # Random Forest

R=df.drop(['MSRP'],axis=1)
t=df[['MSRP']]

R.columns

RC=sc_X.fit_transform(R)
T=sc_X.fit_transform(t)


R_train,R_test,t_train,t_test=train_test_split(R,t)

from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(R_train,t_train);

filename = "finalizedrandom_model.sav"

pickle.dump(rf, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(R_test, t_test)
#print(result)



y_pred=rf.predict(R_test)

metrics.mean_squared_error(t_test,y_pred)

metrics.r2_score(t_test,y_pred)

principalComponents = pca.fit_transform(RC)

p_Df = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3',
                          'principal component 4','principal component 5','principal component 6','principal component 7',
                          'principal component 8'])
rdf=p_Df

rdf_train,rdf_test,T_train,T_test=train_test_split(rdf,T)

rf.fit(rdf_train,T_train);

y_pred_rf=rf.predict(rdf_test)

metrics.mean_squared_error(T_test,y_pred_rf)

metrics.r2_score(T_test,y_pred_rf)


# This clearly proves that Random forest is best amongst all as it has the least mean square error and the highest score and
# also overcomes the limitations or the problems by decision tree
