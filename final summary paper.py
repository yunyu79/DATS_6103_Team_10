#%%
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.metrics import classification_report
import plotly.express as px
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
import statsmodels.api as sm

sns.set_style("whitegrid")
#%% [markdown]
## Introduction
#
# The Black Friday 2022 witnessed consumers spend a record $9.12 billion online, according to Adobe, which monitors sales on retailer websites. It is obvious that E-Commerce is one of the biggest and competitive industries in the world right now. Then how can one E-Commerce have competitive advantage over others?   
#  
# According to the survey regarding E-Commerce delivery, 66% of shoppers decided to buy products from one retailer over other because of the delivery services. The delivery service is playing a crucial role when it comes to the online shopping. Now online retailers are providing an estimated delivery date to customers since 83% of them anticipate a guaranteed arrival date. Even when the package is delayed, customers want the retailer to inform them about the delay and new estimated delivery date. The estimated delivery date and being delivered on-time are directly related to positive experiences from customers.   
#  
# Based on this, the goal of our project is to **predict the on-time delivery** with data from E-Commerce company. We expect our analysis have valuable business insights regarding delivery services to increase the customer satisfaction for E-Commerce retailers.   
#
# We have two **SMART questions** here. 
# 1. What features affect on-time delivery of products?
# 2. How can we predict on-time delivery with the higher accuracy for the E-Commerce company?
#
# This summary paper is organized as follows:   
# 1. Introduction 
# 2. Description of Data 
# 3. EDA 
# 4. Models 
# 5. Conclusion
# %%
data = pd.read_csv("dataset.csv")
data
#%% [markdown]
# We imported dataset as 'data'. The dataset is provided by E-commerce company who sells electronic products.
# The dataset contained 10999 observations of 12 variables.
# %%
data.info()
data.isna().sum() / len(data)
#%% [markdown]
# There are no N/A values.

# %%
# drop column that we are not using
data.drop(columns=['ID'], inplace=True)
data.info()
#%%[markdown]
# We dropped the ID column as it was irrelevant. 

# %%
# differences between on-time and delayed delivery
data.groupby("Reached.on.Time_Y.N").describe().T
#%%[markdown]
# We tried to see the differences on data by the on-time delivery.
# There are statistical differences in some variables by on-time delivery, including discount and weight. We will identify if this differences are statistically significant by visualization and hypothesis testings to answer the first SMART question which is "What features affect on-time delivery of products?". 
# %%
# boxplots for On-time delivery
integer_cols = data.select_dtypes(include = ['int64'])
(integer_cols.head())
intcols = integer_cols.columns
intcols21 = intcols.drop(["Reached.on.Time_Y.N"])
plt.figure(figsize = (16, 20))
sns.set_theme(style="ticks", palette="pastel")
nplot = 1
for i in range(len(intcols21)):
    if nplot <= len(intcols21):
        ax = plt.subplot(4, 2, nplot)
        sns.boxplot(x = intcols21[i], data = data, ax = ax)
        plt.title("Boxplots for On-time delivery by "f"{intcols21[i]}" , fontsize = 13)
        nplot += 1
plt.show()

# %%[markdown]
# From boxplots here, we can see the distribution of On-time delivery by each numberical variables. 

#%%
# Pairplots for integer columns (Initial EDA)
sns.pairplot(data = integer_cols, hue = "Reached.on.Time_Y.N")
#%%[markdown]
# From the pairplot here, we can see the relationships between variables at once. We will continue our exploratory data analysis.
#%%
# countplot for reached on time
sns.countplot(x ="Reached.on.Time_Y.N", data = data)
plt.title("Reached.on.Time_Y.N Counts")

# %%
# counts plots for every column
cols = ['Warehouse_block', 'Mode_of_Shipment', 'Customer_care_calls', 'Customer_rating', 'Prior_purchases', 'Product_importance', 'Gender']
plt.figure(figsize = (16, 20))
sns.set_theme(style="ticks", palette="pastel")
nplot = 1
for i in range(len(cols)):
    if nplot <= len(cols):
        ax = plt.subplot(4, 2, nplot)
        sns.countplot(x = cols[i], data = data, ax = ax, hue = "Reached.on.Time_Y.N")
        plt.title(f"\n{cols[i]} Counts", fontsize = 15)
        nplot += 1
plt.show()

#%%[markdown]
# From the first countplot, we can see there more packages that are delayed than reached on time. We also visualized count plots for each variables but separated by on-time delivery. We can see distributions for each variables here. The delayed packages are more than delivered on-time for every variables which makes sense with the previous countplot. 

#%%
# boxplots for On-time delivery 
intcols21 = intcols.drop(["Reached.on.Time_Y.N"])
plt.figure(figsize = (16, 20))
sns.set_theme(style="ticks", palette="pastel")
nplot = 1
for i in range(len(intcols21)):
    if nplot <= len(intcols21):
        ax = plt.subplot(4, 2, nplot)
        sns.boxplot(y = intcols21[i], x= "Reached.on.Time_Y.N", data = data, ax = ax)
        plt.title("Boxplots for On-time delivery by "f"{intcols21[i]}" , fontsize = 13)
        nplot += 1
plt.show()
#%%[markdown]
# From boxplots separated by on-time delivery, there are not huge differences between each component for every variables except for Discount and Weight. The packages that reached on time tend to have less discount amount and heavier than the packages that not delivered on time.

# %%
# Correlation Matrix
plt.figure(figsize = (15, 8))
sns.heatmap(data.corr(), annot = True, fmt = '0.2f', annot_kws = {'size' : 15},vmin=-1,center=0,vmax=1, linewidth = 5, linecolor = 'orange')
plt.show()

#%%[markdown]
# Based on the correlation matrix here, we decided to look into offered discount, weight, cost, and customer care calls.

# %%
# Histogram for Cost
sns.histplot(data, x = "Cost_of_the_Product", hue = "Reached.on.Time_Y.N", multiple = "stack")
plt.title("Histogram for Cost of the Product")
plt.show()

#%%[markdown]
# The histogram for cost which is stacked by on-time delivery has a bimodal distribution. For the products with lower cost, they tend to be not delivered on-time.

# %%
# Boxplots for Object columns and Cost 
object_cols = data.select_dtypes(include = ['object'])
object_cols.head()
obcol = object_cols.columns

plt.figure(figsize = (16, 20))
sns.set_theme(style="ticks", palette="pastel")
nplot = 1
for i in range(len(obcol)):
    if nplot <= len(obcol):
        ax = plt.subplot(4, 2, nplot)
        sns.boxplot(x = obcol[i], y= "Cost_of_the_Product", hue = "Reached.on.Time_Y.N", data = data, ax = ax)
        plt.title("Boxplot for Cost of the Product by "f"{obcol[i]}" , fontsize = 13)
        nplot += 1
plt.show()
# for boxplots there were no significant differences between x variables but there are differences by whether the product was reached on time
#%%[markdown]
# The boxplots between categorical variables with cost do not show much differences whether the order has been delivered on time or not. The cost of the products that delivered on time are slightly higher for every categorical variable.

# %%
# violins for customer calls and ratings 
intcols2 = intcols.drop(["Cost_of_the_Product", "Reached.on.Time_Y.N"])
intcols22 = intcols2[:2]
plt.figure(figsize = (16, 20))
sns.set_theme(style="ticks", palette="pastel")
nplot = 1
for i in range(len(intcols22)):
    if nplot <= len(intcols22):
        ax = plt.subplot(4, 2, nplot)
        sns.violinplot(x = intcols22[i], y= "Cost_of_the_Product", hue = "Reached.on.Time_Y.N", data = data, ax = ax, split = True)
        plt.title("Violinplots for Cost of the Product by "f"{intcols22[i]}" , fontsize = 13)
        nplot += 1
plt.show()
# the more customers call, the higher its cost
#%%[markdown]
# There are more customer calls when the cost of the product is higher which is reasonable, but there are no big differences between customer rating and cost of products.

#%%
# KDE plot for cost and weight
sns.displot(data=data, x="Cost_of_the_Product", y="Weight_in_gms", hue="Reached.on.Time_Y.N", kind = "kde", multiple="fill", clip=(0, None))
plt.title("KDE plot for Cost of the product by Weight")
plt.show()
#%%[markdown]
# For the lower cost and lighter products, they tend to be not delivered on time based on this plot. And for on-time delivered packages are usually heavy or expensive when they are light weighted.

# %%
intcols3 = intcols.drop(["Customer_care_calls", "Reached.on.Time_Y.N"])
plt.figure(figsize = (16, 20))
sns.set_theme(style="ticks", palette="pastel")
nplot = 1
for i in range(len(intcols3)):
    if nplot <= len(intcols3):
        ax = plt.subplot(4, 2, nplot)
        sns.boxplot(y = intcols3[i], x= "Customer_care_calls", hue = "Reached.on.Time_Y.N", data = data, ax = ax)
        plt.title("Boxplots for Customer Care Calls by "f"{intcols3[i]}" , fontsize = 13)
        nplot += 1
plt.show()
# boxenplots also works
#%%[markdown]
# From the boxplots here, we can see the distribution of customer care calls data by each variable that separated by on time delivery. We still can see the differences in Discount and Weight.

# %%
integer_columns = data.select_dtypes(include = ['int64'])
integer_columns.head()
reached_on_time_y_n = integer_columns['Reached.on.Time_Y.N'].value_counts().reset_index()
reached_on_time_y_n.columns = ['Reached.on.Time_Y.N', 'value_counts']
fig = px.pie(reached_on_time_y_n, names = 'Reached.on.Time_Y.N', values = 'value_counts',
             color_discrete_sequence = px.colors.sequential.Darkmint_r, width = 650, height = 400)
fig.update_traces(textinfo = 'percent+label')


# %%
# distribution plot about "Discount_offered"
plt.figure(figsize = (15, 7))
ax = sns.distplot(data['Discount_offered'], color = 'r')
plt.show()


#%%
# Categorical columns with Discount_offered
cols_cate_disc = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
plt.figure(figsize = (15, 15))
sns.set_theme(style="ticks", palette="pastel")
nplot = 1
for i in range(len(cols_cate_disc)):
    if nplot <= len(cols_cate_disc):
        ax = plt.subplot(2, 2, nplot)
        sns.boxplot(y='Discount_offered', x = cols_cate_disc[i], hue ='Reached.on.Time_Y.N', data = data, ax = ax)
        plt.title(f"\n{cols_cate_disc[i]} Counts", fontsize = 15)
        nplot += 1
plt.show()


# %%
px.box(data, x = 'Reached.on.Time_Y.N', y = 'Discount_offered',color = 'Reached.on.Time_Y.N')


# %%
plt.figure(figsize = (11, 7))
ax = sns.boxplot(x="Product_importance", y="Discount_offered", hue="Reached.on.Time_Y.N", palette={0: "y", 1: "b"}, data=data).set(title = "Do product_importance and discount affect delivery?")
plt.show()


# %%
# violinplot about discount + prior + reach or not
sns.violinplot(x="Prior_purchases", y="Discount_offered", hue="Reached.on.Time_Y.N", split=True, inner="quart",palette={0: "y", 1: "b"}, data=data).set(title = "Do prior_purchases and discount affect delivery?")
sns.despine(left=True)
plt.show()


# %%
# scatter plot about discount + weight + reach or not
plt.figure(figsize = (15, 7))
ax = sns.scatterplot(x='Discount_offered', y='Weight_in_gms', data=data, hue='Reached.on.Time_Y.N')
plt.show()
ax = sns.scatterplot(x='Discount_offered', y='Cost_of_the_Product', data=data, hue='Reached.on.Time_Y.N')
plt.show()


# %%
plt.figure(figsize = (15, 7))
sns.boxplot(x="Customer_care_calls", y="Discount_offered", hue="Reached.on.Time_Y.N", palette={0: "y", 1: "b"}, data=data).set(title = "Do calls and discount affect delivery?")
sns.despine(left=True)
plt.show()


#%%
#Timely delivered Statistics by Mean Weight

print(data.groupby(["Reached.on.Time_Y.N"])["Weight_in_gms"].mean(), "\n")

print(data.groupby(["Reached.on.Time_Y.N","Mode_of_Shipment"])["Weight_in_gms"].mean(), "\n")

print(data.groupby(["Reached.on.Time_Y.N","Warehouse_block"])["Weight_in_gms"].mean(), "\n")


#%%
# Delivery status of Weight by warehouse block
figure = plt.figure(figsize=(15,8))
sns.barplot(x="Reached.on.Time_Y.N",y="Weight_in_gms",
               data=data,color="indigo",hue="Warehouse_block", edgecolor = 'black')
plt.title("Warehouse_block")
plt.legend()
plt.show()

#%%

# Delivery status by weight and product importance and customer rating
figure = plt.figure(figsize=(15,8))
sns.lineplot(x="Customer_rating",y="Weight_in_gms",hue="Reached.on.Time_Y.N",style="Product_importance", palette="flare", data=data)
plt.show()

# %% WEIGHT KDE Histogram
sns.set_theme(style="ticks")
sns.histplot(data, x="Weight_in_gms", kde = True, hue = "Reached.on.Time_Y.N", multiple = "stack", palette="flare")


# %% Boxplot for weight against all vars

cols1 = ['Warehouse_block', 'Mode_of_Shipment', 'Customer_care_calls', 'Customer_rating', 'Prior_purchases', 'Product_importance', 'Gender']
plt.figure(figsize = (16, 28))
sns.set_theme(style="ticks", palette="pastel")
nplot = 1
for i in range(len(cols1)):
    if nplot <= len(cols1):
        ax = plt.subplot(4, 2, nplot)
        sns.boxplot(y='Weight_in_gms', x = cols1[i], hue ='Reached.on.Time_Y.N', data = data, ax = ax)
        plt.title(f"\n{cols1[i]} Counts", fontsize = 15)
        nplot += 1
plt.show()


# %%
#Scatter Plot for cost against weight over time
sns.scatterplot(data, x="Cost_of_the_Product", y="Weight_in_gms", hue="Reached.on.Time_Y.N", edgecolor = 'black')


# %%# customer rating 
sns.histplot(data, x = "Customer_rating", hue = "Reached.on.Time_Y.N", multiple = "dodge", bins=10)
plt.title("Histogram for Customer Rating")
plt.show()

#%%
# Statistical Test
# preparing dataset for anova
Xvar = data.copy()
Xvar['Gender'] = Xvar['Gender'].map({'M':0, 'F':1})
Xvar['Mode_of_Shipment'] = Xvar['Mode_of_Shipment'].map({'Flight': 1, 'Ship':2, 'Road':3})
Xvar['Product_importance'] = Xvar['Product_importance'].map({'high': 1, 'medium':2, 'low':3})
Xvar['Warehouse_block'] = Xvar['Warehouse_block'].map({'A': 1, 'B':2, 'C':3, 'D':4, 'F':5})
Xvar.rename(columns={'Reached.on.Time_Y.N': 'ReachedOnTime'}, inplace=True)
# ANOVA
aov_Customer_care_calls = ols('ReachedOnTime ~ Customer_care_calls', data = Xvar).fit()
aov_Customer_care_callsT = sm.stats.anova_lm(aov_Customer_care_calls, typ=2)
print(aov_Customer_care_callsT)
aov_Customer_rating = ols('ReachedOnTime ~ Customer_rating', data = Xvar).fit()
aov_Customer_ratingT = sm.stats.anova_lm(aov_Customer_rating, typ=2)
print(aov_Customer_ratingT)
aov_Cost_of_the_Product = ols('ReachedOnTime ~ Cost_of_the_Product', data = Xvar).fit()
aov_Cost_of_the_ProductT = sm.stats.anova_lm(aov_Cost_of_the_Product, typ=2)
print(aov_Cost_of_the_ProductT)
aov_Prior_purchases = ols('ReachedOnTime ~ Prior_purchases', data = Xvar).fit()
aov_Prior_purchasesT = sm.stats.anova_lm(aov_Prior_purchases, typ=2)
print(aov_Prior_purchasesT)
aov_Product_importance = ols('ReachedOnTime ~ Product_importance', data = Xvar).fit()
aov_Product_importanceT = sm.stats.anova_lm(aov_Product_importance, typ=2)
print(aov_Product_importanceT)
aov_Discount_offered = ols('ReachedOnTime ~ Discount_offered', data = Xvar).fit()
aov_Discount_offeredT = sm.stats.anova_lm(aov_Discount_offered, typ=2)
print(aov_Discount_offeredT)
aov_Weight_in_gms = ols('ReachedOnTime ~ Weight_in_gms', data = Xvar).fit()
aov_Weight_in_gmst = sm.stats.anova_lm(aov_Weight_in_gms, typ=2)
print(aov_Weight_in_gmst)
#%%[markdown]
# We decided to perform ANOVA tests on numerical variables to identify they have some relationship with on-time delivery. 
# H0: The means of numerical variables are same whether the order is delivered on time or not.
# H1: The means of numerical variables are NOT same whether the order is delivered on time or not.
# P-values from ANOVA tests with all the numerical variables except for Customer rating are small enough to reject the null hypothesis.
# Thus, Numerical variables except for Customer rating and on-time delivery are statistically related.

#CHI-Squared tests B/W Categorical Variables

#making continegncy tables 

contigency1 = pd.crosstab(index=data['Warehouse_block'], columns=data['Reached.on.Time_Y.N'], margins=True, margins_name="Total")
contigency2 = pd.crosstab(index=data['Mode_of_Shipment'], columns=data['Reached.on.Time_Y.N'], margins=True, margins_name="Total")
contigency3 = pd.crosstab(index=data['Gender'], columns=data['Reached.on.Time_Y.N'], margins=True, margins_name="Total")
contigency4 = pd.crosstab(index=data['Product_importance'], columns=data['Reached.on.Time_Y.N'], margins=True, margins_name="Total")

#run functions to get the values

stat1, p1, dof1, expected1 = chi2_contingency(contigency1)
stat2, p2, dof2, expected2 = chi2_contingency(contigency2)
stat3, p3, dof3, expected3 = chi2_contingency(contigency3)
stat4, p4, dof4, expected4 = chi2_contingency(contigency4)

#print values

print("p values for Warehouse Number, Mode of Shipment, Gender, Product importance (in that order)-", p1,p2,p3,p4 )

#None of them are significant

#%%
# KNN model
# 2. confusion matrix
# 3. accuracy score
# 4. classification report
# 5. cross_val_score
# 6. mean_squared_error
# 7. precision_recall_curve

<<<<<<< Updated upstream
=======
# We then performed a statistical tests as shown above where we analyze the relationship between 2 categorical variables.
# We find that the p value for all the variables is much higher than 0.05 which means that we fail to reject the null hypothesis and 
# we can say that there is no significant relationship between the variables.
# We can also say that the variables are independent of each other and they cannot be used to predict the outcome variable's condition. 
# in our case the condition would be whether a product gets delivered on time or not.
# By that, we can conclude that the variables - Mode of Shipment, Product Importance, Warehouse Block, Customer Rating are not to be used for the model building and prediction.
#%%[markdown]
# From all the EDA and statistical testings, we can answer the first SMART question.  
#
# **Q. What features affect on-time delivery of products?**   
#
# The **numerical variables** including customer care calls, weight/cost of products, offered discounts, and prior purchases **affect** the on-time delivery, whereas **customer rating does not**.   
#
# For all **categorical variables**, such as warehouse block, mode of shipment, gender, and importance of products they **do not affect** the on-time delivery.  
>>>>>>> Stashed changes

#%%
# Data Preprocessing
# Dealing with Warehouse_block
n = 'Warehouse_block'
data_copy = data.copy()
label_1 = pd.get_dummies(data_copy,prefix = n ,columns=[n],drop_first=False)
label_1.insert(loc=1, column=n, value=data[n].values)
label_1.drop([n],axis = 1,inplace = True)
label_1


#%%
# Dealing with Mode_of_Shipment
n = 'Mode_of_Shipment'
data_copy = data.copy()
label_2 = pd.get_dummies(label_1,prefix = n ,columns=[n],drop_first=False)
label_2.insert(loc=1, column=n, value=data[n].values)
label_2.drop([n],axis = 1,inplace = True)
label_2

#%%
# Dealing with Mode_of_Shipment
n = 'Product_importance'
data_copy = data.copy()
label_3 = pd.get_dummies(label_2,prefix = n ,columns=[n],drop_first=False)
label_3.insert(loc=5, column=n, value=data[n].values)
label_3.drop([n],axis = 1,inplace = True)
label_3

#%%
# Dealing with Gender
n = 'Gender'
data_copy = data.copy()
label_4 = pd.get_dummies(label_3,prefix = n ,columns=[n],drop_first=False)
label_4.insert(loc=5, column=n, value=data[n].values)
label_4.drop([n],axis = 1,inplace = True)
label_4

#%%
label_4 = label_4[['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
       'Prior_purchases', 'Discount_offered', 'Weight_in_gms',
       'Warehouse_block_A', 'Warehouse_block_B',
       'Warehouse_block_C', 'Warehouse_block_D', 'Warehouse_block_F',
       'Mode_of_Shipment_Flight', 'Mode_of_Shipment_Road',
       'Mode_of_Shipment_Ship', 'Product_importance_high',
       'Product_importance_low', 'Product_importance_medium', 'Gender_F',
       'Gender_M','Reached.on.Time_Y.N',]]
label_4

#%%

#Train and Test Set building

X = label_4.iloc[:, :-1].values
y = label_4.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# X_train = np.delete(X_train, np.s_[6:18],axis=1)
# X_test = np.delete(X_test, np.s_[6:18],axis=1)


#%%

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#%%

from sklearn.neighbors import KNeighborsClassifier

#ACC for KNN
fig,ax=plt.subplots(figsize=(10,10))
k_list=np.arange(1,11)
knn_acc={} # To store k and mse pairs
for i in k_list:
    knn=KNeighborsClassifier(n_neighbors = i)
    model_knn=knn.fit(X_train,y_train)
    y_pred=model_knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    knn_acc[i]=acc
#Plotting
ax.plot(knn_acc.keys(),knn_acc.values())
ax.set_xlabel('k-value', fontsize=20)
ax.set_ylabel('acc' ,fontsize=20)
ax.set_title('ACC in KNN' ,fontsize=28)


#%%
#MSE for KNN
fig,ax=plt.subplots(figsize=(10,10))
k_list=np.arange(1,11)
knn_mse={} # To store k and mse pairs
for i in k_list:
#Knn Model Creation
    knn=KNeighborsClassifier(n_neighbors = i)
    model_knn=knn.fit(X_train,y_train)
    y_pred=model_knn.predict(X_test)
#Storing MSE 
    mse=mean_squared_error(y_test,y_pred)
    knn_mse[i]=mse
#Plotting the results
ax.plot(knn_mse.keys(),knn_mse.values())
ax.set_xlabel('k-value', fontsize=20)
ax.set_ylabel('MSE' ,fontsize=20)
ax.set_title('MSE in KNN' ,fontsize=28)

print("We chose k-neibors = 4, since when n = 4, ACC is highest and MSE is lowest.")

#%%
classifier = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred = classifier.predict(X_test)
print(y_pred)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(f'The confusion matrix is {cm}')
print(f'Train accuracy is {knn.score(X_train, y_train)}')
print(f'Test accuracy is {knn.score(X_test, y_test)}')
print(f'Model accuracy is {accuracy_score(y_test, y_pred)}')

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10).mean()
print(f'The mean accuracy is {accuracies}')

#%%[markdown]
### Logistic Regression 
#%%
# correlation matrix for variables with Xvar data(diff encoding)
plt.figure(figsize = (18, 7))
sns.heatmap(Xvar.iloc[:,:-1].corr(), annot = True, fmt = '0.2f', annot_kws = {'size' : 15}, linewidth = 5, linecolor = 'orange')
plt.show()
#%%[markdown]
# We checked the correlations between variables. And it showed that no two variables have the high correlation. So first, we decided to keep all the variables for the logistic regression model. 
#%%
logitmodel = LogisticRegression()
logitmodel.fit(X_train, y_train)
print('Logit model accuracy (with the test set):', logitmodel.score(X_test, y_test))
print('Logit model accuracy (with the train set):', logitmodel.score(X_train, y_train))
print('Logit model Coefficient:', logitmodel.coef_)
print('Logit model Intercept:', logitmodel.intercept_)

# classification report
y_true, y_pred = y_test, logitmodel.predict(X_test)
cmlogit1 = confusion_matrix(y_test, y_pred)
print(cmlogit1)
print(classification_report(y_true, y_pred))
#%%[markdown]
# We built a logistic regression with all variables first. The model accuracy with the test set is 0.63, and with the train set is 0.64. Also we checked the confusion matrix and classification report.
#%%
# Applying k-Fold Cross Validation to logit1
accuraciesLogit1 = cross_val_score(estimator = logitmodel, X = X_train, y = y_train, cv = 10).mean()
print(f'The mean accuracy for a logistic model 1 is {accuraciesLogit1}')

# %%

# generate a no skill prediction 
ns_probs_rf = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = logitmodel.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs_rf)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs_rf)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
#%%[markdown]
# After the k-Fold Cross Validation, the mean accuracy for this logistic model is 0.64030044472024.
# THe ROC AUC score for this model is 0.718.
#%%
# Random forest

RandomForestModel = RandomForestClassifier(class_weight='balanced')
RandomForestModel.fit(X_train,y_train)
RandomForestPredict = RandomForestModel.predict(X_test)
print(RandomForestPredict)
print(classification_report(y_test,RandomForestPredict))

# Confusion Matrix, Train,test and Model Accuracy
nl = '\n'

tn, fp, fn, tp = confusion_matrix(y_test, RandomForestPredict).ravel()
print(f'The confusion matrix is - {nl}{confusion_matrix(y_test, RandomForestPredict)}')
print(f'Train accuracy is {RandomForestModel.score(X_train, y_train)}')
print(f'Test accuracy is {RandomForestModel.score(X_test, y_test)}')
print(f'Model accuracy is {accuracy_score(y_test, RandomForestPredict)}')
print('Specificity : ', (tn / (tn+fp)) )
print('Sensitivity : ', (tp / (tp+fn)) )



# Applying k-Fold Cross Validation
meanaccuraciesRf = cross_val_score(estimator = RandomForestModel, X = X_train, y = y_train, cv = 10).mean()
print("Mean Accuracy Score is - ", meanaccuraciesRf)

# ROC AUC Curve and Values

# generate a no skill prediction 
ns_probs_rf = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs_rf = RandomForestModel.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs_rf = lr_probs_rf[:, 1]
# calculate scores
ns_auc_rf = roc_auc_score(y_test, ns_probs_rf)
lr_auc_rf = roc_auc_score(y_test, lr_probs_rf)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc_rf))
print('Random Forest: ROC AUC=%.3f' % (lr_auc_rf))
# calculate roc curves
ns_fpr_rf, ns_tpr_rf, _ = roc_curve(y_test, ns_probs_rf)
lr_fpr_rf, lr_tpr_rf, _ = roc_curve(y_test, lr_probs_rf)
# plot the roc curve for the model
plt.plot(ns_fpr_rf, ns_tpr_rf, linestyle='--', label='No Skill')
plt.plot(lr_fpr_rf, lr_tpr_rf, marker='.', label='Random Forest')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

#%%
# data split
data2 = data.copy()
data2.drop(columns=['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender'], inplace=True)
X2 = data2.iloc[:, :-1].values
y2= data2.iloc[:, -1].values
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train2)
X_test2 = sc.transform(X_test2)


#%%

#%%
# Improved knn
from sklearn.neighbors import KNeighborsClassifier
#ACC for KNN
fig,ax=plt.subplots(figsize=(10,10))
k_list=np.arange(1,11)
knn_acc2={} # To store k and mse pairs
for i in k_list:
    knn=KNeighborsClassifier(n_neighbors = i)
    model_knn=knn.fit(X_train2,y_train2)
    y_pred2=model_knn.predict(X_test2)
    acc = accuracy_score(y_test2, y_pred2)
    knn_acc2[i]=acc
#Plotting
ax.plot(knn_acc2.keys(),knn_acc2.values())
ax.set_xlabel('k-value', fontsize=20)
ax.set_ylabel('acc' ,fontsize=20)
ax.set_title('ACC in KNN' ,fontsize=28)

#%%
#MSE for KNN
fig,ax=plt.subplots(figsize=(10,10))
k_list=np.arange(1,11)
knn_mse={} # To store k and mse pairs
for i in k_list:
#Knn Model Creation
    knn=KNeighborsClassifier(n_neighbors = i)
    model_knn=knn.fit(X_train2,y_train2)
    y_pred=model_knn.predict(X_test2)
#Storing MSE 
    mse=mean_squared_error(y_test2,y_pred)
    knn_mse[i]=mse
#Plotting the results
ax.plot(knn_mse.keys(),knn_mse.values())
ax.set_xlabel('k-value', fontsize=20)
ax.set_ylabel('MSE' ,fontsize=20)
ax.set_title('MSE in KNN' ,fontsize=28)

#%%
classifier = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
classifier.fit(X_train2, y_train2)

# Making the Confusion Matrix
y_pred2 = classifier.predict(X_test2)
print(y_pred2)
print(classification_report(y_test2, y_pred2))
cm = confusion_matrix(y_test2, y_pred2)
print(f'The confusion matrix is {cm}')
print(f'Train accuracy is {knn.score(X_train2, y_train2)}')
print(f'Test accuracy is {knn.score(X_test2, y_test2)}')
print(f'Model accuracy is {accuracy_score(y_test2, y_pred2)}')

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train2, y = y_train2, cv = 10).mean()
print(f'The mean accuracy is {accuracies}')

# %%
# Logitmodel with the second data
logitmodel2 = LogisticRegression()
logitmodel2.fit(X_train2, y_train2)
print('Logit model accuracy (with the test set):', logitmodel2.score(X_test2, y_test2))
print('Logit model accuracy (with the train set):', logitmodel2.score(X_train2, y_train2))
print('Logit model Coefficient:', logitmodel2.coef_)
print('Logit model Intercept:', logitmodel2.intercept_)

# classification report
y_true2, y_logitpred2 = y_test2, logitmodel2.predict(X_test2)
cmlogit2 = confusion_matrix(y_test2, y_pred2)
print(cmlogit2)
print(classification_report(y_true2, y_logitpred2))

#%%
# Applying k-Fold Cross Validation to logit2
accuraciesLogit2 = cross_val_score(estimator = logitmodel2, X = X_train2, y = y_train2, cv = 10).mean()
print(f'The mean accuracy for a logistic model 2 is {accuraciesLogit2}')

#%%
# generate a no skill prediction 
ns_probs_rf2 = [0 for _ in range(len(y_test2))]
# predict probabilities
lr_probs2 = logitmodel2.predict_proba(X_test2)
# keep probabilities for the positive outcome only
lr_probs2 = lr_probs2[:, 1]
# calculate scores
ns_auc2 = roc_auc_score(y_test2, ns_probs_rf2)
lr_auc2 = roc_auc_score(y_test2, lr_probs2)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc2))
print('Logistic: ROC AUC=%.3f' % (lr_auc2))
# calculate roc curves
ns_fpr2, ns_tpr2, _ = roc_curve(y_test2, ns_probs_rf2)
lr_fpr2, lr_tpr2, _ = roc_curve(y_test2, lr_probs2)
# plot the roc curve for the model
plt.plot(ns_fpr2, ns_tpr2, linestyle='--', label='No Skill')
plt.plot(lr_fpr2, lr_tpr2, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
#%%[markdown]
# We also tried o build the second logistic regression only with selected features. The accuracy with the test set has been improved by around 0.005 with selected features. However, after the k-Fold Cross Validation, the mean accuracy is 0.6383694022132589. Also the AUC ROC score is 0.716, which is lower than the first one. Thus we can conclude that with the mean accuracy and AUC ROC score, the first model is better between two Logistic Regression models. The logistic regression model with all features has 64% accuracy with AUC ROC score of 0.718.


#%%

# Random Forest after dropping insignificant features 

RandomForestModel2 = RandomForestClassifier(class_weight='balanced')
RandomForestModel2.fit(X_train2,y_train2)
RandomForestPredict2 = RandomForestModel2.predict(X_test2)
print(RandomForestPredict2)
print(classification_report(y_test2,RandomForestPredict2))

## Confusion Matrix, Train,test and Model Accuracy
nl='\n'
tn, fp, fn, tp = confusion_matrix(y_test2, RandomForestPredict2).ravel()

print(f'The confusion matrix is - {nl}{confusion_matrix(y_test2, RandomForestPredict2)}')
print(f'Train accuracy is {RandomForestModel2.score(X_train2, y_train2)}')
print(f'Test accuracy is {RandomForestModel2.score(X_test2, y_test2)}')
print(f'Model accuracy is {accuracy_score(y_test2, RandomForestPredict2)}')
print('Specificity : ', (tn / (tn+fp)) )
print('Sensitivity : ', (tp / (tp+fn)) )


# Applying k-Fold Cross Validation
meanaccuraciesRf2 = cross_val_score(estimator = RandomForestModel2, X = X_train2, y = y_train2, cv = 10).mean()
print("Mean Accuracy Score is - ", meanaccuraciesRf2)

# ROC AUC Curve and Values

# generate a no skill prediction 
ns_probs_rf2 = [0 for _ in range(len(y_test2))]
# predict probabilities
lr_probs_rf2 = RandomForestModel2.predict_proba(X_test2)
# keep probabilities for the positive outcome only
lr_probs_rf2 = lr_probs_rf2[:, 1]
# calculate scores
ns_auc_rf2 = roc_auc_score(y_test2, ns_probs_rf2)
lr_auc_rf2 = roc_auc_score(y_test2, lr_probs_rf2)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc_rf2))
print('Random Forest: ROC AUC=%.3f' % (lr_auc_rf2))
# calculate roc curves
ns_fpr_rf2, ns_tpr_rf2, _ = roc_curve(y_test2, ns_probs_rf2)
lr_fpr_rf2, lr_tpr_rf2, _ = roc_curve(y_test2, lr_probs_rf2)
# plot the roc curve for the model
plt.plot(ns_fpr_rf2, ns_tpr_rf2, linestyle='--', label='No Skill')
plt.plot(lr_fpr_rf2, lr_tpr_rf2, marker='.', label='Random Forest 2')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

#%%
#Finding the best hyperparameters for tuning the Random Forest

from sklearn.model_selection import RandomizedSearchCV
# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# number of features at every split
max_features = ['auto', 'sqrt']

# max depth
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)
# create random grid
random_grid = {
 'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth
 }
# Random search of parameters
rfc_randomSearch = RandomizedSearchCV(estimator = RandomForestModel2, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)
# Fit the model
rfc_randomSearch.fit(X_train, y_train)
# print results
print(rfc_randomSearch.best_params_)



# %%

RandomForestModel3 = RandomForestClassifier(n_estimators=200, max_depth=260, max_features='sqrt', class_weight='balanced')
RandomForestModel3.fit(X_train2,y_train2)
RandomForestPredict3 = RandomForestModel3.predict(X_test2)
print(RandomForestPredict3)
print(classification_report(y_test2,RandomForestPredict3))

## Confusion Matrix, Train,test and Model Accuracy
nl = '\n'

tn, fp, fn, tp = confusion_matrix(y_test2, RandomForestPredict3).ravel()
print(f'The confusion matrix is - {nl}{confusion_matrix(y_test2, RandomForestPredict3)}')
print(f'Train accuracy is {RandomForestModel3.score(X_train2, y_train2)}')
print(f'Test accuracy is {RandomForestModel3.score(X_test2, y_test2)}')
print(f'Model accuracy is {accuracy_score(y_test2, RandomForestPredict3)}')
print('Specificity : ', (tn / (tn+fp)) )
print('Sensitivity : ', (tp / (fn+tp)) )


# Applying k-Fold Cross Validation
meanaccuraciesRf3 = cross_val_score(estimator = RandomForestModel3, X = X_train2, y = y_train2, cv = 10, scoring = 'roc_auc').mean()
print("Mean Accuracy Score is - ", meanaccuraciesRf3)

# ROC AUC Curve and Values

# generate a no skill prediction 
ns_probs_rf3 = [0 for _ in range(len(y_test2))]
# predict probabilities
lr_probs_rf3 = RandomForestModel3.predict_proba(X_test2)
# keep probabilities for the positive outcome only
lr_probs_rf3 = lr_probs_rf3[:, 1]
# calculate scores
ns_auc_rf3 = roc_auc_score(y_test2, ns_probs_rf3)
lr_auc_rf3 = roc_auc_score(y_test2, lr_probs_rf3)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc_rf3))
print('Random Forest: ROC AUC=%.3f' % (lr_auc_rf3))
# calculate roc curves
ns_fpr_rf3, ns_tpr_rf3, _ = roc_curve(y_test2, ns_probs_rf3)
lr_fpr_rf3, lr_tpr_rf3, _ = roc_curve(y_test2, lr_probs_rf3)
# plot the roc curve for the model
plt.plot(ns_fpr_rf3, ns_tpr_rf3, linestyle='--', label='No Skill')
plt.plot(lr_fpr_rf3, lr_tpr_rf3, marker='.', label='Random Forest 3')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


# %%
#Variable Importance Plot

X_train2 = pd.DataFrame(X_train2, columns = ['Customer_care_calls',	'Customer_rating',	'Cost_of_the_Product',	'Prior_purchases',	'Discount_offered', 'Weight_in_gms'])
feature_scores = pd.Series(RandomForestModel3.feature_importances_, index = X_train2.columns).sort_values(ascending=True)
feature_scores.plot(kind='barh')

<<<<<<< Updated upstream
=======
# %% [markdown]
# We also plotted the variable importance plot to see which variables are the most important for the model to predict the target variable and it is clear that 
# the weight is the most  important variable followed by the discount offered and the cost of product other than that the lesser important variables are the 
# customer rating, prior purchases and the customer care calls.
#
## Conclusion  
# While our EDA shows that numeric variables are equally important in figuring out the trends and patterns in data, 
# we can see that the model is able to best predict the target variable using the Random Forest model with hyperparameter tuning 
# and numerical features(subset of the all the features). The Random Forest Model has the highest mean prediction accuracy of 73.75% 
# (66.18 % for singular running), Thus we can recommend the use of Random Forest model over the other models to predict the on-time delivery 
# for the electronics company.

# References - 
#
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html


>>>>>>> Stashed changes
# %%

