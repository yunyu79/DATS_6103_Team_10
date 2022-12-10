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
from sklearn.metrics import  confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error
from scipy.stats import chi2_contingency

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

sns.set_style("whitegrid")
# %%
data = pd.read_csv("dataset.csv")
data
# %%
data.info()
data.isna().sum() / len(data)
# %%
# drop column that we are not using
data.drop(columns=['ID'], inplace=True)
data.info()

# %%
# differences between on-time and delayed delivery
data.groupby("Reached.on.Time_Y.N").describe().T
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

#%%
# Pairplots for integer columns (Initial EDA)
sns.pairplot(data = integer_cols, hue = "Reached.on.Time_Y.N")
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


#%%
# countplot for reached on time
sns.countplot(x ="Reached.on.Time_Y.N", data = data)
plt.title("Reached.on.Time_Y.N Counts")

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


# %%
# Correlation Matrix
plt.figure(figsize = (15, 8))
sns.heatmap(data.corr(), annot = True, fmt = '0.2f', annot_kws = {'size' : 15},vmin=-1,center=0,vmax=1, linewidth = 5, linecolor = 'orange')
plt.show()


# %%
# Histogram for Cost
sns.histplot(data, x = "Cost_of_the_Product", hue = "Reached.on.Time_Y.N", multiple = "stack")
plt.title("Histogram for Cost of the Product")
plt.show()


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


#%%
# KDE plot for cost and weight
sns.displot(data=data, x="Cost_of_the_Product", y="Weight_in_gms", hue="Reached.on.Time_Y.N", kind = "kde", multiple="fill", clip=(0, None))
plt.title("KDE plot for Cost of the product by Weight")
plt.show()


# %%
# violins for customer calls 
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


# %%
# Histogram for Customer Calls
sns.histplot(data, x = "Customer_care_calls", hue = "Reached.on.Time_Y.N", multiple = "dodge", bins=10)
plt.title("Histogram for Customer Calls")
plt.show()


# %%
# Violinplots for Customer Calls by Ontime
plt.figure(figsize = (16, 20))
sns.set_theme(style="ticks", palette="pastel")
nplot = 1
for i in range(len(obcol)):
    if nplot <= len(obcol):
        ax = plt.subplot(4, 2, nplot)
        sns.violinplot(x = obcol[i], y= "Customer_care_calls", hue = "Reached.on.Time_Y.N", data = data, ax = ax, split = True)
        plt.title("Violinplots for Customer Calls by "f"{obcol[i]}" , fontsize = 13)
        nplot += 1
plt.show()


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


# %% WEIGHT KDE Histogram
sns.set_theme(style="ticks", palette="pastel")
sns.histplot(data, x="Weight_in_gms", kde = True, hue = "Reached.on.Time_Y.N", multiple = "stack")


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
sns.scatterplot(data, x="Cost_of_the_Product", y="Weight_in_gms", hue="Reached.on.Time_Y.N", edgecolor = 'black')


# %%# customer rating 
sns.histplot(data, x = "Customer_rating", hue = "Reached.on.Time_Y.N", multiple = "dodge", bins=10)
plt.title("Histogram for Customer Rating")
plt.show()


#%%
# VIF
#data['Gender'] = data['Gender'].map({'Male':0, 'Female':1})
  
# the independent variables set
vif_x = integer_cols
vif_x.drop(columns=["Reached.on.Time_Y.N"], inplace=True)
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = vif_x.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(vif_x.values, i)
                          for i in range(len(vif_x.columns))]
vif_data

#%%
#Statistical Test

#ANOVA



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

#%%
# data split
data.drop(columns=['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender'], inplace=True)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





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
ax.set_xlabel('K-VALUE', fontsize=20)
ax.set_ylabel('ACC' ,fontsize=20)
ax.set_title('ACC PLOT' ,fontsize=28)


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
ax.set_xlabel('K-VALUE', fontsize=20)
ax.set_ylabel('MSE' ,fontsize=20)
ax.set_title('ELBOW PLOT' ,fontsize=28)

print("We chose k-neibors = 4, since when n = 4, ACC is highest and MSE is lowest.")

#%%
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred = classifier.predict(X_test)
print(y_pred)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(f'The confusion matrix is {cm}')
print(f'Test accuracy is {accuracy_score(y_test, y_pred)}')

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10).mean()
print(f'The mean accuracy is {accuracies}')


#%%
# Logistic Regression 

# encoding
# data['Gender'] = data['Gender'].map({'M':0, 'F':1})
# data['Mode_of_Shipment'] = data['Mode_of_Shipment'].map({'Flight': 1, 'Ship':2, 'Road':3})
# data['Product_importance'] = data['Product_importance'].map({'high': 1, 'medium':2, 'low':3})
# data['Warehouse_block'] = data['Warehouse_block'].map({'A': 1, 'B':2, 'C':3, 'D':4, 'F':5})

#%%
## split data
# X = data.drop("Reached.on.Time_Y.N", axis = 1)
# Y = data["Reached.on.Time_Y.N"]
# xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size = 0.8, random_state=1)

#%%
# X = label_4.drop("Reached.on.Time_Y.N", axis = 1)
# Y = label_4["Reached.on.Time_Y.N"]
#xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size = 0.8, random_state=1)

#%%
logitmodel = LogisticRegression()
logitmodel.fit(X_train, y_train)
print('Logit model accuracy (with the test set):', logitmodel.score(X_test, y_test))
print('Logit model accuracy (with the train set):', logitmodel.score(X_train, y_train))
print('Logit model Coefficient:', logitmodel.coef_)
print('Logit model Intercept:', logitmodel.intercept_)

# classification report
y_true, y_pred = y_test, logitmodel.predict(X_test)
print(classification_report(y_true, y_pred))


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

#%%
# Random forest

RandomForestModel = RandomForestClassifier(class_weight='balanced')
RandomForestModel.fit(X_train,y_train)
RandomForestPredict = RandomForestModel.predict(X_test)
print(RandomForestPredict)
print(classification_report(y_test,RandomForestPredict))

# Making the Confusion Matrix

cmRf = confusion_matrix(y_test, RandomForestPredict)

print(cmRf)
accuracyRf = accuracy_score(y_test, RandomForestPredict)
print(accuracyRf)

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
print('Logistic: ROC AUC=%.3f' % (lr_auc_rf))
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
