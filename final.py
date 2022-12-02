#%%
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
%matplotlib inline
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
#non-delay delivery
data[data['Reached.on.Time_Y.N'] == 0].describe().T
# %%
print(sns.boxplot(data['Customer_rating']))
print(sns.boxplot(data['Cost_of_the_Product']))
#%% 
#only one column may need to remove outliers
print(sns.boxplot(data['Prior_purchases']))
#%%
print(sns.boxplot(data['Discount_offered']))
#%%
print(sns.boxplot(data['Weight_in_gms']))

#%%
# Pairplots for integer columns (Initial EDA)
integer_cols = data.select_dtypes(include = ['int64'])
print(integer_cols.head())
intcols = integer_cols.columns
# %%
# counts plots for every column
cols = ['Warehouse_block', 'Mode_of_Shipment', 'Customer_care_calls', 'Customer_rating', 'Prior_purchases', 'Product_importance', 'Gender', 'Reached.on.Time_Y.N']
plt.figure(figsize = (16, 20))
sns.set_theme(style="ticks", palette="pastel")
nplot = 1
for i in range(len(cols)):
    if nplot <= len(cols):
        ax = plt.subplot(4, 2, nplot)
        sns.countplot(x = cols[i], data = data, ax = ax)
        plt.title(f"\n{cols[i]} Counts", fontsize = 15)
        nplot += 1
plt.show()
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
plt.figure(figsize = (18, 7))
sns.heatmap(data.corr(), annot = True, fmt = '0.2f', annot_kws = {'size' : 15}, linewidth = 5, linecolor = 'orange')
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
import plotly.express as px
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
# %%
px.box(data, x = 'Reached.on.Time_Y.N', y = 'Discount_offered',color = 'Reached.on.Time_Y.N')
# %%
ware_block_weight = data.groupby(['Warehouse_block'])['Weight_in_gms'].sum().reset_index()
ware_block_weight
# %%
plt.figure(figsize = (11, 7))
ax = sns.boxplot(x="Product_importance", y="Discount_offered", hue="Reached.on.Time_Y.N", palette={0: "y", 1: "b"}, data=data).set(title = "Do product_importance and discount affect delivery?")
plt.show()

# %%
# violinplot about discount + prior + reach or not
sns.violinplot(x="Prior_purchases", y="Discount_offered", hue="Reached.on.Time_Y.N", split=True, inner="quart",palette={0: "y", 1: "b"}, data=data).set(title = "ethnics, gender, education in world2")
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
sns.boxplot(x="Customer_care_calls", y="Discount_offered", hue="Reached.on.Time_Y.N", palette={0: "y", 1: "b"}, data=data).set(title = "do calls and discount affect delivery?")
sns.despine(left=True)
plt.show()
# %%
