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

# %%
import matplotlib.pyplot as plt
plt.figure(figsize = (18, 7))
sns.heatmap(data.corr(), annot = True, fmt = '0.2f', annot_kws = {'size' : 15}, linewidth = 5, linecolor = 'orange')
plt.show()
# %%
