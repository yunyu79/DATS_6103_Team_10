#%%
import pandas as pd
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
# %%
data = pd.read_csv(r"D:/GraSem1/DS6103/FinalProject/Train.csv")
data
# %%
data.info()
data.isna().sum() / len(data)
# %%
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
