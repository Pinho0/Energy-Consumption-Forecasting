import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest

###Data Preparation ----------------------------------------

df = pd.read_csv('./energy_consumption.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

#Convert 'timestamp' column to datetime and extract components
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['second'] = df['timestamp'].dt.second
del df['timestamp']

#Minute and second is always 1000 and year is always 2022, so we can ignore them 
del df['year']
del df['minute']
del df['second']

#Process the values of the categorical columns
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

#Get the columns by data type    
categorical_columns = df.select_dtypes(include=['object', 'int']).columns.tolist() #occupancy is a int but should be on the numerical array 
numerical_columns = df.select_dtypes(include=['float']).columns.tolist()
if 'occupancy' in categorical_columns:
    categorical_columns.remove('occupancy')
if 'occupancy' not in numerical_columns:
    numerical_columns.append('occupancy')
    
#Remove duplicates 
df = df.drop_duplicates()

#Check for NA
print(df.isnull().sum())

#save the dataset
df.to_csv('./energy_consumption_processed.csv', index=False)


###Exploratory Data Analysis(EDA) --------------------------

#Correlation Heatmap 
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df[numerical_columns].corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
plt.show()

#Pairplot
sns.pairplot(df[numerical_columns])
plt.show()
     
#Distribution
fig,ax = plt.subplots(2,3,figsize=(15,10))
row = col = 0
for n,i in enumerate(numerical_columns):
    if (n%3 == 0) & (n > 0):
        row += 1
        col = 0
    df[i].plot(kind="kde",ax=ax[row,col])
    ax[row,col].set_title(i)
    col += 1
plt.tight_layout()
plt.show()

#D’Agostino-Pearson’s normality test
for i in numerical_columns:
    print(f'{i}: {"Not Gaussian" if normaltest(df[i].values,)[1]<0.05 else "Gaussian"}  {normaltest(df[i].values)}')
