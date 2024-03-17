import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

import sqlite3
import pandas as pd
# Create a SQL connection to our SQLite database
con = sqlite3.connect("C:\AIAP15\data\cruise_pre.db")
cur = con.cursor()
df_pre = pd.read_sql_query("SELECT * from cruise_pre", con)
# Verify that result of SQL query is stored in the dataframe
print(df_pre.head())
con.close()

import sqlite3
import pandas as pd
con = sqlite3.connect("C:\AIAP15\data\cruise_post.db")
cur = con.cursor()
df_post = pd.read_sql_query("SELECT * from cruise_post", con)
# Verify that result of SQL query is stored in the dataframe
print(df_post.head())
con.close()

df = pd.concat((df_pre,df_post), axis=1) # to consolidate into one dataframe
# Verify that result of SQL query is stored in the dataframe
print(df.head())

df_distance=df.dropna(subset=['Cruise Distance']).reset_index() # clear na to prepare data for conversion to age
CatToNum =lambda x: ((x[-8:-3])) # to slice off 5 characters from Cruise Distance km value strings as numeric
CatToNum1 =lambda x: ((x[-11:-6])) # to slice off 5 characters from Cruise Distance miles value strings as numeric
# for loop to extract km values
for i in range (len(df_distance['Cruise Distance'])):
  try:
    df_distance.loc[i,'km values'] = CatToNum(df_distance.loc[i, 'Cruise Distance'])
    df_distance.loc[i,'km values'] = abs(float((df_distance.loc[i,'km values'])))
  except ValueError:
    df_distance.loc[i,'km values'] = CatToNum1(df_distance.loc[i, 'Cruise Distance'])
    df_distance.loc[i,'km values'] = 1.60934*abs(float((df_distance.loc[i,'km values'])))
#print(df_distance['km values']) # to inspect new df_distance 
mean_km = int(df_distance['km values'].mean()) # to get integer value of mean_km
#print(mean_km) # to verify mean_km value makes sense
mean_km_str = str(str(mean_km) +' KM') # to make mean_km_str
print(mean_km_str) # to inspect mean_km_str 
df['Cruise Distance'] = df['Cruise Distance'].fillna(mean_km_str) # to impute df with mean_km_str
#print(df['Cruise Distance'].head(20)) # to check fillna has worked
for i in range (len(df['Cruise Distance'])): # to make df['km values']
  try:
    df.loc[i,'km values'] = CatToNum(df.loc[i, 'Cruise Distance'])
    df.loc[i,'km values'] = abs(float((df.loc[i,'km values'])))
  except ValueError:
    df.loc[i,'km values'] = CatToNum1(df.loc[i, 'Cruise Distance'])
    df.loc[i,'km values'] = 1.60934*abs(float((df.loc[i,'km values'])))
df['km values']=df['km values'].astype(float) # to convert df['km values'] from object to float
print(df['km values'].describe()) # to inspect new df['km values']

import numpy as np
import pandas as pd
from datetime import datetime, date
def AGE(born): #function to convert DOB to age, as model can only process numeric data
    born = datetime.strptime(born, "%d/%m/%Y").date()
    today = date.today()
    return today.year - born.year - ((today.month,
                                      today.day) < (born.month,
                                                    born.day))
#print(AGE('23/08/2006')) # verified function can work
df_DOB=df.dropna(subset=['Date of Birth']).reset_index() # clear na to prepare data for conversion to age
print(df_DOB['Date of Birth']) # to check dropna has worked
dt_Age=[] # to construct data-Age array
for i in range (len(df_DOB['Date of Birth'])):
  try:
    dt_Age = np.append(dt_Age, AGE(df_DOB.loc[i, 'Date of Birth']))
  except ValueError:
    df_DOB.drop(df_DOB.index[i])
#print(dt_Age.shape) # to check data_Age array shape
#print(dt_Age) # to inspect data_Age array
mean_age = dt_Age.mean() # to find mean of data_Age
print(mean_age)
mean_DOB = "01/01/1984" # manually convert mean age to mean DOB
df['Date of Birth'] = df['Date of Birth'].fillna(mean_DOB) # to impute to na entries with mean DOB
#print(df['Date of Birth'].head()) # to inspect new df['Date of Birth']
df['Date of Birth']=df['Date of Birth'].fillna('01/01/1984') # feature engineering - to clear na
#print(df['Date of Birth']) # to check that fillna has worked
#print(AGE(df.loc[0, 'Date of Birth'])) # to check defined function AGE can work 
#DOBtoAge=lambda x: AGE(x) # tried but cannot work
#df[0,'Age']=AGE(df.loc[0, 'Date of Birth']) # tried this format and it worked
for i in range (len(df['Date of Birth'])):
  try:
    df.loc[i, 'Age']=AGE(df['Date of Birth'][i])
  except ValueError:
    df['Age'][i]=mean_age # to fill non-conforming DOB with mean_age
  #display(df.loc[i, 'Age']) # to check algo is updating age
#print(df['Age'])
print(df.head()) # to verify column Age is created in df

# Making Dictionaries of ordinal features
Ticket_Type_map = {
    'Standard'    :    1,
    'Deluxe'      :    2,
    'Luxury'      :    3
}

Gender_map = {'Female':  1.0,'Male':    2.0 }

Source_of_Traffic_map = {
    'Direct - Company Website'      :    3,
    'Direct - Email Marketing'      :    2,
    'Indirect - Search Engine'       :    1
}

Onboard_Wifi_Service_map = {
    'Not at all important':1, 'A little important':2, 'Somewhat important':3, 'Very important':4, 'Extremely important':5
}

Onboard_Dining_Service_map = {
    'Not at all important':1, 'A little important':2, 'Somewhat important':3, 'Very important':4, 'Extremely important':5
}

Onboard_Entertainment_map = {
    'Not at all important':1, 'A little important':2, 'Somewhat important':3, 'Very important':4, 'Extremely important':5
}
# Transform ordinal categorical features into numerical features

def encode(df_pre1): # to create a mapping function
    df_pre1.loc[:,'Ticket Type'] = df_pre1['Ticket Type'].map(Ticket_Type_map)
    df_pre1.loc[:,'Gender'] = df_pre1['Gender'].map(Gender_map)
    df_pre1.loc[:,'Source of Traffic'] = df_pre1['Source of Traffic'].map(Source_of_Traffic_map)
    df_pre1.loc[:,'Onboard Wifi Service'] = df_pre1['Onboard Wifi Service'].map(Onboard_Wifi_Service_map)
    df_pre1.loc[:,'Onboard Dining Service'] = df_pre1['Onboard Dining Service'].map(Onboard_Dining_Service_map)
    df_pre1.loc[:,'Onboard Entertainment'] = df_pre1['Onboard Entertainment'].map(Onboard_Entertainment_map)
    return df_pre1

df= encode(df) # convert ordinal categorical values in df to numerical


num_cols = ['Source of Traffic', 'Onboard Wifi Service','Onboard Dining Service', 'Onboard Entertainment',
            'Embarkation/Disembarkation time convenient','Ease of Online booking','Online Check-in',
            'Cabin Comfort','Cabin service','Baggage handling','Port Check-in Service','Onboard Service',
            'Cleanliness', 'Age', 'km values'] # to create numerical column list for pipeline
cat_cols = ['Gender', 'Gate location'] # to create categorical column list for pipeline


mode=df['Ticket Type'].dropna().mode()[0]
#print(mode)
df['Ticket Type'].fillna(mode,inplace=True) # to prepare target dataframe

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
num_pipeline = Pipeline(steps=[ 
    ('impute', SimpleImputer(strategy='mean')),
    ('scale',MinMaxScaler())
])
cat_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot',OneHotEncoder(handle_unknown='ignore', sparse=False))
])

from sklearn.compose import ColumnTransformer
col_trans = ColumnTransformer(transformers=[
    ('num_pipeline',num_pipeline,num_cols),
    ('cat_pipeline',cat_pipeline,cat_cols)
    ],
    remainder='drop',
    n_jobs=-1)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0) # to choose first clf model
clf_pipeline = Pipeline(steps=[          # to create clf pipeline
    ('col_trans', col_trans),
    ('model', clf)
])

from sklearn import set_config
set_config(display='diagram')
print(clf_pipeline) # visualisation of pipeline

from sklearn.model_selection import train_test_split
X = df[num_cols+cat_cols] # to create input dataframe
y = df['Ticket Type'] # to create target dataframe
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

clf_pipeline.fit(X_train, y_train) # to fit model
preds = clf_pipeline.predict(X_test) # to output prediction for visualisation
score = clf_pipeline.score(X_test, y_test) # to evaluate model efficiency
print(f"y_preds: {preds}") # visualisation of X_test Ticket Type prediction
print(f"logreg Model score: {score}") # to output model accuracy

#clf_pipeline.get_params() # to identify params for grid_params
grid_params = {'model__penalty' : ['none', 'l2'],    
               'model__C' : np.logspace(0.4, 4, 10)}

from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(clf_pipeline, grid_params, cv=3, scoring='accuracy') 
gs.fit(X_train, y_train)

print("logreg Best Score of train set: "+str(gs.best_score_))
print("logreg Best parameter set: "+str(gs.best_params_))
print("logreg Test Score: "+str(gs.score(X_test,y_test)))

from sklearn.neighbors import KNeighborsClassifier # repeat same process for another model
clf = KNeighborsClassifier()

clf_pipeline = Pipeline(steps=[
    ('col_trans', col_trans),
    ('model', clf)
])

from sklearn import set_config
set_config(display='diagram')
print(clf_pipeline)

from sklearn.model_selection import train_test_split
X = df[num_cols+cat_cols]
y = df['Ticket Type']
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

clf_pipeline.fit(X_train, y_train)
preds = clf_pipeline.predict(X_test)
score = clf_pipeline.score(X_test, y_test)
print(f"y_preds: {preds}") # X_test Ticket Type prediction
print(f"knn Model score: {score}") # model accuracy

#print(clf_pipeline.get_params())
grid_params = { 'model__weights' : ['uniform','distance'],
               'model__n_neighbors' : [5,7,9,11]}

from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(clf_pipeline, grid_params, cv=3, scoring='accuracy') # disable for error
gs.fit(X_train, y_train)
print("knn Best Score of train set: "+str(gs.best_score_))
print("knn Best parameter set: "+str(gs.best_params_))
print("knn Test Score: "+str(gs.score(X_test,y_test)))

from sklearn.ensemble import RandomForestClassifier # repeat same process for another model
clf = RandomForestClassifier (random_state=0)

clf_pipeline = Pipeline(steps=[
    ('col_trans', col_trans),
    ('model', clf)
])

from sklearn import set_config
set_config(display='diagram')
print(clf_pipeline)

from sklearn.model_selection import train_test_split
X = df[num_cols+cat_cols]
y = df['Ticket Type']
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

clf_pipeline.fit(X_train, y_train)
preds = clf_pipeline.predict(X_test)
score = clf_pipeline.score(X_test, y_test)
print(f"y_preds: {preds}") # X_test Ticket Type prediction
print(f"RF Model score: {score}") # model accuracy

#clf_pipeline.get_params() # to identify params for grid_params
grid_params = { 
    'model__n_estimators': [50, 100, 150], 
    'model__max_features': ['sqrt', 'log2'] 
} 
#from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(clf_pipeline, grid_params, cv=3, scoring='accuracy') 
gs.fit(X_train, y_train)
print("RF Best Score of train set: "+str(gs.best_score_))
print("RF Best parameter set: "+str(gs.best_params_)) 
print("RF Test Score: "+str(gs.score(X_test,y_test)))