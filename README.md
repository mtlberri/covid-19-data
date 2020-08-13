# Study of COVID data in the US (based on NYT data)


```python
import pandas as pd
import numpy as np
import matplotlib
import datetime
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
from matplotlib.ticker import FuncFormatter
%matplotlib inline
import seaborn as sns
sns.set()
import time
import math
```


```python
# matplotlib.style.use('seaborn-whitegrid')
```

## Get NYT data and prepare it


```python
# Read from NYT data CSV
df_ = pd.read_csv('us-states.csv')
```


```python
# Make date string a proper datetime data type
df_['date']=df_['date'].apply(lambda x:pd.Timestamp(x))
df_.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>state</th>
      <th>fips</th>
      <th>cases</th>
      <th>deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-21</td>
      <td>Washington</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-22</td>
      <td>Washington</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-23</td>
      <td>Washington</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-24</td>
      <td>Illinois</td>
      <td>17</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-24</td>
      <td>Washington</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add states abbreviations
df_abb = pd.read_csv('other_input/us-states-abbreviations.csv')
df = pd.merge(left=df_, right=df_abb, on='state', how='outer')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>state</th>
      <th>fips</th>
      <th>cases</th>
      <th>deaths</th>
      <th>Abbreviation</th>
      <th>Postal Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-21</td>
      <td>Washington</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>Wash.</td>
      <td>WA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-22</td>
      <td>Washington</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>Wash.</td>
      <td>WA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-23</td>
      <td>Washington</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>Wash.</td>
      <td>WA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-24</td>
      <td>Washington</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>Wash.</td>
      <td>WA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-25</td>
      <td>Washington</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>Wash.</td>
      <td>WA</td>
    </tr>
  </tbody>
</table>
</div>




```python
last_date = df.date.max()
last_date
```




    Timestamp('2020-07-28 00:00:00')



## US States ranked by number of cases


```python
# States with most cases (on last available date)
df_ti = df[df['date'] == last_date].sort_values(by='cases', ascending=False)
df_ti.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>state</th>
      <th>fips</th>
      <th>cases</th>
      <th>deaths</th>
      <th>Abbreviation</th>
      <th>Postal Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>562</th>
      <td>2020-07-28</td>
      <td>California</td>
      <td>6</td>
      <td>474951</td>
      <td>8716</td>
      <td>Calif.</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>1889</th>
      <td>2020-07-28</td>
      <td>Florida</td>
      <td>12</td>
      <td>441969</td>
      <td>6116</td>
      <td>Fla.</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>2039</th>
      <td>2020-07-28</td>
      <td>New York</td>
      <td>36</td>
      <td>417591</td>
      <td>32333</td>
      <td>N.Y.</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1269</th>
      <td>2020-07-28</td>
      <td>Texas</td>
      <td>48</td>
      <td>412744</td>
      <td>6515</td>
      <td>Tex.</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>2782</th>
      <td>2020-07-28</td>
      <td>New Jersey</td>
      <td>34</td>
      <td>182215</td>
      <td>15825</td>
      <td>N.J.</td>
      <td>NJ</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the States ranked by number of cases
fig, ax = plt.subplots(figsize=(12, 8))
number_of_states_to_visualize = 20
df_ti_p = df_ti.head(number_of_states_to_visualize).copy()
x = df_ti_p['Abbreviation']
y = df_ti_p['cases']
# bar_container = ax.bar(x=x, height=y)
sns.barplot(x=x, y=y, palette='Reds_r', ax=ax)
ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x,p: format(int(x),',')))
ax.get_xaxis().label.set_visible(False)
ax.get_yaxis().label.set_visible(False)
out = ax.set_title(f'Total number of cases per State (on {last_date.strftime("%B %d, %Y")})')
```


![png](https://mtlberriawsbucket.s3.us-east-2.amazonaws.com/Covid/output_10_0.png)



```python
# fig.savefig(f'output_visuals/Total_number_of_Cases_{last_date.strftime("%B_%d_%Y")}.png')
```

## Evolution throught time of the number of cases per state


```python
# Number of states considered (among top impacted)
nb_states_time_plot = 10
df_states_considered = df_ti['state'].head(nb_states_time_plot)
# index on the most impacted states only
boolean_index = df['state'].isin(df_states_considered)
# df time plot
df_tp = df[boolean_index]
df_tp.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>state</th>
      <th>fips</th>
      <th>cases</th>
      <th>deaths</th>
      <th>Abbreviation</th>
      <th>Postal Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2778</th>
      <td>2020-07-24</td>
      <td>New Jersey</td>
      <td>34</td>
      <td>180265</td>
      <td>15765</td>
      <td>N.J.</td>
      <td>NJ</td>
    </tr>
    <tr>
      <th>2779</th>
      <td>2020-07-25</td>
      <td>New Jersey</td>
      <td>34</td>
      <td>180778</td>
      <td>15776</td>
      <td>N.J.</td>
      <td>NJ</td>
    </tr>
    <tr>
      <th>2780</th>
      <td>2020-07-26</td>
      <td>New Jersey</td>
      <td>34</td>
      <td>181283</td>
      <td>15787</td>
      <td>N.J.</td>
      <td>NJ</td>
    </tr>
    <tr>
      <th>2781</th>
      <td>2020-07-27</td>
      <td>New Jersey</td>
      <td>34</td>
      <td>181732</td>
      <td>15804</td>
      <td>N.J.</td>
      <td>NJ</td>
    </tr>
    <tr>
      <th>2782</th>
      <td>2020-07-28</td>
      <td>New Jersey</td>
      <td>34</td>
      <td>182215</td>
      <td>15825</td>
      <td>N.J.</td>
      <td>NJ</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x='date', y='cases', hue='state', palette='bright', hue_order=df_states_considered,
             ax=ax, data=df_tp)
ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x,p: format(int(x),',')))
```


![png](https://mtlberriawsbucket.s3.us-east-2.amazonaws.com/Covid/output_14_0.png)


TAKE-AWAY: New York is flatening, but California/Florida/Texas are experiencing an exponential growth of the number of cases.


```python
# fig.savefig(f'output_visuals/Evolution_total_number_of_cases_through_time_' \
#             f'{last_date.strftime("%B_%d_%Y")}.png')
```

## Face Masks Effectiveness (study per US county)


```python
# Initiate dfm dataframe to do that mask correlation analysis (cases per county through time)
dfm0 = pd.read_csv('us-counties.csv')
# Cast date into proper datetime objects
dfm0['date'] = dfm0['date'].apply(lambda x: pd.Timestamp(x))
dfm0.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>county</th>
      <th>state</th>
      <th>fips</th>
      <th>cases</th>
      <th>deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-21</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>53061.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-22</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>53061.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-23</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>53061.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-24</td>
      <td>Cook</td>
      <td>Illinois</td>
      <td>17031.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-24</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>53061.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(dfm0)
```




    379550




```python
# Inital number of counties from the input file
nb_fips0 = len(set(dfm0.fips))
print(f'The initial number of counties out of the input file is: {nb_fips0}')
```

    The initial number of counties out of the input file is: 3187


#### Side analysis on why fips should be used (and not only the county name which is not unique in US)


```python
# County name and its fips
dfcf = dfm0.loc[:, ['county', 'fips', 'state']].drop_duplicates()
dfcf.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>county</th>
      <th>fips</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Snohomish</td>
      <td>53061</td>
      <td>Washington</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfcf1 = dfcf.groupby(by='county').count()
dfcf2 = dfcf1[dfcf1.fips > 1]
dfcf2.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>state</th>
    </tr>
    <tr>
      <th>county</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adair</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Adams</th>
      <td>12</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
avg_county_repeat = np.round(dfcf2.state.mean(), 1)
```


```python
dfcf[dfcf.county=='Adair']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>county</th>
      <th>fips</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3388</th>
      <td>Adair</td>
      <td>19001</td>
      <td>Iowa</td>
    </tr>
    <tr>
      <th>8892</th>
      <td>Adair</td>
      <td>29001</td>
      <td>Missouri</td>
    </tr>
    <tr>
      <th>11906</th>
      <td>Adair</td>
      <td>40001</td>
      <td>Oklahoma</td>
    </tr>
    <tr>
      <th>33965</th>
      <td>Adair</td>
      <td>21001</td>
      <td>Kentucky</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f'There are {len(dfcf2)} counties in the US whose same county name is also used \n'
      f'in an average of {avg_county_repeat - 1} other states')
```

    There are 428 counties in the US whose same county name is also used 
    in an average of 3.0 other states


For example, the 'Adair' county name is used in the state of Iowa, but also in 3 other states: 'Missouri', 'Oklahoma' and 'Kentucky'.

###  Prepare and standardize the counties fips codes for use as a key for table merging


```python
# fips dtype
# dfm0.fips.dtype
```


```python
def fipsAsString(fips_float):
    """Converts a fips code read as float into a formatted string"""
    try:
        fips_int = int(fips_float)
        fips_str = str(fips_int)
    except:
        fips_str = 'NA'
    return fips_str
```


```python
# If the fips column is a flaot, convert it to a string (for use as a key later)
def convertFromTypeToStr(df, from_type):
    """Converts a dataframe (Series) fips column data type into a formatted string"""
    if df.dtype == from_type:
        df = df.apply(fipsAsString)
        df = df.astype('str')
        print('df converted to string')
    else:
        print(f'df dtype already: {df.dtypes}')
    return df
```


```python
dfm0['fips'] = convertFromTypeToStr(df=dfm0['fips'], from_type=np.float64)
# fips are now properly formatted as strings in the main dfm
dfm0.head(1)
```

    df dtype already: object





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>county</th>
      <th>state</th>
      <th>fips</th>
      <th>cases</th>
      <th>deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-21</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>53061</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
set(dfm0[dfm0['fips']=="NA"].county)
```




    {'Joplin', 'Kansas City', 'New York City', 'Unknown'}



Note for future: above counties are not considered in a first iteration of tha study because they do not map to a single fips. Could be adjusted in future iterations of that study.
Reason for that is the choice of NYT to aggregate NYC boroughs (which are all separate counties), into one big bucket called NYC (which consequently do not have a single fips code). 
That could be worked around with a little thinking in the next iterations.


```python
len(dfm0)
```




    379550




```python
# For now, droppping counties without fip (may have to change that in the future,
# because dropping New York City and the counties mentioned above)
dfm0 = dfm0[dfm0.fips != 'NA']
len(dfm0)
```




    375734




```python
# Create the target dfm
dfm = dfm0.loc[:, ['fips', 'county', 'state']].copy()
dfm.drop_duplicates(subset='fips', inplace=True)
# m2w means: mean over past 2 weeks
dfm['new_cases_m2w'] = np.nan
dfm.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>county</th>
      <th>state</th>
      <th>new_cases_m2w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>53061</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17031</td>
      <td>Cook</td>
      <td>Illinois</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6059</td>
      <td>Orange</td>
      <td>California</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4013</td>
      <td>Maricopa</td>
      <td>Arizona</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6037</td>
      <td>Los Angeles</td>
      <td>California</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Number of lines (counties)
print(f'Number of counties (fips) being considered in the study at that point: {len(dfm)}')
```

    Number of counties (fips) being considered in the study at that point: 3187


### Calculate the average number of new cases per day per county on the last two weeks (new_cases_m2w)


```python
# Calulate the new cases each day
dfm1 = dfm0.copy()
dfm1.sort_values(by='date', inplace=True)
# Only keep the last two keeks
time_condition = dfm1['date'].iloc[-1] - dfm1['date'] < pd.Timedelta(weeks=2)
dfm1 = dfm1[time_condition]
dfm1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>county</th>
      <th>state</th>
      <th>fips</th>
      <th>cases</th>
      <th>deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>336809</th>
      <td>2020-07-15</td>
      <td>Grady</td>
      <td>Oklahoma</td>
      <td>40051</td>
      <td>264</td>
      <td>5</td>
    </tr>
    <tr>
      <th>336808</th>
      <td>2020-07-15</td>
      <td>Garvin</td>
      <td>Oklahoma</td>
      <td>40049</td>
      <td>140</td>
      <td>3</td>
    </tr>
    <tr>
      <th>336807</th>
      <td>2020-07-15</td>
      <td>Garfield</td>
      <td>Oklahoma</td>
      <td>40047</td>
      <td>138</td>
      <td>2</td>
    </tr>
    <tr>
      <th>336802</th>
      <td>2020-07-15</td>
      <td>Creek</td>
      <td>Oklahoma</td>
      <td>40037</td>
      <td>231</td>
      <td>9</td>
    </tr>
    <tr>
      <th>336805</th>
      <td>2020-07-15</td>
      <td>Dewey</td>
      <td>Oklahoma</td>
      <td>40043</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>377407</th>
      <td>2020-07-28</td>
      <td>Meade</td>
      <td>Kentucky</td>
      <td>21163</td>
      <td>72</td>
      <td>2</td>
    </tr>
    <tr>
      <th>377408</th>
      <td>2020-07-28</td>
      <td>Menifee</td>
      <td>Kentucky</td>
      <td>21165</td>
      <td>24</td>
      <td>0</td>
    </tr>
    <tr>
      <th>377409</th>
      <td>2020-07-28</td>
      <td>Mercer</td>
      <td>Kentucky</td>
      <td>21167</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>377411</th>
      <td>2020-07-28</td>
      <td>Monroe</td>
      <td>Kentucky</td>
      <td>21171</td>
      <td>83</td>
      <td>2</td>
    </tr>
    <tr>
      <th>379549</th>
      <td>2020-07-28</td>
      <td>Weston</td>
      <td>Wyoming</td>
      <td>56045</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>44469 rows × 6 columns</p>
</div>




```python
nb_fips1 = len(set(dfm1.fips))
nb_fips1
```




    3186




```python
print(f'Number of fips for which no data was reported on the past two weeks: {nb_fips0 - nb_fips1}')
```

    Number of fips for which no data was reported on the past two weeks: 1


No Data will be possible to be computed for the m2w (mean past two weeks) for those fips.


```python
# fips code for which no data was reported on thew past two weeks
set(dfm0.fips).difference(set(dfm1.fips))
```




    {'16061'}



m2w computation loop:


```python
# Loop through the counties (using their fips) and compute mean of new cases on past two weeks
start_time = time.time()
for fips in dfm.fips:
    dfc = dfm1[dfm1.fips == fips].copy()
    dfc['new_cases'] = dfc.cases - dfc.cases.shift(1)
    fips_mean = dfc.new_cases.mean()
    dfm.loc[dfm.fips == fips, 'new_cases_m2w'] = fips_mean
print(f'Data processing time: {time.time() - start_time}')
```

    Data processing time: 23.282297134399414



```python
dfm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>county</th>
      <th>state</th>
      <th>new_cases_m2w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>53061</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>64.692308</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17031</td>
      <td>Cook</td>
      <td>Illinois</td>
      <td>521.461538</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6059</td>
      <td>Orange</td>
      <td>California</td>
      <td>600.153846</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4013</td>
      <td>Maricopa</td>
      <td>Arizona</td>
      <td>1920.230769</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6037</td>
      <td>Los Angeles</td>
      <td>California</td>
      <td>2741.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>368310</th>
      <td>30019</td>
      <td>Daniels</td>
      <td>Montana</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>368434</th>
      <td>31183</td>
      <td>Wheeler</td>
      <td>Nebraska</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>369765</th>
      <td>54017</td>
      <td>Doddridge</td>
      <td>West Virginia</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>375990</th>
      <td>49055</td>
      <td>Wayne</td>
      <td>Utah</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>378314</th>
      <td>38001</td>
      <td>Adams</td>
      <td>North Dakota</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3187 rows × 4 columns</p>
</div>



### Include the population per county (merge on county fips code)


```python
# Read population csv
df_pop = pd.read_csv('other_input/us-county-population-census.csv', encoding='iso-8859-1')
df_pop.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>STATE</th>
      <th>COUNTY</th>
      <th>STNAME</th>
      <th>CTYNAME</th>
      <th>POPULATION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>Alabama</td>
      <td>Alabama</td>
      <td>4903185</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Alabama</td>
      <td>Autauga County</td>
      <td>55869</td>
    </tr>
  </tbody>
</table>
</div>



Need to create first a fips 'key' out of the STATE and COUNTY numbers, for later merge in the main df:


```python
# df_pop.dtypes
```


```python
# Make STATE and COUNTY strings
df_pop['STATE'] = convertFromTypeToStr(df=df_pop['STATE'], from_type=np.int64)
df_pop['COUNTY'] = convertFromTypeToStr(df=df_pop['COUNTY'], from_type=np.int64)
# Zero padd the county (should be 3 digits)
df_pop['COUNTY'] = df_pop['COUNTY'].apply(lambda x: x.zfill(3))
# Create the county fips column in the population df
df_pop['fips'] = df_pop['STATE'] + df_pop['COUNTY']
# Drop unnecessary columns before merge
df_pop1 = df_pop.drop(columns=['STATE', 'COUNTY', 'STNAME'])
df_pop1.head(3)
```

    df converted to string
    df converted to string





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CTYNAME</th>
      <th>POPULATION</th>
      <th>fips</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>4903185</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Autauga County</td>
      <td>55869</td>
      <td>1001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Baldwin County</td>
      <td>223234</td>
      <td>1003</td>
    </tr>
  </tbody>
</table>
</div>



Merge the population data into the main df (base on fips key)


```python
# Merge the population data into the main df
dfm2 = pd.merge(left=dfm, right=df_pop1, on='fips', how='inner', left_index=True)
dfm2.drop(columns='CTYNAME', inplace=True)
dfm2.reset_index(inplace=True, drop=True)
```

Check if some fips from the main df where not found in the population data:


```python
nb_fips2 = len(set(dfm2.fips))
print(f'Number of fips for which we have no data in term of population (lost when inner-merging): '\
      f'{nb_fips1 - nb_fips2}')
```

    Number of fips for which we have no data in term of population (lost when inner-merging): 82



```python
# Identify fips that were in the main df but not found in the population data
fips_nopop = set(dfm1.fips).difference(dfm2.fips)
# Full list of fips counties that were not found in the population data
# dfm1[dfm1.fips.isin(fips_nopop)].drop_duplicates(subset=['county', 'state'])
```


```python
# Full list of states that were not found in the population data
set(dfm1[dfm1.fips.isin(fips_nopop)].state)
```




    {'Northern Mariana Islands', 'Puerto Rico', 'Virgin Islands'}



STATUS ON COUNTIES DROPPED: So up to this cell, we had dropped already Joplin, Kansas City and NYC because NYT data was not linking those directly to a unique fips. And now we drop as well the Northern Marina Islands, Puerto Rico and the Virgin Islands because the population data was not available in our source file for those. (Those are known limitations of the first sprint iteration, can be fixed in later iteration). 
We are still covering mopre than 97% of all the counties in US at that point.


```python
print(f'Initial number of fips: {nb_fips0}\n'\
     f'Current number of fips in the main df: {nb_fips2}')
```

    Initial number of fips: 3187
    Current number of fips in the main df: 3104



```python
nb_fips2 / nb_fips0
```




    0.9739566990900533




```python
dfm2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>county</th>
      <th>state</th>
      <th>new_cases_m2w</th>
      <th>POPULATION</th>
      <th>new_cases_m2wN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>53061</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>64.692308</td>
      <td>822083</td>
      <td>7.869316</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17031</td>
      <td>Cook</td>
      <td>Illinois</td>
      <td>521.461538</td>
      <td>5150233</td>
      <td>10.125009</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6059</td>
      <td>Orange</td>
      <td>California</td>
      <td>600.153846</td>
      <td>3175692</td>
      <td>18.898364</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4013</td>
      <td>Maricopa</td>
      <td>Arizona</td>
      <td>1920.230769</td>
      <td>4485414</td>
      <td>42.810558</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6037</td>
      <td>Los Angeles</td>
      <td>California</td>
      <td>2741.000000</td>
      <td>10039107</td>
      <td>27.303225</td>
    </tr>
  </tbody>
</table>
</div>



We now have the mean number of cases per county over the past two weeks and the population per county.
We can now normalize the number of cases over the population of the county.
m2wN: mean number of cases over the last two weeks, Normalized over the population (per 100,000 capita).


```python
# Normalize the number of new cases with the population (per 100,000 habitants)
pop_basis = 100000
# m2wN : mean 2 weeks Normalized
dfm2['new_cases_m2wN'] = (dfm2['new_cases_m2w'] / dfm2['POPULATION']) * pop_basis
dfm2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>county</th>
      <th>state</th>
      <th>new_cases_m2w</th>
      <th>POPULATION</th>
      <th>new_cases_m2wN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>53061</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>64.692308</td>
      <td>822083</td>
      <td>7.869316</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17031</td>
      <td>Cook</td>
      <td>Illinois</td>
      <td>521.461538</td>
      <td>5150233</td>
      <td>10.125009</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6059</td>
      <td>Orange</td>
      <td>California</td>
      <td>600.153846</td>
      <td>3175692</td>
      <td>18.898364</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4013</td>
      <td>Maricopa</td>
      <td>Arizona</td>
      <td>1920.230769</td>
      <td>4485414</td>
      <td>42.810558</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6037</td>
      <td>Los Angeles</td>
      <td>California</td>
      <td>2741.000000</td>
      <td>10039107</td>
      <td>27.303225</td>
    </tr>
  </tbody>
</table>
</div>



### Include percentage of mask use per county


```python
# Import the use of mask and bucket FREQUENTLY and ALWAYS together (NYT data)
df_mu = pd.read_csv('mask-use/mask-use-by-county.csv')
df_mu.rename(columns={'COUNTYFP': 'fips'}, inplace=True)
df_mu.fips = convertFromTypeToStr(df=df_mu.fips, from_type=np.int64)
df_mu['mask_use'] = (df_mu['FREQUENTLY'] + df_mu['ALWAYS']) * 100
df_mu.head()
```

    df converted to string





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>NEVER</th>
      <th>RARELY</th>
      <th>SOMETIMES</th>
      <th>FREQUENTLY</th>
      <th>ALWAYS</th>
      <th>mask_use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>0.053</td>
      <td>0.074</td>
      <td>0.134</td>
      <td>0.295</td>
      <td>0.444</td>
      <td>73.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1003</td>
      <td>0.083</td>
      <td>0.059</td>
      <td>0.098</td>
      <td>0.323</td>
      <td>0.436</td>
      <td>75.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1005</td>
      <td>0.067</td>
      <td>0.121</td>
      <td>0.120</td>
      <td>0.201</td>
      <td>0.491</td>
      <td>69.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1007</td>
      <td>0.020</td>
      <td>0.034</td>
      <td>0.096</td>
      <td>0.278</td>
      <td>0.572</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1009</td>
      <td>0.053</td>
      <td>0.114</td>
      <td>0.180</td>
      <td>0.194</td>
      <td>0.459</td>
      <td>65.3</td>
    </tr>
  </tbody>
</table>
</div>



Defining 'mask_use' as the percentage of people who wear the mask 'FREQUENTLY' or 'ALWAYS' in a given county. Meaning the percentage of people wearing their mask often in a given state.


```python
try:
    df_mu.drop(columns=['NEVER', 'RARELY', 'SOMETIMES', 'FREQUENTLY', 'ALWAYS'], inplace=True)
except:
    print('Could not drop the columns, potentially because cell has already been '\
          'run and the columns are already deleted')
df_mu.head(3)
```

    Could not drop the columns, potentially because cell has already been run and the columns are already deleted





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>mask_use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>73.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1003</td>
      <td>75.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1005</td>
      <td>69.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merge with main df
dfm3 = pd.merge(left=dfm2, right=df_mu, on='fips', how='inner', left_index=True)
# Rename the new cases column for convenience
dfm3.rename(columns={'new_cases_m2wN': 'new_cases', 'POPULATION': 'county_population'}, inplace=True)
dfm3.reset_index(inplace=True, drop=True)
```

Check if some fips from the main df where not found in the mask use data:


```python
nb_fips3 = len(set(dfm3.fips))
print(f'Number of fips for which we have no data in term of mask use (lost when inner-merging): '\
      f'{nb_fips2 - nb_fips3}')
```

    Number of fips for which we have no data in term of mask use (lost when inner-merging): 0



```python
# Complete df for mask use study
dfm3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>county</th>
      <th>state</th>
      <th>new_cases_m2w</th>
      <th>county_population</th>
      <th>new_cases</th>
      <th>mask_use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>53061</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>64.692308</td>
      <td>822083</td>
      <td>7.869316</td>
      <td>91.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17031</td>
      <td>Cook</td>
      <td>Illinois</td>
      <td>521.461538</td>
      <td>5150233</td>
      <td>10.125009</td>
      <td>88.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6059</td>
      <td>Orange</td>
      <td>California</td>
      <td>600.153846</td>
      <td>3175692</td>
      <td>18.898364</td>
      <td>91.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4013</td>
      <td>Maricopa</td>
      <td>Arizona</td>
      <td>1920.230769</td>
      <td>4485414</td>
      <td>42.810558</td>
      <td>89.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6037</td>
      <td>Los Angeles</td>
      <td>California</td>
      <td>2741.000000</td>
      <td>10039107</td>
      <td>27.303225</td>
      <td>91.7</td>
    </tr>
  </tbody>
</table>
</div>



## New Cases | Use of Mask

Study the correlation (if any) in between the use of mask in a county, and the average number of daily new cases over the past two week. Hypothesis being that: "The more people wear the mask, the less we should have new cases per capita".


```python
# Scatter plot and correlation study
def myScatterPlot(df, save_name=None):
    """Scatter plot for covid mask use study"""
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.scatterplot(x='mask_use', y='new_cases', size='county_population', sizes=(10, 500),
                    data=df, ax=ax, alpha=0.9)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.get_xaxis().set_label_text('% of population wearing the mask frequently')
    ax.get_yaxis().set_label_text('new cases per day (per 100,000 caapita)')
    ax.legend(labelspacing=1.5)
    ax.set_title('Correlation between the number of new cases per 100,000 capita and the use of mask'\
                 '\nPer US county')
    if save_name:
        ax.get_figure().savefig('output_visuals/' + save_name, dpi=300)
```


```python
# myScatterPlot(df=dfm3)
```


```python
dfm3.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>new_cases_m2w</th>
      <th>county_population</th>
      <th>new_cases</th>
      <th>mask_use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3102.000000</td>
      <td>3.104000e+03</td>
      <td>3102.000000</td>
      <td>3104.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>20.784323</td>
      <td>1.030396e+05</td>
      <td>16.831821</td>
      <td>71.607764</td>
    </tr>
    <tr>
      <th>std</th>
      <td>104.620879</td>
      <td>3.281160e+05</td>
      <td>18.805614</td>
      <td>13.084540</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-12.076923</td>
      <td>4.040000e+02</td>
      <td>-6.597176</td>
      <td>25.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.769231</td>
      <td>1.123400e+04</td>
      <td>4.991518</td>
      <td>62.400000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.230769</td>
      <td>2.621500e+04</td>
      <td>10.776937</td>
      <td>72.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>11.384615</td>
      <td>6.857625e+04</td>
      <td>22.446587</td>
      <td>81.825000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2925.769231</td>
      <td>1.003911e+07</td>
      <td>266.980360</td>
      <td>99.200000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Removing extreme cases
myScatterPlot(df=dfm3[dfm3.new_cases < 150], save_name=None)
```


![png](https://mtlberriawsbucket.s3.us-east-2.amazonaws.com/Covid/output_78_0.png)


TAKE-AWAY: The percentage of population wearing a mask alone do not seem to be strongly correlated to the number of new cases p.h.t.c.. in a given county. It even seems to be the opposite of my initial hypothesis as I had most probably inversed the causality. It is not because people wear the mask frequently that there are less cases. The causality is inverse: it is because there are a lot of new cases that people wear the mask frequently.

What is more visible on that data is that the most populated counties tend to have the higher propency to wear masks (big disks on the right) while having a number of new cases p.h.t.c. being equal or higher than smaller (and potentially less dense) counties.

Next step will be to study the effect of the county population density, which intuitively should be positively correlated to the number of new cases p.h.t.c..

*p.h.t.c. = per hundred thousand capita

## New Cases | Population Density

Hypothesis: The more a county has a dense population, the more new cases there should be.


```python
# Import the superficie of each county
dfa = pd.read_csv('other_input/LND01-census-land-area.csv')
dfa.drop(columns='Areaname', inplace=True)
dfa.rename(columns={'STCOU': 'fips', 'LND010200D': 'county_area'}, inplace=True)
dfa.fips = convertFromTypeToStr(df=dfa.fips, from_type=np.int64)
dfa.head(3)
```

    df converted to string





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>county_area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3794083.06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000</td>
      <td>52419.02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001</td>
      <td>604.45</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merge are in the main df
dfm4 = dfm3.copy()
dfm4 = pd.merge(left=dfm3, right=dfa, on='fips', how='inner', left_index=True)
# dfm4
```

Check if some fips from the main df where not found in the county land area data:


```python
nb_fips4 = len(set(dfm4.fips))
print(f'Number of fips for which we have no data in term of land area (lost when inner-merging): '\
      f'{nb_fips3 - nb_fips4}')
```

    Number of fips for which we have no data in term of land area (lost when inner-merging): 2



```python
# Identify fips that were in the main df but not found in the population data
fips_noarea = set(dfm3.fips).difference(dfm4.fips)
# Full list of fips counties that were not found in the population data
# dfm1[dfm1.fips.isin(fips_nopop)].drop_duplicates(subset=['county', 'state'])
```


```python
# Full list of states that were not found in the county area data
set(dfm3[dfm3.fips.isin(fips_noarea)].state)
```




    {'Alaska', 'South Dakota'}



Calculate the population density per county now:


```python
# Calculate population density
dfm4['pop_density'] = dfm4.county_population / dfm4.county_area
dfm4.replace(np.inf, np.nan, inplace=True)
dfm4.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>county</th>
      <th>state</th>
      <th>new_cases_m2w</th>
      <th>county_population</th>
      <th>new_cases</th>
      <th>mask_use</th>
      <th>county_area</th>
      <th>pop_density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3036</th>
      <td>53061</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>64.692308</td>
      <td>822083</td>
      <td>7.869316</td>
      <td>91.2</td>
      <td>2196.41</td>
      <td>374.284856</td>
    </tr>
    <tr>
      <th>625</th>
      <td>17031</td>
      <td>Cook</td>
      <td>Illinois</td>
      <td>521.461538</td>
      <td>5150233</td>
      <td>10.125009</td>
      <td>88.4</td>
      <td>1635.04</td>
      <td>3149.912540</td>
    </tr>
    <tr>
      <th>221</th>
      <td>6059</td>
      <td>Orange</td>
      <td>California</td>
      <td>600.153846</td>
      <td>3175692</td>
      <td>18.898364</td>
      <td>91.0</td>
      <td>947.98</td>
      <td>3349.956750</td>
    </tr>
    <tr>
      <th>107</th>
      <td>4013</td>
      <td>Maricopa</td>
      <td>Arizona</td>
      <td>1920.230769</td>
      <td>4485414</td>
      <td>42.810558</td>
      <td>89.2</td>
      <td>9224.27</td>
      <td>486.262219</td>
    </tr>
    <tr>
      <th>210</th>
      <td>6037</td>
      <td>Los Angeles</td>
      <td>California</td>
      <td>2741.000000</td>
      <td>10039107</td>
      <td>27.303225</td>
      <td>91.7</td>
      <td>4752.32</td>
      <td>2112.464438</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot new cases vs population density
def pop_density_plot(df, save_name=None):
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.scatterplot(x='pop_density', y='new_cases', size='county_population', sizes=(10, 500),
                    hue='mask_use', data=df, ax=ax, alpha=0.9)
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.get_xaxis().set_label_text('county population density (people per square mile)')
    ax.get_yaxis().set_label_text('new cases per day (per 100,000 caapita)')
    ax.legend(labelspacing=1.5)
    ax.set_title('Correlation between the number of new cases per 100,000 capita '\
                 'and the population density\nPer US county')
    if save_name:
        ax.get_figure().savefig('output_visuals/' + save_name, dpi=300)
```


```python
# pop_density_plot(df=dfm4)
```


```python
# Removing the extreme cases
pop_density_plot(df=dfm4[(dfm4.pop_density < 4000) & (dfm4.new_cases < 150)],
                 save_name='pop_density_dfm4.png')
```


![png](https://mtlberriawsbucket.s3.us-east-2.amazonaws.com/Covid/output_92_0.png)


TAKE-AWAY: No clear correlation in between the population density of a county and the number of daily new cases. The number of cases does not increase with the population density (which was our hypothesis). A possible explaination being that the more dense areas tend to wear the mask more (as shown by the color coding), thus curbing the effect of the population density in the spread of COVID.

## Mask-Use | Population Density

Given the prevous study, the correlation seems to be more centered around the fact that counties with higher population density tend to wear the mask more. Let's check that based on the data.


```python
dfm4.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>county</th>
      <th>state</th>
      <th>new_cases_m2w</th>
      <th>county_population</th>
      <th>new_cases</th>
      <th>mask_use</th>
      <th>county_area</th>
      <th>pop_density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3036</th>
      <td>53061</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>64.692308</td>
      <td>822083</td>
      <td>7.869316</td>
      <td>91.2</td>
      <td>2196.41</td>
      <td>374.284856</td>
    </tr>
    <tr>
      <th>625</th>
      <td>17031</td>
      <td>Cook</td>
      <td>Illinois</td>
      <td>521.461538</td>
      <td>5150233</td>
      <td>10.125009</td>
      <td>88.4</td>
      <td>1635.04</td>
      <td>3149.912540</td>
    </tr>
    <tr>
      <th>221</th>
      <td>6059</td>
      <td>Orange</td>
      <td>California</td>
      <td>600.153846</td>
      <td>3175692</td>
      <td>18.898364</td>
      <td>91.0</td>
      <td>947.98</td>
      <td>3349.956750</td>
    </tr>
    <tr>
      <th>107</th>
      <td>4013</td>
      <td>Maricopa</td>
      <td>Arizona</td>
      <td>1920.230769</td>
      <td>4485414</td>
      <td>42.810558</td>
      <td>89.2</td>
      <td>9224.27</td>
      <td>486.262219</td>
    </tr>
    <tr>
      <th>210</th>
      <td>6037</td>
      <td>Los Angeles</td>
      <td>California</td>
      <td>2741.000000</td>
      <td>10039107</td>
      <td>27.303225</td>
      <td>91.7</td>
      <td>4752.32</td>
      <td>2112.464438</td>
    </tr>
  </tbody>
</table>
</div>




```python
def pop_density_mask_plot(df, save_name=None):
    """plots the mask use correlation to population density"""
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.scatterplot(x='pop_density', y='mask_use', size='county_population', sizes=(10, 500),
#                     hue='new_cases', 
                    data=df, ax=ax, alpha=1)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.get_xaxis().set_label_text('county population density (people per square mile)')
    ax.get_yaxis().set_label_text('% of people saying they wear the mask "frequently" or "always"')
    ax.legend(labelspacing=1.5)
    ax.set_title('Correlation between the population density '\
                 'and the use of mask\nPer US county')
    if save_name:
        ax.get_figure().savefig('output_visuals/' + save_name, dpi=300)
    return ax
```


```python
output_ax = pop_density_mask_plot(df=dfm4[(dfm4.pop_density < 4000) & (dfm4.new_cases < 100)], 
                     save_name='density_mask.png')
```


![png](https://mtlberriawsbucket.s3.us-east-2.amazonaws.com/Covid/output_98_0.png)


TAKE-AWAY: People living in high population density counties tend to consistently wear a mask.
People from lower density counties tend to be more distributed along the spectrum of mask usage.

## Political | Mask-Use

Gong to check if the political color of each state has an influence over its people propency to use the mask frequently.

### Get the political color of each county


```python
dfp = pd.read_csv('other_input/countypres_2000-2016.csv')
dfp.drop(labels=dfp[dfp.year != 2016].index, inplace=True)
dfp.rename(columns={'FIPS': 'fips'}, inplace=True)
dfp.fips = convertFromTypeToStr(df=dfp.fips, from_type=np.float64)
dfp['party_perc'] = (dfp.candidatevotes / dfp.totalvotes) * 100
dfp.drop(labels=dfp[dfp.fips=='NA'].index, inplace=True) 
dfp2 = dfp.loc[:, ['fips', 'party', 'party_perc']].copy()
dfp2.head(2)
```

    df converted to string





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>party</th>
      <th>party_perc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40517</th>
      <td>1001</td>
      <td>democrat</td>
      <td>23.769671</td>
    </tr>
    <tr>
      <th>40518</th>
      <td>1001</td>
      <td>republican</td>
      <td>72.766588</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfp3 = dfp2.pivot(index='fips', columns='party', values='party_perc')
dfp3.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>party</th>
      <th>NaN</th>
      <th>democrat</th>
      <th>republican</th>
    </tr>
    <tr>
      <th>fips</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10001</th>
      <td>5.705247</td>
      <td>44.707633</td>
      <td>49.587120</td>
    </tr>
    <tr>
      <th>10003</th>
      <td>5.315350</td>
      <td>62.090163</td>
      <td>32.594487</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfp3['maj_party'] = np.nan
dfp3.loc[dfp3[dfp3.democrat > dfp3.republican].index, 'maj_party'] = 'blue_state'
dfp3.loc[dfp3[dfp3.democrat < dfp3.republican].index, 'maj_party'] = 'red_state'
dfp3.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>party</th>
      <th>NaN</th>
      <th>democrat</th>
      <th>republican</th>
      <th>maj_party</th>
    </tr>
    <tr>
      <th>fips</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10001</th>
      <td>5.705247</td>
      <td>44.707633</td>
      <td>49.587120</td>
      <td>red_state</td>
    </tr>
    <tr>
      <th>10003</th>
      <td>5.315350</td>
      <td>62.090163</td>
      <td>32.594487</td>
      <td>blue_state</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfp3.drop(columns=[np.nan, 'democrat', 'republican'], inplace=True)
dfp3.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>party</th>
      <th>maj_party</th>
    </tr>
    <tr>
      <th>fips</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10001</th>
      <td>red_state</td>
    </tr>
    <tr>
      <th>10003</th>
      <td>blue_state</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merge the political color in the main df
dfm5 = pd.merge(left=dfm4, right=dfp3, on='fips', how='inner', left_index=True)
dfm5.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>county</th>
      <th>state</th>
      <th>new_cases_m2w</th>
      <th>county_population</th>
      <th>new_cases</th>
      <th>mask_use</th>
      <th>county_area</th>
      <th>pop_density</th>
      <th>maj_party</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2866</th>
      <td>53061</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>64.692308</td>
      <td>822083</td>
      <td>7.869316</td>
      <td>91.2</td>
      <td>2196.41</td>
      <td>374.284856</td>
      <td>blue_state</td>
    </tr>
    <tr>
      <th>360</th>
      <td>17031</td>
      <td>Cook</td>
      <td>Illinois</td>
      <td>521.461538</td>
      <td>5150233</td>
      <td>10.125009</td>
      <td>88.4</td>
      <td>1635.04</td>
      <td>3149.912540</td>
      <td>blue_state</td>
    </tr>
    <tr>
      <th>3054</th>
      <td>6059</td>
      <td>Orange</td>
      <td>California</td>
      <td>600.153846</td>
      <td>3175692</td>
      <td>18.898364</td>
      <td>91.0</td>
      <td>947.98</td>
      <td>3349.956750</td>
      <td>blue_state</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>4013</td>
      <td>Maricopa</td>
      <td>Arizona</td>
      <td>1920.230769</td>
      <td>4485414</td>
      <td>42.810558</td>
      <td>89.2</td>
      <td>9224.27</td>
      <td>486.262219</td>
      <td>red_state</td>
    </tr>
    <tr>
      <th>3043</th>
      <td>6037</td>
      <td>Los Angeles</td>
      <td>California</td>
      <td>2741.000000</td>
      <td>10039107</td>
      <td>27.303225</td>
      <td>91.7</td>
      <td>4752.32</td>
      <td>2112.464438</td>
      <td>blue_state</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Identify fips that were in the main df but not found in the political data
fips_nopol = set(dfm4.fips).difference(dfm5.fips)
```


```python
# Full list of states that were not found in the political data
set(dfm4[dfm4.fips.isin(fips_nopol)].state)
```




    {'Alaska'}



### Add the political color to the plot


```python
"""plots the mask use correlation to population density"""
fig, ax = plt.subplots(figsize=(12, 12))
sns.scatterplot(x='pop_density', y='mask_use', size='county_population', sizes=(10, 500),
                hue='maj_party', 
                palette=['#0015BC', '#FF0000'],
                data=dfm5[dfm5.pop_density < 4000], 
                ax=ax, alpha=1)
ax.get_yaxis().set_major_formatter(matplotlib.ticker.PercentFormatter())
ax.get_xaxis().set_label_text('county population density (people per square mile)')
ax.get_yaxis().set_label_text('% of people saying they wear the mask "frequently" or "always"')
ax.legend(labelspacing=1.5)
ax.set_title('Correlation between the population density '\
             'and the use of mask\nPer US county')
ax.get_figure().savefig('output_visuals/' + 'Population-density_Mask-use_Political-color.png', dpi=300)
```


![png](https://mtlberriawsbucket.s3.us-east-2.amazonaws.com/Covid/output_111_0.png)


TAKE-AWAY: The most populated counties are Blue, and tend to wear the mask more consistently.

### Represent as a boxplot


```python
# Create the bins for population density
# max_dens = dfm5.pop_density.max()
# rounded_max_dens = math.ceil(max_dens / 1000) * 1000
# pop_density_steps = 500
max_density_considered = 4000
bins_buckets = np.arange(0, max_density_considered + 1, 500)
bins_buckets
```




    array([   0,  500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])




```python
# Bucket the county population density into bins
dfm6 = dfm5.copy()
dfm6.drop(labels=dfm6[dfm6.pop_density > max_density_considered].index, inplace=True)
pop_dens_bins = pd.cut(x=dfm6.pop_density, bins=bins_buckets)
dfm6['pop_density_bins'] = pop_dens_bins
dfm6.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>county</th>
      <th>state</th>
      <th>new_cases_m2w</th>
      <th>county_population</th>
      <th>new_cases</th>
      <th>mask_use</th>
      <th>county_area</th>
      <th>pop_density</th>
      <th>maj_party</th>
      <th>pop_density_bins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2866</th>
      <td>53061</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>64.692308</td>
      <td>822083</td>
      <td>7.869316</td>
      <td>91.2</td>
      <td>2196.41</td>
      <td>374.284856</td>
      <td>blue_state</td>
      <td>(0, 500]</td>
    </tr>
    <tr>
      <th>360</th>
      <td>17031</td>
      <td>Cook</td>
      <td>Illinois</td>
      <td>521.461538</td>
      <td>5150233</td>
      <td>10.125009</td>
      <td>88.4</td>
      <td>1635.04</td>
      <td>3149.912540</td>
      <td>blue_state</td>
      <td>(3000, 3500]</td>
    </tr>
    <tr>
      <th>3054</th>
      <td>6059</td>
      <td>Orange</td>
      <td>California</td>
      <td>600.153846</td>
      <td>3175692</td>
      <td>18.898364</td>
      <td>91.0</td>
      <td>947.98</td>
      <td>3349.956750</td>
      <td>blue_state</td>
      <td>(3000, 3500]</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>4013</td>
      <td>Maricopa</td>
      <td>Arizona</td>
      <td>1920.230769</td>
      <td>4485414</td>
      <td>42.810558</td>
      <td>89.2</td>
      <td>9224.27</td>
      <td>486.262219</td>
      <td>red_state</td>
      <td>(0, 500]</td>
    </tr>
    <tr>
      <th>3043</th>
      <td>6037</td>
      <td>Los Angeles</td>
      <td>California</td>
      <td>2741.000000</td>
      <td>10039107</td>
      <td>27.303225</td>
      <td>91.7</td>
      <td>4752.32</td>
      <td>2112.464438</td>
      <td>blue_state</td>
      <td>(2000, 2500]</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(12, 12))
sns.boxplot(x='pop_density_bins', y='mask_use', 
#             size='county_population', sizes=(10, 500),
            hue='maj_party', 
            palette=['#9da2c9', '#ffadad'],
#             palette=None,
            data=dfm6, 
            ax=ax)
sns.swarmplot(x='pop_density_bins', y='mask_use', 
#             size='county_population', sizes=(10, 500),
            hue='maj_party', 
            palette=['#0015BC', '#FF0000'],
#             palette='Reds',
            dodge=True,
            data=dfm6, 
            ax=ax)
ax.get_yaxis().set_major_formatter(matplotlib.ticker.PercentFormatter())
ax.get_xaxis().set_label_text('county population density (people per square mile)')
ax.get_yaxis().set_label_text('% of people saying they wear the mask "frequently" or "always"')
# ax.legend(labelspacing=1.5)
ax.set_title('Correlation between the population density '\
             'and the use of mask\nPer US county')
```




    Text(0.5, 1.0, 'Correlation between the population density and the use of mask\nPer US county')




![png](https://mtlberriawsbucket.s3.us-east-2.amazonaws.com/Covid/output_116_1.png)


TAKE-AWAY: Low population density counties tend to be more Repulican and to wear the mask less consistently than Democrat counties. Let's focus more on those low denity counties in the next section.

### Low population density counties


```python
dfm7 = dfm6[dfm6.pop_density_bins == pd.Interval(0, 500, closed='right')].copy()
dfm7.pop_density_bins = '[0-500]'
dfm7
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>county</th>
      <th>state</th>
      <th>new_cases_m2w</th>
      <th>county_population</th>
      <th>new_cases</th>
      <th>mask_use</th>
      <th>county_area</th>
      <th>pop_density</th>
      <th>maj_party</th>
      <th>pop_density_bins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2866</th>
      <td>53061</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>64.692308</td>
      <td>822083</td>
      <td>7.869316</td>
      <td>91.2</td>
      <td>2196.41</td>
      <td>374.284856</td>
      <td>blue_state</td>
      <td>[0-500]</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>4013</td>
      <td>Maricopa</td>
      <td>Arizona</td>
      <td>1920.230769</td>
      <td>4485414</td>
      <td>42.810558</td>
      <td>89.2</td>
      <td>9224.27</td>
      <td>486.262219</td>
      <td>red_state</td>
      <td>[0-500]</td>
    </tr>
    <tr>
      <th>2942</th>
      <td>55025</td>
      <td>Dane</td>
      <td>Wisconsin</td>
      <td>50.307692</td>
      <td>546695</td>
      <td>9.202150</td>
      <td>87.1</td>
      <td>1238.32</td>
      <td>441.481200</td>
      <td>blue_state</td>
      <td>[0-500]</td>
    </tr>
    <tr>
      <th>3036</th>
      <td>6023</td>
      <td>Humboldt</td>
      <td>California</td>
      <td>3.538462</td>
      <td>135558</td>
      <td>2.610293</td>
      <td>93.4</td>
      <td>4052.22</td>
      <td>33.452774</td>
      <td>blue_state</td>
      <td>[0-500]</td>
    </tr>
    <tr>
      <th>3072</th>
      <td>6095</td>
      <td>Solano</td>
      <td>California</td>
      <td>75.461538</td>
      <td>447643</td>
      <td>16.857527</td>
      <td>87.6</td>
      <td>906.67</td>
      <td>493.722082</td>
      <td>blue_state</td>
      <td>[0-500]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1398</th>
      <td>30019</td>
      <td>Daniels</td>
      <td>Montana</td>
      <td>0.000000</td>
      <td>1690</td>
      <td>0.000000</td>
      <td>39.8</td>
      <td>1426.52</td>
      <td>1.184701</td>
      <td>red_state</td>
      <td>[0-500]</td>
    </tr>
    <tr>
      <th>1536</th>
      <td>31183</td>
      <td>Wheeler</td>
      <td>Nebraska</td>
      <td>0.000000</td>
      <td>783</td>
      <td>0.000000</td>
      <td>60.0</td>
      <td>575.57</td>
      <td>1.360391</td>
      <td>red_state</td>
      <td>[0-500]</td>
    </tr>
    <tr>
      <th>2883</th>
      <td>54017</td>
      <td>Doddridge</td>
      <td>West Virginia</td>
      <td>0.333333</td>
      <td>8448</td>
      <td>3.945707</td>
      <td>72.7</td>
      <td>320.48</td>
      <td>26.360459</td>
      <td>red_state</td>
      <td>[0-500]</td>
    </tr>
    <tr>
      <th>2611</th>
      <td>49055</td>
      <td>Wayne</td>
      <td>Utah</td>
      <td>0.000000</td>
      <td>2711</td>
      <td>0.000000</td>
      <td>68.9</td>
      <td>2466.47</td>
      <td>1.099142</td>
      <td>red_state</td>
      <td>[0-500]</td>
    </tr>
    <tr>
      <th>1782</th>
      <td>38001</td>
      <td>Adams</td>
      <td>North Dakota</td>
      <td>NaN</td>
      <td>2216</td>
      <td>NaN</td>
      <td>42.6</td>
      <td>988.84</td>
      <td>2.241010</td>
      <td>red_state</td>
      <td>[0-500]</td>
    </tr>
  </tbody>
</table>
<p>2843 rows × 11 columns</p>
</div>




```python
fig, ax = plt.subplots(figsize=(12, 12))
sns.swarmplot(x='pop_density_bins', y='mask_use', 
            hue='maj_party', 
            palette=['#0015BC', '#FF0000'],
            dodge=True,
            data=dfm7, 
            ax=ax)
ax.get_yaxis().set_major_formatter(matplotlib.ticker.PercentFormatter())
ax.get_xaxis().set_label_text('county population density (people per square mile)')
ax.get_yaxis().set_label_text('% of people saying they wear the mask "frequently" or "always"')
# ax.legend(labelspacing=1.5)
ax.set_title('Correlation between the political color of each county and its '\
             'population propency to wear'\
             ' the mask frequently \n(limited to low density counties, less than '\
             '500 people per square mile)')
ax.get_figure().savefig('output_visuals/' + 'Mask-use_Political_light-dens-co9unties.png', dpi=300)
```


![png](https://mtlberriawsbucket.s3.us-east-2.amazonaws.com/Covid/output_120_0.png)


TAKE-AWAY: In light population density counties (0 to 500 habitants per square mile), people are wearing the mask significantly more frequently in the democrat counties (compared to the republican counties).

## Shame Counties

Counties with low use of mask, and high number of daily new cases.


```python
dfm7[(dfm7.mask_use < 50) & (dfm7.new_cases > 40)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>county</th>
      <th>state</th>
      <th>new_cases_m2w</th>
      <th>county_population</th>
      <th>new_cases</th>
      <th>mask_use</th>
      <th>county_area</th>
      <th>pop_density</th>
      <th>maj_party</th>
      <th>pop_density_bins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2255</th>
      <td>47041</td>
      <td>DeKalb</td>
      <td>Tennessee</td>
      <td>11.230769</td>
      <td>20490</td>
      <td>54.810977</td>
      <td>46.0</td>
      <td>328.98</td>
      <td>62.283421</td>
      <td>red_state</td>
      <td>[0-500]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1027</td>
      <td>Clay</td>
      <td>Alabama</td>
      <td>6.615385</td>
      <td>13235</td>
      <td>49.984017</td>
      <td>49.8</td>
      <td>606.00</td>
      <td>21.839934</td>
      <td>red_state</td>
      <td>[0-500]</td>
    </tr>
    <tr>
      <th>632</th>
      <td>19187</td>
      <td>Webster</td>
      <td>Iowa</td>
      <td>16.076923</td>
      <td>35904</td>
      <td>44.777526</td>
      <td>46.3</td>
      <td>718.05</td>
      <td>50.002089</td>
      <td>red_state</td>
      <td>[0-500]</td>
    </tr>
    <tr>
      <th>573</th>
      <td>19069</td>
      <td>Franklin</td>
      <td>Iowa</td>
      <td>4.461538</td>
      <td>10070</td>
      <td>44.305248</td>
      <td>44.7</td>
      <td>583.01</td>
      <td>17.272431</td>
      <td>red_state</td>
      <td>[0-500]</td>
    </tr>
    <tr>
      <th>2254</th>
      <td>47039</td>
      <td>Decatur</td>
      <td>Tennessee</td>
      <td>5.615385</td>
      <td>11663</td>
      <td>48.147000</td>
      <td>48.6</td>
      <td>344.91</td>
      <td>33.814618</td>
      <td>red_state</td>
      <td>[0-500]</td>
    </tr>
    <tr>
      <th>1784</th>
      <td>38005</td>
      <td>Benson</td>
      <td>North Dakota</td>
      <td>4.153846</td>
      <td>6832</td>
      <td>60.799856</td>
      <td>39.9</td>
      <td>1439.27</td>
      <td>4.746851</td>
      <td>red_state</td>
      <td>[0-500]</td>
    </tr>
  </tbody>
</table>
</div>



Counties with low mask-use (<50%), still having a way abover average number of new cases (> 40 new cases per day per 100,000 habitants, when the mean is 16)

## Mask-Use | Percentage of total population


```python
dfm7 = dfm3.copy()
dfm7['pop_perc'] = (dfm7.county_population / dfm7.county_population.sum()) * 100
dfm7.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>county</th>
      <th>state</th>
      <th>new_cases_m2w</th>
      <th>county_population</th>
      <th>new_cases</th>
      <th>mask_use</th>
      <th>pop_perc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>53061</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>64.692308</td>
      <td>822083</td>
      <td>7.869316</td>
      <td>91.2</td>
      <td>0.257033</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17031</td>
      <td>Cook</td>
      <td>Illinois</td>
      <td>521.461538</td>
      <td>5150233</td>
      <td>10.125009</td>
      <td>88.4</td>
      <td>1.610278</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6059</td>
      <td>Orange</td>
      <td>California</td>
      <td>600.153846</td>
      <td>3175692</td>
      <td>18.898364</td>
      <td>91.0</td>
      <td>0.992916</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4013</td>
      <td>Maricopa</td>
      <td>Arizona</td>
      <td>1920.230769</td>
      <td>4485414</td>
      <td>42.810558</td>
      <td>89.2</td>
      <td>1.402415</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6037</td>
      <td>Los Angeles</td>
      <td>California</td>
      <td>2741.000000</td>
      <td>10039107</td>
      <td>27.303225</td>
      <td>91.7</td>
      <td>3.138840</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create mask-use buckets
mask_use_bins = np.arange(20, 110, 10)
dfm7['mask_use_buckets'] = pd.cut(x=dfm7.mask_use, bins=mask_use_bins)
dfm7.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fips</th>
      <th>county</th>
      <th>state</th>
      <th>new_cases_m2w</th>
      <th>county_population</th>
      <th>new_cases</th>
      <th>mask_use</th>
      <th>pop_perc</th>
      <th>mask_use_buckets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>53061</td>
      <td>Snohomish</td>
      <td>Washington</td>
      <td>64.692308</td>
      <td>822083</td>
      <td>7.869316</td>
      <td>91.2</td>
      <td>0.257033</td>
      <td>(90, 100]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17031</td>
      <td>Cook</td>
      <td>Illinois</td>
      <td>521.461538</td>
      <td>5150233</td>
      <td>10.125009</td>
      <td>88.4</td>
      <td>1.610278</td>
      <td>(80, 90]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6059</td>
      <td>Orange</td>
      <td>California</td>
      <td>600.153846</td>
      <td>3175692</td>
      <td>18.898364</td>
      <td>91.0</td>
      <td>0.992916</td>
      <td>(90, 100]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4013</td>
      <td>Maricopa</td>
      <td>Arizona</td>
      <td>1920.230769</td>
      <td>4485414</td>
      <td>42.810558</td>
      <td>89.2</td>
      <td>1.402415</td>
      <td>(80, 90]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6037</td>
      <td>Los Angeles</td>
      <td>California</td>
      <td>2741.000000</td>
      <td>10039107</td>
      <td>27.303225</td>
      <td>91.7</td>
      <td>3.138840</td>
      <td>(90, 100]</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfm7s = dfm7.loc[:, ['mask_use_buckets', 'pop_perc']]
dfm7s.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mask_use_buckets</th>
      <th>pop_perc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(90, 100]</td>
      <td>0.257033</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(80, 90]</td>
      <td>1.610278</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(90, 100]</td>
      <td>0.992916</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(80, 90]</td>
      <td>1.402415</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(90, 100]</td>
      <td>3.138840</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfm7sg = dfm7s.groupby(by='mask_use_buckets').sum()
dfm7sg.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pop_perc</th>
    </tr>
    <tr>
      <th>mask_use_buckets</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(20, 30]</th>
      <td>0.013659</td>
    </tr>
    <tr>
      <th>(30, 40]</th>
      <td>0.073255</td>
    </tr>
    <tr>
      <th>(40, 50]</th>
      <td>0.880558</td>
    </tr>
    <tr>
      <th>(50, 60]</th>
      <td>3.187771</td>
    </tr>
    <tr>
      <th>(60, 70]</th>
      <td>8.362238</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x=dfm7sg.index.values, y=dfm7sg.pop_perc, palette='Greens')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.PercentFormatter())
ax.get_xaxis().set_label_text('% of people wearing the mask frequently')
ax.get_yaxis().set_label_text('% of the total popualtion of the US')
ax.set_title('Breakdown of the US population per frequent mask-use %')
# ax.get_figure().savefig('output_visuals/' + 'Mask-use_Political_light-dens-co9unties.png', dpi=300)
```




    Text(0.5, 1.0, 'Breakdown of the US population per frequent mask-use %')




![png](https://mtlberriawsbucket.s3.us-east-2.amazonaws.com/Covid/output_131_1.png)



```python

```


```python

```


```python

```

## Sates ranked per number of cases


```python
last_timestamp = df.index.sort_values()[-1]
```


```python

df[last_timestamp:].sort_values(by='cases',ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>state</th>
      <th>fips</th>
      <th>cases</th>
      <th>deaths</th>
      <th>Abbreviation</th>
      <th>Postal Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8153</th>
      <td>2020-07-28</td>
      <td>Northern Mariana Islands</td>
      <td>69</td>
      <td>40</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_summary = df.loc[last_timestamp:, ['cases', 'deaths']].sum().copy()
df_summary.index.values
```




    array(['cases', 'deaths'], dtype=object)




```python
fig, ax = plt.subplots()
rects = ax.bar(x=[0, 0.3], height=df_summary.values, width=0.2, tick_label=df_summary.index.values)
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,p: format(int(x),',')))


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects)

fig.tight_layout()

```


![png](https://mtlberriawsbucket.s3.us-east-2.amazonaws.com/Covid/output_139_0.png)


### Mortality rate


```python
df_summary['deaths'] / df_summary['cases']
```




    0.05



## Number of New Cases per day (for a list of states)


```python
dft = df.copy()
# Make it a time series
dft.set_index(keys='date', drop=True, inplace=True)
dft
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>fips</th>
      <th>cases</th>
      <th>deaths</th>
      <th>Abbreviation</th>
      <th>Postal Code</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-21</th>
      <td>Washington</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>Wash.</td>
      <td>WA</td>
    </tr>
    <tr>
      <th>2020-01-22</th>
      <td>Washington</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>Wash.</td>
      <td>WA</td>
    </tr>
    <tr>
      <th>2020-01-23</th>
      <td>Washington</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>Wash.</td>
      <td>WA</td>
    </tr>
    <tr>
      <th>2020-01-24</th>
      <td>Washington</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>Wash.</td>
      <td>WA</td>
    </tr>
    <tr>
      <th>2020-01-25</th>
      <td>Washington</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>Wash.</td>
      <td>WA</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-07-24</th>
      <td>Northern Mariana Islands</td>
      <td>69</td>
      <td>38</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-07-25</th>
      <td>Northern Mariana Islands</td>
      <td>69</td>
      <td>40</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-07-26</th>
      <td>Northern Mariana Islands</td>
      <td>69</td>
      <td>40</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-07-27</th>
      <td>Northern Mariana Islands</td>
      <td>69</td>
      <td>40</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-07-28</th>
      <td>Northern Mariana Islands</td>
      <td>69</td>
      <td>40</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>8154 rows × 6 columns</p>
</div>




```python
states_list = ["New York","New Jersey","California","Michigan","Massachusetts","Florida","Washington","Illinois","Pennsylvania","Louisiana", "Texas"]
len(states_list)
```




    11




```python
df_states_list = []
# Iterate over all states, create the df and append to the list
for i,state in enumerate(states_list):
    # create a local copy of the df
    df_state = dft[dft.state == state].copy()
    # calculate the number of new cases per day
    df_state['new cases'] = df_state['cases'] - df_state['cases'].shift(1)
    df_state.fillna(0,inplace=True)
    # calculate SMA
    df_state['SMA'] = ( df_state['new cases'] + df_state['new cases'].shift(1) + df_state['new cases'].shift(2) )*1/3
    # append the df of that state to the list
    df_states_list.append(df_state)
```


```python
df_states_list[10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>fips</th>
      <th>cases</th>
      <th>deaths</th>
      <th>Abbreviation</th>
      <th>Postal Code</th>
      <th>new cases</th>
      <th>SMA</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-02-12</th>
      <td>Texas</td>
      <td>48</td>
      <td>1</td>
      <td>0</td>
      <td>Tex.</td>
      <td>TX</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-02-13</th>
      <td>Texas</td>
      <td>48</td>
      <td>2</td>
      <td>0</td>
      <td>Tex.</td>
      <td>TX</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-02-14</th>
      <td>Texas</td>
      <td>48</td>
      <td>2</td>
      <td>0</td>
      <td>Tex.</td>
      <td>TX</td>
      <td>0.0</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>2020-02-15</th>
      <td>Texas</td>
      <td>48</td>
      <td>2</td>
      <td>0</td>
      <td>Tex.</td>
      <td>TX</td>
      <td>0.0</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>2020-02-16</th>
      <td>Texas</td>
      <td>48</td>
      <td>2</td>
      <td>0</td>
      <td>Tex.</td>
      <td>TX</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-07-24</th>
      <td>Texas</td>
      <td>48</td>
      <td>383662</td>
      <td>4876</td>
      <td>Tex.</td>
      <td>TX</td>
      <td>7558.0</td>
      <td>9356.000000</td>
    </tr>
    <tr>
      <th>2020-07-25</th>
      <td>Texas</td>
      <td>48</td>
      <td>391609</td>
      <td>5002</td>
      <td>Tex.</td>
      <td>TX</td>
      <td>7947.0</td>
      <td>8349.333333</td>
    </tr>
    <tr>
      <th>2020-07-26</th>
      <td>Texas</td>
      <td>48</td>
      <td>395738</td>
      <td>5090</td>
      <td>Tex.</td>
      <td>TX</td>
      <td>4129.0</td>
      <td>6544.666667</td>
    </tr>
    <tr>
      <th>2020-07-27</th>
      <td>Texas</td>
      <td>48</td>
      <td>402295</td>
      <td>6292</td>
      <td>Tex.</td>
      <td>TX</td>
      <td>6557.0</td>
      <td>6211.000000</td>
    </tr>
    <tr>
      <th>2020-07-28</th>
      <td>Texas</td>
      <td>48</td>
      <td>412744</td>
      <td>6515</td>
      <td>Tex.</td>
      <td>TX</td>
      <td>10449.0</td>
      <td>7045.000000</td>
    </tr>
  </tbody>
</table>
<p>168 rows × 8 columns</p>
</div>



### PLOT OF NEW CASES (for the 4 most impacted states)


```python
def my_new_cases_plotter(df_list,states_list):

    nb_columns_fig = 2
    nb_rows_fig = 6
    
    fig, ax_arr = plt.subplots(nb_rows_fig, nb_columns_fig)  # create a figure with a 'rows x columns' grid of axes
    fig.set_size_inches(16,24)
    fig.suptitle("New cases registered per day")
    
    for df_index in range(len(df_list)):  # iterate over all the data frames to plot
    
        i_fig = int((np.floor(df_index / nb_columns_fig)))  # row position of the axes on that given figure
        j_fig = int((df_index % nb_columns_fig))  # column position of the axes on that given figure

        df = df_list[df_index].loc['20200301':]  # df to plot at that position

        ax_arr[i_fig,j_fig].bar(x=df['new cases'].index,height=df['new cases'],color="#900C3F")
        ax_arr[i_fig,j_fig].plot(df['new cases'].index,df['SMA'],color="#FFC300")

        ax_arr[i_fig,j_fig].xaxis.set_major_locator(WeekdayLocator(MONDAY))
#         ax_arr[i_fig,j_fig].xaxis.set_minor_locator(DayLocator())
        ax_arr[i_fig,j_fig].xaxis.set_major_formatter(DateFormatter('%d'))
#         ax_arr[i_fig,j_fig].xaxis.set_minor_formatter(DateFormatter('%d'))
        ax_arr[i_fig,j_fig].set_title(states_list[df_index])
    return fig
        
```


```python
fig = my_new_cases_plotter(df_states_list,states_list)
```


![png](https://mtlberriawsbucket.s3.us-east-2.amazonaws.com/Covid/output_149_0.png)



```python
# fig.savefig('covid_per_US_state.png')
```
