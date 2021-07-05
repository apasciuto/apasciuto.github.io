---
layout: post
title:  "K-Nearest Neighbors: Predicting Car Prices"
cover: "/assets/media/car-prices/car.png"
permalink: /:categories/:year/:month/:day/K-Nearest-Neighbors
excerpt_separator: <!--end-->
categories: projects
date: 2017-09-04
---

Using k-Nearest Neighbors (k-NN) to Predict Car Prices.

<!--end-->


# Car Prices: K-Nearest Neighbors

**Using k-Nearest Neighbors (k-**NN**) to Predict Car Prices.**  

# 1. The Data

The [data](https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data) that we will be using is compiled from the [University of California Irvine's Website](https://archive.ics.uci.edu/ml/datasets/automobile) and contains the following car attributes:  

| Columns       | Description   |  
| :------------- |:------------- |  
| **symboling**   | -3, -2, -1, 0, 1, 2, 3 |
| **normalized-losses**   | continuous from 65 to 256     |
| **make**   | alfa-romero, audi, bmw, chevrolet, dodge, honda, isuzu, jaguar, mazda, mercedes-benz, mercury, mitsubishi, nissan, peugot, plymouth, porsche, renault, saab, subaru, toyota, volkswagen, volvo  |
| **fuel-type**   | diesel, gas    |
| **aspiration**   | std, turbo    |
| **num-of-doors**   | four, two  |
| **body-style**   | hardtop, wagon, sedan, hatchback, convertible  |
| **drive-wheels**   | 4wd, fwd, rwd |
| **engine-location**   | front, rear  |
| **wheel-base**   | continuous from 86.6 120.9 |
| **length**   | continuous from 141.1 to 208.1   |
| **width**   | continuous from 60.3 to 72.3  |
| **height**   | continuous from 47.8 to 59.8  |
| **curb-weight**   | continuous from 1488 to 4066   |
| **engine-type**   | dohc, dohcv, l, ohc, ohcf, ohcv, rotor |
| **num-of-cylinders**   | eight, five, four, six, three, twelve, two |
| **engine-size**   | continuous from 61 to 326 |
| **fuel-system**   | 1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi |
| **bore**   | continuous from 2.54 to 3.94 |
| **stroke**   | continuous from 2.07 to 4.17 |
| **compression-ratio**   | continuous from 7 to 23 |
| **horsepower**   | continuous from 48 to 288 |
| **peak-rpm**   | continuous from 4150 to 6600 |
| **city-mpg**   | continuous from 13 to 49 |
| **highway-mpg**   | continuous from 16 to 54 |
| **price**   | continuous from 5118 to 45400 |

The `imports-85.data` data set columns do not match the columns from our [Dataset's Documentation](https://archive.ics.uci.edu/ml/datasets/automobile) and we will need to provide additional parameters to effectively work with our data.


```python
import numpy as np
import pandas as pd

pd.options.display.max_columns = 99
```


```python
cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars = pd.read_csv('imports-85.data', names=cols)
```


```python
cars.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>engine-type</th>
      <th>num-of-cylinders</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-rate</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548</td>
      <td>dohc</td>
      <td>four</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548</td>
      <td>dohc</td>
      <td>four</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>171.2</td>
      <td>65.5</td>
      <td>52.4</td>
      <td>2823</td>
      <td>ohcv</td>
      <td>six</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>176.6</td>
      <td>66.2</td>
      <td>54.3</td>
      <td>2337</td>
      <td>ohc</td>
      <td>four</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>176.6</td>
      <td>66.4</td>
      <td>54.3</td>
      <td>2824</td>
      <td>ohc</td>
      <td>five</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Data Cleaning

After observing the data set above we can see that the `normalized-losses` column contains missing values that are represented by `?`. Let's replace these values and look for the presence of missing values in other numeric columns. We will also need to normalize the values in all numeric columns if we want to use them for predicitive modeling.

### 2.1 Selecting Continuous Values

We will select only the columns with continuous values based off of our [Dataset's Documentation](https://archive.ics.uci.edu/ml/datasets/automobile)


```python
continuous_values_cols = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
numeric_cars = cars[continuous_values_cols]
```


```python
numeric_cars.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>normalized-losses</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-rate</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>?</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>?</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>?</td>
      <td>94.5</td>
      <td>171.2</td>
      <td>65.5</td>
      <td>52.4</td>
      <td>2823</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>164</td>
      <td>99.8</td>
      <td>176.6</td>
      <td>66.2</td>
      <td>54.3</td>
      <td>2337</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>4</th>
      <td>164</td>
      <td>99.4</td>
      <td>176.6</td>
      <td>66.4</td>
      <td>54.3</td>
      <td>2824</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2 Replacing Values

Using the [DataFrame.replace()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.replace.html) method we can replace all of the `?` values with the `numpy.nan` missing value:


```python
numeric_cars = numeric_cars.replace('?', np.nan)
numeric_cars.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>normalized-losses</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-rate</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>94.5</td>
      <td>171.2</td>
      <td>65.5</td>
      <td>52.4</td>
      <td>2823</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>164</td>
      <td>99.8</td>
      <td>176.6</td>
      <td>66.2</td>
      <td>54.3</td>
      <td>2337</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>4</th>
      <td>164</td>
      <td>99.4</td>
      <td>176.6</td>
      <td>66.4</td>
      <td>54.3</td>
      <td>2824</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
  </tbody>
</table>
</div>



After replacing the `?` values, we need to determine the columns that need to be converted to numeric types. We will then use the [DataFrame.astype()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.astype.html) method to convert the column types:

### 2.3 Converting Values


```python
numeric_cars = numeric_cars.astype('float')
numeric_cars.isnull().sum()
```




    normalized-losses    41
    wheel-base            0
    length                0
    width                 0
    height                0
    curb-weight           0
    bore                  4
    stroke                4
    compression-rate      0
    horsepower            2
    peak-rpm              2
    city-mpg              0
    highway-mpg           0
    price                 4
    dtype: int64



### 2.4 Columns with Missing Values

We observe that the following columns have missing values:  

- `41` rows have a missing value for the `normalized-losses` column  
- `4` rows have a missing value for the `bore` column  
- `4` rows have a missing value for the `stroke` column  
- `2` rows have a missing value for the `horsepower` column  
- `2` rows have a missing value for the `peak-rpm` column  
- `4` rows have a missing value for the `price` column  

We need to determine the correct approach to handling these missing values so that we don't interfer with the integrity of the data.

### 2.5 Identifying the Target Column and Handling Missing Values

Because the `price` column is what we want to predict, let's remove any rows with missing `price` values:



```python
numeric_cars = numeric_cars.dropna(subset=['price'])
numeric_cars.isnull().sum()
```




    normalized-losses    37
    wheel-base            0
    length                0
    width                 0
    height                0
    curb-weight           0
    bore                  4
    stroke                4
    compression-rate      0
    horsepower            2
    peak-rpm              2
    city-mpg              0
    highway-mpg           0
    price                 0
    dtype: int64



We will replace missing values in the other columns using the column's mean (average) value:


```python
numeric_cars = numeric_cars.fillna(numeric_cars.mean())
```

Let's confirm that there are no more missing values:


```python
numeric_cars.isnull().sum()
```




    normalized-losses    0
    wheel-base           0
    length               0
    width                0
    height               0
    curb-weight          0
    bore                 0
    stroke               0
    compression-rate     0
    horsepower           0
    peak-rpm             0
    city-mpg             0
    highway-mpg          0
    price                0
    dtype: int64



### 2.6 Normalizing Columns

Finally, we can normalize all columnns to range from `0` to `1` except our target column (`price`).


```python
price_col = numeric_cars['price']
numeric_cars = (numeric_cars.max() - numeric_cars)/(numeric_cars.max())
numeric_cars['price'] = price_col
```

# 3. Univariate Model

We will start with some univariate k-nearest neighbors models and move to more complex ones to help us structure our code workflow and better understand the features.

We will create a function named `knn_train_test()` that encapsulates the traing and simple validation process. The function will:  
- Split the data set into a training and test set
- Instantiate the KNeighborsRegressor class, fit the model on the training set, and make predictions on the test set
- Calculate the RMSE and return that value


```python
def knn_train_test(train_col, target_col, df):
    np.random.seed(1)
        
    # Randomize order of rows in DataFrame
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set
    train_df = rand_df.iloc[0:last_train_row]
    # Select the second half and set as test set
    test_df = rand_df.iloc[last_train_row:]
    
    # train, and test a univariate model using the following k values (1, 3, 5, 7, 9)
    k_values = [1,3,5,7,9]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[[train_col]], train_df[target_col])

        # Make predictions using model
        predicted_labels = knn.predict(test_df[[train_col]])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses

k_rmse_results = {}

# For each column (minus `price`), train a model, return RMSE value and add to the dictionary `rmse_results`
train_cols = numeric_cars.columns.drop('price')
for col in train_cols:
    rmse_val = knn_train_test(col, 'price', numeric_cars)
    k_rmse_results[col] = rmse_val

k_rmse_results
```




    {'bore': {1: 7496.1492312406444,
      3: 6936.9888741632003,
      5: 6816.8537123691885,
      7: 7062.0613050538341,
      9: 6869.7274373649016},
     'city-mpg': {1: 4540.3610032247389,
      3: 4662.4683767438482,
      5: 4729.6734209992692,
      7: 5099.2742894698586,
      9: 4999.2917237740958},
     'compression-rate': {1: 9024.9026779536325,
      3: 7033.5529229950389,
      5: 6736.676353123451,
      7: 7459.1131944220724,
      9: 7219.385481303907},
     'curb-weight': {1: 5518.8832374058084,
      3: 5048.6077260366692,
      5: 4437.9343946355393,
      7: 4369.3490898512136,
      9: 4632.2055452210743},
     'height': {1: 9108.4718365936551,
      3: 8049.9871472883196,
      5: 7487.6525188849646,
      7: 7753.7974180840583,
      9: 7695.632426557866},
     'highway-mpg': {1: 5270.360471073066,
      3: 4618.1866223408379,
      5: 4579.0372499290315,
      7: 4914.2600028726101,
      9: 5181.9124189636359},
     'horsepower': {1: 3749.5962185254293,
      3: 3964.9503610053594,
      5: 4007.4723516831596,
      7: 4391.4816735297054,
      9: 4505.1886320053109},
     'length': {1: 5291.7851645472883,
      3: 5267.2167776785409,
      5: 5382.6711551381659,
      7: 5396.362242025737,
      9: 5420.5479164322587},
     'normalized-losses': {1: 7906.5941410250143,
      3: 6712.8733553798356,
      5: 7635.1704160923791,
      7: 7870.6510032392407,
      9: 8221.5784655443185},
     'peak-rpm': {1: 9825.559283202294,
      3: 8025.1729800507092,
      5: 7498.7464749413657,
      7: 7296.5172664110205,
      9: 7239.4781688794701},
     'stroke': {1: 7282.3488587810798,
      3: 7664.9840308065386,
      5: 8078.4912887356768,
      7: 7754.4838594616886,
      9: 7723.9131538450647},
     'wheel-base': {1: 5964.6822353178914,
      3: 5246.472910232148,
      5: 5527.6824887322919,
      7: 5485.6830335257237,
      9: 5734.4339857054465},
     'width': {1: 4453.161424568767,
      3: 4697.2871145506588,
      5: 4644.8984285434217,
      7: 4562.1341847495605,
      9: 4643.8823393933362}}



Below we will visualize the results using a line plot:


```python
import matplotlib.pyplot as plt
%matplotlib inline

for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    
    plt.plot(x,y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')
```


![png](/assets/media/car-prices/CarPrices_28_0.png)


# 4. Multivariate Model

Compute average RMSE across different `k` values for each feature:


```python
feature_avg_rmse = {}
for k,v in k_rmse_results.items():
    avg_rmse = np.mean(list(v.values()))
    feature_avg_rmse[k] = avg_rmse
series_avg_rmse = pd.Series(feature_avg_rmse)
series_avg_rmse.sort_values()
```




    horsepower           4123.737847
    width                4600.272698
    curb-weight          4801.395999
    city-mpg             4806.213763
    highway-mpg          4912.751353
    length               5351.716651
    wheel-base           5591.790931
    bore                 7036.356112
    compression-rate     7494.726126
    normalized-losses    7669.373476
    stroke               7700.844238
    peak-rpm             7977.094835
    height               8019.108269
    dtype: float64



We can now modify the `knn_train_test()` function to work with multiple columns:


```python
def knn_train_test(train_cols, target_col, df):
    np.random.seed(1)
    
    # Randomize order of rows in DataFrame
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set
    train_df = rand_df.iloc[0:last_train_row]
    # Select the second half and set as test set
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [5]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[train_cols], train_df[target_col])

        # Make predictions using model
        predicted_labels = knn.predict(test_df[train_cols])

        # Calculate and return RMSE
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses

k_rmse_results = {}

# Use the best 2 features to train and test a multivariate k-nearest neighbors model using the default `k` value
two_best_features = ['horsepower', 'width']
rmse_val = knn_train_test(two_best_features, 'price', numeric_cars)
k_rmse_results["two best features"] = rmse_val

# Use the best 3 features to train and test a multivariate k-nearest neighbors model using the default `k` value
three_best_features = ['horsepower', 'width', 'curb-weight']
rmse_val = knn_train_test(three_best_features, 'price', numeric_cars)
k_rmse_results["three best features"] = rmse_val

# Use the best 4 features to train and test a multivariate k-nearest neighbors model using the default `k` value
four_best_features = ['horsepower', 'width', 'curb-weight', 'city-mpg']
rmse_val = knn_train_test(four_best_features, 'price', numeric_cars)
k_rmse_results["four best features"] = rmse_val

# Use the best 5 features to train and test a multivariate k-nearest neighbors model using the default `k` value
five_best_features = ['horsepower', 'width', 'curb-weight' , 'city-mpg' , 'highway-mpg']
rmse_val = knn_train_test(five_best_features, 'price', numeric_cars)
k_rmse_results["five best features"] = rmse_val

# Use the best 6 features to train and test a multivariate k-nearest neighbors model using the default `k` value
six_best_features = ['horsepower', 'width', 'curb-weight' , 'city-mpg' , 'highway-mpg', 'length']
rmse_val = knn_train_test(six_best_features, 'price', numeric_cars)
k_rmse_results["six best features"] = rmse_val

# Display all of the RMSE values
k_rmse_results
```




    {'five best features': {5: 3346.6737097607775},
     'four best features': {5: 3232.1036292326721},
     'six best features': {5: 3398.1290113563641},
     'three best features': {5: 3212.5596306057919},
     'two best features': {5: 3681.3980922556266}}



# 5. Hyperparameter Tuning

We will now optimize the top three models that performed the best, varying the hyperparameter value from `1` to `25`.


```python
def knn_train_test(train_cols, target_col, df):
    np.random.seed(1)
    
    # Randomize order of rows in DataFrame
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set
    train_df = rand_df.iloc[0:last_train_row]
    # Select the second half and set as test set
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [i for i in range(1, 25)]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[train_cols], train_df[target_col])

        # Make predictions using model
        predicted_labels = knn.predict(test_df[train_cols])

        # Calculate and return RMSE
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses

k_rmse_results = {}

three_best_features = ['horsepower', 'width', 'curb-weight']
rmse_val = knn_train_test(three_best_features, 'price', numeric_cars)
k_rmse_results["three best features"] = rmse_val

four_best_features = ['horsepower', 'width', 'curb-weight', 'city-mpg']
rmse_val = knn_train_test(four_best_features, 'price', numeric_cars)
k_rmse_results["four best features"] = rmse_val

five_best_features = ['horsepower', 'width', 'curb-weight' , 'city-mpg' , 'highway-mpg']
rmse_val = knn_train_test(five_best_features, 'price', numeric_cars)
k_rmse_results["five best features"] = rmse_val

k_rmse_results
```




    {'five best features': {1: 2561.7319037195625,
      2: 2567.2749455482176,
      3: 2949.9007889192553,
      4: 3074.6091106298891,
      5: 3346.6737097607775,
      6: 3686.4646211770864,
      7: 3907.1959982578019,
      8: 4104.0339873177718,
      9: 4335.7141974258602,
      10: 4463.6007084810435,
      11: 4444.0259889090448,
      12: 4534.547516044051,
      13: 4638.5257014541967,
      14: 4686.7680627393893,
      15: 4676.6172318274348,
      16: 4706.4889916373404,
      17: 4714.757468354599,
      18: 4724.0179262108768,
      19: 4780.0364569672583,
      20: 4790.8654014852591,
      21: 4788.4429142051176,
      22: 4820.2560355653704,
      23: 4823.6246116515467,
      24: 4830.7715122893824},
     'four best features': {1: 3135.5489073677436,
      2: 2514.1812009849527,
      3: 2788.5519417420178,
      4: 2917.4679936225316,
      5: 3232.1036292326721,
      6: 3566.725419074407,
      7: 3834.9804809872821,
      8: 3927.3952487590609,
      9: 4078.9765839753827,
      10: 4199.8376270003955,
      11: 4345.0069904611819,
      12: 4451.3870113027624,
      13: 4550.1634683008278,
      14: 4591.5340160428832,
      15: 4630.3996426828098,
      16: 4711.9117982858279,
      17: 4692.3372730081592,
      18: 4709.1872236435829,
      19: 4698.1962740829795,
      20: 4738.5487814580347,
      21: 4727.3518464816807,
      22: 4719.3369599341022,
      23: 4707.9563401268824,
      24: 4753.4193738950999},
     'three best features': {1: 3308.7499419294022,
      2: 3044.812909435545,
      3: 3042.2117028741623,
      4: 2958.964739955848,
      5: 3212.5596306057919,
      6: 3542.3007736748041,
      7: 3801.5597829031262,
      8: 4007.7501484785639,
      9: 4074.3452185932656,
      10: 4225.0494506919176,
      11: 4338.8991649386644,
      12: 4428.0841388589351,
      13: 4496.3621365502913,
      14: 4540.1357252028592,
      15: 4614.0272979737174,
      16: 4654.474275823789,
      17: 4714.0580949648638,
      18: 4645.9886513064885,
      19: 4628.211244787356,
      20: 4665.0992005704829,
      21: 4648.5009310888045,
      22: 4610.0134050293573,
      23: 4642.8367354686252,
      24: 4669.5676777327653}}



Let's plot the resulting RMSE values:


```python
for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    
    plt.plot(x,y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')
```


![png](/assets/media/car-prices/CarPrices_37_0.png)


# 6. Recommendations

- Modify the `knn_train_test()` function to use k-fold cross validation instead of test/train validation
- Modify the `knn_train_test()` function to perform the data cleaning as well
