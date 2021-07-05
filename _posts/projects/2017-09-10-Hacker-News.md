---
layout: post
title:  "Natural Language Processing: Hacker News"
cover: "/assets/media/hacker-news/hacker-news.png"
permalink: /:categories/:year/:month/:day/Natural-Language-Processing
excerpt_separator: <!--end-->
categories: projects
date: 2017-09-10
---

Analyzing stories from Hacker News using Natural Language Processing (NLP).

<!--end-->


# Hacker News: Natural Language Processing

**Analyzing stories from Hacker News using Natural Language Processing (NLP).**  

Natural Language Processing (NLP) is the study of enabling computers to understand human languages. This field may involve teaching computers to automatically score essays, infer grammatical rules, or determine the emotions associated with text.  

In this project we will be analyzing stories from Hacker News using NLP. We will be predicting the number of upvotes the articles received, based on their headlines. Because upvotes are an indicator of popularity, we'll discover which types of articles tend to be the most popular.

# 1. About Hacker News

[Hacker News](https://news.ycombinator.com/) is a community where users can submit articles, and other users can upvote those articles. The articles with the most upvotes make it to the front page, where they're more visible to the community.

# 2. The Data

Our data set was collected using the Hacker News API to scrape the data and consists of submissions users made to Hacker News from 2006 to 2015.

`3000` rows have been sampled from the data randomly, and all of the unnecessary columns have been removed. Our data now only has four columns:  

| Columns       | Description   |
| ------------- |:------------- |
| **submission_time**   | When the article was submitted |
| **upvotes**   | The number of upvotes the article received     |
| **url**   | The base URL of the article    |
| **headlines**   | The article's headline    |


```python
import numpy as np
import pandas as pd

submissions = pd.read_csv("sel_hn_stories.csv")

submissions.columns = ["submission_time", "upvotes", "url", "headline"]
submissions = submissions.dropna()

submissions.head()
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
      <th>submission_time</th>
      <th>upvotes</th>
      <th>url</th>
      <th>headline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-02-17T16:57:59Z</td>
      <td>1</td>
      <td>blog.jonasbandi.net</td>
      <td>Software: Sadly we did adopt from the construc...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-02-04T02:36:30Z</td>
      <td>1</td>
      <td>blogs.wsj.com</td>
      <td>Google’s Stock Split Means More Control for L...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-10-26T07:11:29Z</td>
      <td>1</td>
      <td>threatpost.com</td>
      <td>SSL DOS attack tool released exploiting negoti...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-04-03T15:43:44Z</td>
      <td>67</td>
      <td>algorithm.com.au</td>
      <td>Immutability and Blocks Lambdas and Closures</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-13T16:49:20Z</td>
      <td>1</td>
      <td>winmacsofts.com</td>
      <td>Comment optimiser la vitesse de Wordpress?</td>
    </tr>
  </tbody>
</table>
</div>



# 3. Tokenizing Headlines
Our goal is to train a linear regression algorithm that predicts the number of upvotes a headline would receive. To do this, we will need to convert each headline into a numerical representation and will utilize the `bag of words` model. A [bag of words model](https://en.wikipedia.org/wiki/Bag-of-words_model) represents each piece of text as a numerical vector.

The first step in creating a bag of words model is [tokenization](https://en.wikipedia.org/wiki/Tokenization). In tokenization, all we are doing is splitting each sentence into a list of individual words, or tokens. The split occurs on the space character (`" "`).


```python
tokenized_headlines = []
for item in submissions["headline"]:
    tokenized_headlines.append(item.split(" "))
```

# 4. Preprocessing Tokens to Increase Accuracy

We will process our tokens to make our predictions more accurate. We will need to convert variations of the same word, for example: `China`, `China.`, `china`, so that they're consistent. 


```python
clean_tokenized = []
punctuation = [",", ":", ";", ".", "'", '"', "’", "?", "/", "-", "+", "&", "(", ")"]
# Loop through each item in tokenized_headlines, which is a list of lists
for item in tokenized_headlines:
    tokens = []
    for token in item:
        # Convert each individual token to lowercase
        token = token.lower()
        # Remove all of the items in the punctuation list from each individual token
        for punc in punctuation:
            token = token.replace(punc, "")
        tokens.append(token)
    # Append the clean list to clean_tokenized
    clean_tokenized.append(tokens)
```

`Clean_tokenized` should now be a list of lists. Each list should contain the preprocessed tokens associated with the `headline` in the corresponding position of the `submissions` DataFrame.

# 5. Assemble a Matrix of Unique Words
Now that we have our tokens, we can begin converting the sentences to their numerical representations.

First, we'll retrieve all of the unique words from all of the headlines. Then, we'll create a matrix, and assign those words as the column headers. We'll initialize all of the values in the matrix to `0`.


```python
unique_tokens = []
single_tokens = []

for tokens in clean_tokenized:
    for token in tokens:
        if token not in single_tokens:
            single_tokens.append(token)
        elif token in single_tokens and token not in unique_tokens:
            unique_tokens.append(token)

counts = pd.DataFrame(0, index=np.arange(len(clean_tokenized)), columns=unique_tokens)
```

# 6. Counting Token Occurances
Now that we have a matrix where all our values are `0`, we need to fill in the correct counts for each cell. This involves going through each set of tokens, and incrementing the column counters in the appropriate row.  

When we're finished, we will have a row vector for each headline that tells us how many times each token occured in that headline.

To accomplish this, we can loop through each list of tokens in `clean_tokenized`, then loop through each token in the list and increment the proper cell.


```python
# We've already loaded in clean_tokenized and counts
for i, item in enumerate(clean_tokenized):
    for token in item:
        if token in unique_tokens:
            counts.iloc[i][token] += 1
```

# 7. Removing Columns to Increase Accuracy
We have over `2000` columns in our matrix. This can make it very hard for a linear regression model to make good predictions. Too many columns will cause the model to fit to noise instead of the signal in the data.  

There are two kinds of features that will reduce prediction accuracy.  

1. Features that occur only a few times will cause overfitting, because the model doesn't have enough information to accurately decide whether they're important. These features will probably correlate differently with upvotes in the test set and the training set.  

2. Features that occur too many times can also cause issues. These are words like `and` and `to`, which occur in nearly every headline. These words don't add any information, because they don't necessarily correlate with upvotes. These types of words are sometimes called `stopwords`.

To reduce the number of features and enable the linear regression model to make better predictions, we'll remove any words that occur fewer than `5` times or more than `100` times.


```python
# We've already loaded in clean_tokenized and counts
word_counts = counts.sum(axis=0)

counts = counts.loc[:,(word_counts >= 5) & (word_counts <= 100)]
```

# 8. Splitting the Data into Train and Test Sets
Now we need to split the data into two sets so that we can evaluate our algorithm effectively. We'll train our algorithm on a training set, then test its performance on a test set.

The `[train_test_split()` function from scikit-learn will help us accomplish this.  

We'll pass in `.2` for the `test_size` parameter to randomly select `20%` of the rows for our test set, and `80%` for our training set.  

`X_train` and `X_test` contain the predictors, and `y_train` and `y_test` contain the value we're trying to predict (upvotes).


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(counts, submissions["upvotes"], test_size=0.2, random_state=1)
```

# 9. Making Predictions
Now that we have a training set and a test set, let's train a model and make test predictions.   

First we'll initialize the model using the `LinearRegression` class. Then, we'll use the `fit()` method on the model to train with `X_train` and `y_train`. Finally, we'll make predictions with `X_test`


```python
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
```

When we make predictions with a linear regression model, the model assigns coefficients to each column. Essentially, the model is determining which words correlate with more upvotes, and which with less.  

By finding these correlations, the model will be able to predict which headlines will be highly upvoted in the future. While the algorithm won't have a high level of understanding of the text, linear regression can generate surprisingly good results.

# 10. Calculating Prediction Error

Now that we have predictions, we can calculate our prediction error. We will use Mean Squared (MSE), which is a common error metric.

With MSE, we subtract the predictions from the actual values, square the results, and find the mean. Because the errors are squared, MSE penalizes errors further away from the actual value more than those close to the actual value. We want to use MSE because we'd like all of our predictions to be relatively close to the actual values.


```python
mse = sum((predictions - y_test) ** 2) / len(predictions)
mse
```




    2652.6082512522839



Our MSE is `2652`, which is a fairly large value. There's no hard and fast rule about what a "good" error rate is, because it depends on the problem we're solving and our error tolerance.  

In this case, the mean number of upvotes is `10`, and the standard deviation is `39.5`. If we take the square root of our MSE to calculate error in terms of upvotes, we get `46.7`. This means that our average error is `46.7` upvotes away from the true value. This is higher than the standard deviation, so our predictions are often far off-base.  

# 11. Recommendations

We can take several steps to reduce the error and explore natural language processing further:  

**1. Collect more data.**  
- There are many features in natural language processing. Using more data will ensure that the model will find more occurrences of the same features in the test and training sets, which will help the model make better predictions  

**2. Add "meta" features like headline length and average word length**  

**3. Use a random forest, or another more powerful machine learning technique**  

**4. Explore different thresholds for removing extraneous columns**
