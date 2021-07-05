---
layout: post
title:  "Recommendation System: IMDB"
cover: "/assets/media/recommendation-system/imdb.png"
permalink: /:categories/:year/:month/:day/Recommendation-System
excerpt_separator: <!--end-->
categories: projects
date: 2017-09-15
---

Building A Recommendation System for Movies.

<!--end-->

# Recommendation System: IMDB

**Building A Recommendation System for Movies.**

A Recommendation System is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item. Many companies apply recommendation systems in one form or another; Amazon uses it to suggest products to customers, YouTube uses it to decide which video to play next, and Facebook uses it to recommend pages to like and people to follow.  

Recommendation Systems can be classified into 3 types:

1. `Simple Recommendation`: offer generalized recommendations to every user, based on movie popularity and/or genre. The basic idea behind this system is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience. IMDB Top 250 is an example of this system.
2. `Content-based Recommendation`: suggest similar items based on a particular item. This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommendation systems is that if a person liked a particular item, he or she will also like an item that is similar to it.
3. `Collaborative Filtering Engines`: these systems try to predict the rating or preference that a user would give an item-based on past ratings and preferences of other users. Collaborative filters do not require item metadata like its content-based counterparts.

# 1. Simple Recommendation System

**A simplified clone of IMDB Top 250 Movies using metadata collected from IMDB.**

Below are the steps we will take to build our Simple Recommendation System:
1. Decide on the metric or score to rate movies on
2. Calculate the score for every movie
3. Sort the movies based on the score and output the top results


```python
# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Print the first three rows
metadata.head(3)
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
      <th>adult</th>
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>...</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>video</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>
      <td>30000000</td>
      <td>[{'id': 16, 'name': 'Animation'}, {'id': 35, '...</td>
      <td>http://toystory.disney.com/toy-story</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>...</td>
      <td>1995-10-30</td>
      <td>373554033.0</td>
      <td>81.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Toy Story</td>
      <td>False</td>
      <td>7.7</td>
      <td>5415.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>NaN</td>
      <td>65000000</td>
      <td>[{'id': 12, 'name': 'Adventure'}, {'id': 14, '...</td>
      <td>NaN</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>...</td>
      <td>1995-12-15</td>
      <td>262797249.0</td>
      <td>104.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>Roll the dice and unleash the excitement!</td>
      <td>Jumanji</td>
      <td>False</td>
      <td>6.9</td>
      <td>2413.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>{'id': 119050, 'name': 'Grumpy Old Men Collect...</td>
      <td>0</td>
      <td>[{'id': 10749, 'name': 'Romance'}, {'id': 35, ...</td>
      <td>NaN</td>
      <td>15602</td>
      <td>tt0113228</td>
      <td>en</td>
      <td>Grumpier Old Men</td>
      <td>A family wedding reignites the ancient feud be...</td>
      <td>...</td>
      <td>1995-12-22</td>
      <td>0.0</td>
      <td>101.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Still Yelling. Still Fighting. Still Ready for...</td>
      <td>Grumpier Old Men</td>
      <td>False</td>
      <td>6.5</td>
      <td>92.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 24 columns</p>
</div>



One of the most basic metrics you can think of is the rating. However, using this metric has a few caveats. For one, it does not take into consideration the popularity of a movie. Therefore, a movie with a rating of 9 from 10 voters will be considered 'better' than a movie with a rating of 8.9 from 10,000 voters.

On a related note, this metric will also tend to favor movies with smaller number of voters with skewed and/or extremely high ratings. As the number of voters increase, the rating of a movie regularizes and approaches towards a value that is reflective of the movie's quality. It is more difficult to discern the quality of a movie with extremely few voters.

Taking these shortcomings into consideration, it is necessary that we come up with a weighted rating that takes into account the average rating and the number of votes it has garnered. Such a system will make sure that a movie with a 9 rating from 100,000 voters gets a (fair) higher score than a movie with the same rating but fewer voters.

Since we are trying to build a clone of IMDB's Top 250, we will use its weighted rating formula as our metric/score. Mathematically, it is represented as follows:

![WEIGHTED-RATING](/assets/media/recommendation-system/weighted_rating.png)

where,
* *v* is the number of votes for the movie
* *m* is the minimum votes required to be listed in the chart
* *R* is the average rating of the movie
* *C* is the mean vote across the whole report

We already have the values to ***v*** (`vote_count`) and ***R*** (`vote_average`) for each movie in the dataset. It is also possible to directly calculate ***C*** from this data.

What we need to determine is an appropriate value for ***m***, the minimum votes required to be listed in the chart. There is no right value for ***m***. We can view it as a preliminary negative filter that ignores movies which have less than a certain number of votes. The selectivity of our filter is up to your discretion.

In this case, we will use the 90th percentile as our cutoff. In other words, for a movie to feature in the charts, it must have more votes than at least 90% of the movies in the list. (If we chose the 75th percentile as our cutoff, we would have considered the top 25% of the movies in terms of the number of votes garnered. As the percentile decreases, the number of movies considered increases).

As a first step, let's calculate the value of ***C***, the mean rating across all movies:


```python
# Calculate C
C = metadata['vote_average'].mean()
print(C)
```

    5.618207215133889


The average rating of a movie on IMDB is around 5.6, on a scale of 10.

Next, let's calculate the number of votes, ***m***, received by a movie in the 90th percentile. We will use the `.quantile()` method from the Pandas library:


```python
# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)
print(m)
```

    160.0


Next, we can filter the movies that qualify for the chart, based on their vote counts:


```python
# Filter out all qualified movies into a new DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
q_movies.shape
```




    (4555, 24)



We use the `.copy()` method to ensure that the new `q_movies` DataFrame created is independent of our original metadata DataFrame. In other words, any changes made to the `q_movies` DataFrame does not affect the metadata.

There are 4555 movies which qualify to be in this list. Now, we need to calculate our metric for each qualified movie. To do this, we will define a function, `weighted_rating()` and define a new feature `score`, of which we'll calculate the value by applying this function to our DataFrame of qualified movies:


```python
# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)
```


```python
# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
```

Finally, let's sort the DataFrame based on the `score` feature and output the title, vote count, vote average and weighted rating or score of the top 15 movies.


```python
#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15)
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
      <th>title</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>314</th>
      <td>The Shawshank Redemption</td>
      <td>8358.0</td>
      <td>8.5</td>
      <td>8.445869</td>
    </tr>
    <tr>
      <th>834</th>
      <td>The Godfather</td>
      <td>6024.0</td>
      <td>8.5</td>
      <td>8.425439</td>
    </tr>
    <tr>
      <th>10309</th>
      <td>Dilwale Dulhania Le Jayenge</td>
      <td>661.0</td>
      <td>9.1</td>
      <td>8.421453</td>
    </tr>
    <tr>
      <th>12481</th>
      <td>The Dark Knight</td>
      <td>12269.0</td>
      <td>8.3</td>
      <td>8.265477</td>
    </tr>
    <tr>
      <th>2843</th>
      <td>Fight Club</td>
      <td>9678.0</td>
      <td>8.3</td>
      <td>8.256385</td>
    </tr>
    <tr>
      <th>292</th>
      <td>Pulp Fiction</td>
      <td>8670.0</td>
      <td>8.3</td>
      <td>8.251406</td>
    </tr>
    <tr>
      <th>522</th>
      <td>Schindler's List</td>
      <td>4436.0</td>
      <td>8.3</td>
      <td>8.206639</td>
    </tr>
    <tr>
      <th>23673</th>
      <td>Whiplash</td>
      <td>4376.0</td>
      <td>8.3</td>
      <td>8.205404</td>
    </tr>
    <tr>
      <th>5481</th>
      <td>Spirited Away</td>
      <td>3968.0</td>
      <td>8.3</td>
      <td>8.196055</td>
    </tr>
    <tr>
      <th>2211</th>
      <td>Life Is Beautiful</td>
      <td>3643.0</td>
      <td>8.3</td>
      <td>8.187171</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>The Godfather: Part II</td>
      <td>3418.0</td>
      <td>8.3</td>
      <td>8.180076</td>
    </tr>
    <tr>
      <th>1152</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>3001.0</td>
      <td>8.3</td>
      <td>8.164256</td>
    </tr>
    <tr>
      <th>351</th>
      <td>Forrest Gump</td>
      <td>8147.0</td>
      <td>8.2</td>
      <td>8.150272</td>
    </tr>
    <tr>
      <th>1154</th>
      <td>The Empire Strikes Back</td>
      <td>5998.0</td>
      <td>8.2</td>
      <td>8.132919</td>
    </tr>
    <tr>
      <th>1176</th>
      <td>Psycho</td>
      <td>2405.0</td>
      <td>8.3</td>
      <td>8.132715</td>
    </tr>
  </tbody>
</table>
</div>



We see that our chart has a lot of movies in common with the IMDB Top 250 chart: for example, our top two movies, "Shawshank Redemption" and "The Godfather", are the same as IMDB.

![TOP-250](/assets/media/recommendation-system/top_250.png)

# 2. Content-Based Recommender in Python
**Creating a 'Plot Description' Based Recommendation System.**  

In this section, we will build a system that recommends movies that are similar to a particular movie. More specifically, we will compute pairwise similarity scores for all movies based on their plot descriptions and recommend movies based on that similarity score.

The plot description is available to us as the `overview` feature in our `metadata` dataset. Let's explore the plots of the first five movies:


```python
#Print plot overviews of the first 5 movies.
metadata['overview'].head()
```




    0    Led by Woody, Andy's toys live happily in his ...
    1    When siblings Judy and Peter discover an encha...
    2    A family wedding reignites the ancient feud be...
    3    Cheated on, mistreated and stepped on, the wom...
    4    Just when George Banks has recovered from his ...
    Name: overview, dtype: object



In its current form, it is not possible to compute the similarity between any two overviews. In order to achieve our goal, we will have to compute the Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each overview. This will give us a matrix where each column represents a word in the overview vocabulary and each column represents a movie.

In its essence, the TF-IDF score is the frequency of a word occurring in a document, down-weighted by the number of documents in which it occurs. This is done to reduce the importance of words that occur frequently in plot overviews and therefore, their significance in computing the final similarity score.

Fortunately, scikit-learn gives us a built-in `TfIdfVectorizer` class that produces the TF-IDF matrix in a couple of lines of code.


```python
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape
```




    (45466, 75827)



We see that over 75,000 different words were used to describe the 45,000 movies in our dataset.

With this matrix in hand, we can now compute a similarity score. There are several candidates for this; such as the euclidean, the Pearson and the cosine similarity scores. Again, there is no right answer to which score is the best. Different scores work well in different scenarios and it is often a good idea to experiment with different metrics.

We will be using the cosine similarity to calculate a numeric quantity that denotes the similarity between two movies. We use the cosine similarity score since it is independent of magnitude and is relatively easy and fast to calculate (especially when used in conjunction with TF-IDF scores, which will be explained later). Mathematically, it is defined as follows: 

$cosine(x,y) = \frac{x. y^\intercal}{||x||.||y||} $

Since we have used the TF-IDF vectorizer, calculating the dot product will directly give us the cosine similarity score. Therefore, we will use `sklearn`'s `linear_kernel()` instead of `cosine_similarities()` since it is faster.


```python
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```

We're going to define a function that takes in a movie title as an input and outputs a list of the 10 most similar movies. Firstly, for this, we need a reverse mapping of movie titles and DataFrame indices. In other words, we need a mechanism to identify the index of a movie in your `metadata` DataFrame, given its title.


```python
#Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
```

We are now in a good position to define our recommendation function. These are the following steps we will follow:

1. Get the index of the movie given its title.
2. Get the list of cosine similarity scores for that particular movie with all movies. Convert it into a list of tuples where the first element is its position and the second is the similarity score.
3. Sort the aforementioned list of tuples based on the similarity scores; that is, the second element.
4. Get the top 10 elements of this list. Ignore the first element as it refers to itself.
5. Return the titles corresponding to the indices of the top elements.


```python
# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]
```


```python
get_recommendations('The Dark Knight Rises')
```




    12481                                      The Dark Knight
    150                                         Batman Forever
    1328                                        Batman Returns
    15511                           Batman: Under the Red Hood
    585                                                 Batman
    21194    Batman Unmasked: The Psychology of the Dark Kn...
    9230                    Batman Beyond: Return of the Joker
    18035                                     Batman: Year One
    19792              Batman: The Dark Knight Returns, Part 1
    3095                          Batman: Mask of the Phantasm
    Name: title, dtype: object




```python
get_recommendations('The Godfather')
```




    1178               The Godfather: Part II
    44030    The Godfather Trilogy: 1972-1990
    1914              The Godfather: Part III
    23126                          Blood Ties
    11297                    Household Saints
    34717                   Start Liquidation
    10821                            Election
    38030            A Mother Should Be Loved
    17729                   Short Sharp Shock
    26293                  Beck 28 - Familjen
    Name: title, dtype: object



While our system has done a good job of finding movies with similar plot descriptions, the quality of recommendations could be improved by considering other features of a movie.

# 2.1 Credits, Genres and Keywords Based Recommendations
With the usage of more features, we are going to build a recommendation system based on the following metadata: 

- The 3 Top Actors
- The director
- Related Genres
- Movie Plot Keywords

Lets start by loading and merging our new datasets:


```python
# Load keywords and credits
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

# Remove rows with bad IDs.
metadata = metadata.drop([19730, 29503, 35587])

# Convert IDs to int. Required for merging
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')
```


```python
# Print the first two movies of your newly merged metadata
metadata.head(2)
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
      <th>adult</th>
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>...</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>video</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>cast</th>
      <th>crew</th>
      <th>keywords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>
      <td>30000000</td>
      <td>[{'id': 16, 'name': 'Animation'}, {'id': 35, '...</td>
      <td>http://toystory.disney.com/toy-story</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>...</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Toy Story</td>
      <td>False</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>[{'cast_id': 14, 'character': 'Woody (voice)',...</td>
      <td>[{'credit_id': '52fe4284c3a36847f8024f49', 'de...</td>
      <td>[{'id': 931, 'name': 'jealousy'}, {'id': 4290,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>NaN</td>
      <td>65000000</td>
      <td>[{'id': 12, 'name': 'Adventure'}, {'id': 14, '...</td>
      <td>NaN</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>...</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>Roll the dice and unleash the excitement!</td>
      <td>Jumanji</td>
      <td>False</td>
      <td>6.9</td>
      <td>2413.0</td>
      <td>[{'cast_id': 1, 'character': 'Alan Parrish', '...</td>
      <td>[{'credit_id': '52fe44bfc3a36847f80a7cd1', 'de...</td>
      <td>[{'id': 10090, 'name': 'board game'}, {'id': 1...</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 27 columns</p>
</div>



From our new dataset, we will need to extract the three top actors, the director and the keywords associated with each movie. Right now, our data is present in the form of "stringified" lists, we will need to convert them into a form that we can use with our code:


```python
# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)
```

Next, we will write functions that will extract the required information from each feature.


```python
import numpy as np
```


```python
# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
```


```python
# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []
```


```python
# Define new director, cast, genres and keywords features that are in a suitable form.
metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)
```


```python
# Print the new features of the first 3 films
metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
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
      <th>title</th>
      <th>cast</th>
      <th>director</th>
      <th>keywords</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toy Story</td>
      <td>[Tom Hanks, Tim Allen, Don Rickles]</td>
      <td>John Lasseter</td>
      <td>[jealousy, toy, boy]</td>
      <td>[Animation, Comedy, Family]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jumanji</td>
      <td>[Robin Williams, Jonathan Hyde, Kirsten Dunst]</td>
      <td>Joe Johnston</td>
      <td>[board game, disappearance, based on children'...</td>
      <td>[Adventure, Fantasy, Family]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grumpier Old Men</td>
      <td>[Walter Matthau, Jack Lemmon, Ann-Margret]</td>
      <td>Howard Deutch</td>
      <td>[fishing, best friend, duringcreditsstinger]</td>
      <td>[Romance, Comedy]</td>
    </tr>
  </tbody>
</table>
</div>



The next step would be to convert the names and keyword instances into lowercase and strip all the spaces between them. This is done so that our vectorizer doesn't count the Johnny of "Johnny Depp" and "Johnny Galecki" as the same. After this processing step, the aforementioned actors will be represented as "johnnydepp" and "johnnygalecki" and will be distinct to our vectorizer.


```python
# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
```


```python
# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)
```

We are now ready to create our "metadata soup", which is a string that contains all the metadata that we want to run through our vectorizer.


```python
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
```


```python
# Create a new soup feature
metadata['soup'] = metadata.apply(create_soup, axis=1)
```

The next step is similar to our Plot Description Based Recommendation System. The one important difference is that we will use the `CountVectorizer()` instead of TF-IDF. This is because we do not want to downgrade the presence of an actor/director if he or she has acted or directed in relatively more movies.


```python
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])
```


```python
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
```


```python
# Reset index of your main DataFrame and construct reverse mapping as before
metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])
```

We can reuse the `get_recommendations()` function by passing in the new `cosine_sim2` matrix as our second argument.


```python
get_recommendations('The Dark Knight Rises', cosine_sim2)
```




    12589      The Dark Knight
    10210        Batman Begins
    9311                Shiner
    9874       Amongst Friends
    7772              Mitchell
    516      Romeo Is Bleeding
    11463         The Prestige
    24090            Quicksand
    25038             Deadfall
    41063                 Sara
    Name: title, dtype: object




```python
get_recommendations('The Godfather', cosine_sim2)
```




    1934            The Godfather: Part III
    1199             The Godfather: Part II
    15609                   The Rain People
    18940                         Last Exit
    34488                              Rege
    35802            Manuscripts Don't Burn
    35803            Manuscripts Don't Burn
    8001     The Night of the Following Day
    18261                 The Son of No One
    28683            In the Name of the Law
    Name: title, dtype: object



Our recommendation system has been successful in capturing more information due to more metadata and has given us a better recommendations. There are numerous ways of playing with this system in order to improve recommendations.  

Some suggestions:  

- Introduce a popularity filter: this recommendation system would take the list of the 30 most similar movies, calculate the weighted ratings (using the IMDB formula from above), sort movies based on this rating and return the top 10 movies.
- Other crew members: other crew member names, such as screenwriters and producers, could also be included.
- Increasing weight of the director: to give more weight to the director, he or she could be mentioned multiple times in the soup to increase the similarity scores of movies with the same director.

# 3. Collaborative Filtering
Another popular type of recommendation systems is known as Collaborative Filtering.

Collaborative Filtering can further be classified into two types:

1. `User-based Filtering`: these systems recommend products to a user that similar users have liked. For example, let's say Alice and Bob have a similar interest in books (that is, they largely like and dislike the same books). Now, let's say a new book has been launched into the market and Alice has read and loved it. It is therefore, highly likely that Bob will like it too and therefore, the system recommends this book to Bob.

2. `Item-based Filtering`: these systems are extremely similar to the content recommendation engine that you built. These systems identify similar items based on how people have rated it in the past. For example, if Alice, Bob and Eve have given 5 stars to The Lord of the Rings and The Hobbit, the system identifies the items as similar. Therefore, if someone buys The Lord of the Rings, the system also recommends The Hobbit to him or her.

Our content based engine suffers from some limitations. It is only capable of suggesting movies that are close to a certain movie. That is, it is not capable of capturing tastes and providing recommendations across genres.

Also, the engine that we built is not really personal in that it doesn't capture the personal tastes and biases of a user. Anyone querying our engine for recommendations based on a movie will receive the same recommendations for that movie, regardless of their personal attributes.

Therefore, in this section, we will use a technique called Collaborative Filtering to make recommendations to movie viewers. Collaborative Filtering is based on the idea that users similar to an individual can be used to predict how much they will like a particular product or service that those users have experienced but they have not.

We will be using the Surprise library that uses extremely powerful algorithms like Singular Value Decomposition (SVD) to minimise RMSE (Root Mean Square Error) and give better recommendations.


```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate
```


```python
reader = Reader()
```


```python
ratings = pd.read_csv('ratings_small.csv')
ratings.head()
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1129</td>
      <td>2.0</td>
      <td>1260759185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1172</td>
      <td>4.0</td>
      <td>1260759205</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)
```


```python
svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE'])
```




    Evaluating RMSE, MAE of algorithm SVD.
    
    ------------
    Fold 1
    RMSE: 0.8934
    MAE:  0.6887
    ------------
    Fold 2
    RMSE: 0.8969
    MAE:  0.6900
    ------------
    Fold 3
    RMSE: 0.8926
    MAE:  0.6874
    ------------
    Fold 4
    RMSE: 0.8961
    MAE:  0.6919
    ------------
    Fold 5
    RMSE: 0.9006
    MAE:  0.6929
    ------------
    ------------
    Mean RMSE: 0.8959
    Mean MAE : 0.6902
    ------------
    ------------





    CaseInsensitiveDefaultDict(list,
                               {'rmse': [0.8933860941982121,
                                 0.8969233722558052,
                                 0.8925931195006682,
                                 0.896131927615416,
                                 0.9005532920542285],
                                'mae': [0.688678230673435,
                                 0.6899762071788088,
                                 0.6874410687546039,
                                 0.6919108848656457,
                                 0.6928566751129651]})



We get a mean Root Mean Sqaure Error of 0.8963 which is more than good enough for our case. Now lets train on our dataset and arrive at predictions.


```python
trainset = data.build_full_trainset()
svd.train(trainset)
```




    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x10d23b438>



Lets pick userid 1 and check the ratings they have given.


```python
ratings[ratings['userId'] == 1]
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1129</td>
      <td>2.0</td>
      <td>1260759185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1172</td>
      <td>4.0</td>
      <td>1260759205</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1263</td>
      <td>2.0</td>
      <td>1260759151</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1287</td>
      <td>2.0</td>
      <td>1260759187</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1293</td>
      <td>2.0</td>
      <td>1260759148</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1339</td>
      <td>3.5</td>
      <td>1260759125</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1343</td>
      <td>2.0</td>
      <td>1260759131</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>1371</td>
      <td>2.5</td>
      <td>1260759135</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>1405</td>
      <td>1.0</td>
      <td>1260759203</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>1953</td>
      <td>4.0</td>
      <td>1260759191</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>2105</td>
      <td>4.0</td>
      <td>1260759139</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>2150</td>
      <td>3.0</td>
      <td>1260759194</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>2193</td>
      <td>2.0</td>
      <td>1260759198</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>2294</td>
      <td>2.0</td>
      <td>1260759108</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>2455</td>
      <td>2.5</td>
      <td>1260759113</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>2968</td>
      <td>1.0</td>
      <td>1260759200</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>3671</td>
      <td>3.0</td>
      <td>1260759117</td>
    </tr>
  </tbody>
</table>
</div>




```python
svd.predict(1, 302, 3)
```




    Prediction(uid=1, iid=302, r_ui=3, est=2.8603227982448582, details={'was_impossible': False})



For movie with ID 302, we get an estimated prediction of 2.686. One great feature of this recommendation system is that it doesn't care what the movie is, or what it contains. It works purely on the basis of an assigned movie ID and tries to predict ratings based on how the other users have predicted the movie.


```python

```
