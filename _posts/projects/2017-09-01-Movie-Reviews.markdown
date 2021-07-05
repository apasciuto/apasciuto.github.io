---
layout: post
title:  "Sentiment Analysis: Movie Reviews"
cover: "../assets/media/movie-reviews/popcorn.png"
permalink: /:categories/:year/:month/:day/Sentiment-Analysis-Naive-Bayes
excerpt_separator: <!--end-->
categories: projects
date: 2017-09-01
---

Using Naive Bayes to Classify Movie Reviews Based on Sentiment.

<!--end-->


# Sentiment Analysis: Naive Bayes

**Using Naive Bayes to Classify Movie Reviews Based on Sentiment.**  

# 1. Movie Reviews

We will be working with a CSV file containing movie reviews. Each row contains the text of the review, as well as a number indicating whether the tone of the review is positive(`1`) or negative(`-1`).  

We want to predict whether a review is negative or positive, based on the text alone. To do this, we'll train an algorithm using the reviews and classifications in `train.csv`, and then make predictions on the reviews in `test.csv`. We'll be able to calculate our error using the actual classifications in `test.csv` to see how good our predictions were.  

We'll use [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) for our classification algorithm. A Naive Bayes classifier works by figuring out how likely data attributes are to be associated with a certain class.  

Bayes' theorem is stated mathematically as the following equation:  

$$ P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)} $$  

This equation states that: "the probability of A given that B is true equals the probability of B given that A is true times the probability of A being true, divided by the probability of B being true."

# 2. Finding Word Count

Our goal is to determine if we should classify a data row as negative or positive.  

We have to calculate the probabilities of each classification, and the probabilities of each feature falling into each classification. To do this we will need to generate features from one long strong by splitting the text up into words based on whitespace. Each word in a movie review will then be a feature that we can work with. We can then count up how many times each word occurs in the negative reviews, and how many times each word occurs in the positive reviews. Eventually, we will use the counts to compute the probability that a new review will belong to one class versus the other.


```python
import re
import csv
# A Python class that lets us count how many times items occur in a list
from collections import Counter

# Read in the training data
with open("train.csv", 'r') as file:
    reviews = list(csv.reader(file))

def get_text(reviews, score):
    # Join together the text in the reviews for a particular tone
    # Convert the text to lowercase so the algorithm doesn't see "Not" and "not" as different words
    return " ".join([r[0].lower() for r in reviews if r[1] == str(score)])

def count_text(text):
    # Split text into words based on whitespace
    words = re.split("\s+", text)
    # Count up the occurrence of each word
    return Counter(words)

negative_text = get_text(reviews, -1)
positive_text = get_text(reviews, 1)
# Generate word counts for negative tone
negative_counts = count_text(negative_text)
# Generate word counts for positive tone
positive_counts = count_text(positive_text)

print("Negative text sample: {0}".format(negative_text[:100]))
print("Positive text sample: {0}".format(positive_text[:100]))
```

    Negative text sample: plot : two teen couples go to a church party drink and then drive . they get into an accident . one 
    Positive text sample: films adapted from comic books have had plenty of success whether they're about superheroes ( batman


# 3. Naive Bayes: The Python Way

### 3.1 Making Predictions About Review Classifications
Now that we have the word counts, we just need to convert them to probabilities and multiply them out to predict the classifications.  

Let's say we wanted to find the probability that the review `didn't like it` expresses a negative sentiment. We would find the total number of times the word `didn't` occurred in the negative reviews, and divide it by the total number of words in the negative reviews to get the probability of `x` given `y`. We would then do the same for `like` and `it`. We would multiply all three probabilities, and then multiply by the probability of any document expressing a negative sentiment to get our final probability that the sentence expresses negative sentiment.  

We would do the same for positive sentiment. Then, whichever probability is greater would be the class that the algorithm assigns the review to.  

To accomplish all of this, we'll need to determine the probabilities of each class occurring in the data, and then make a function that determines the classification:


```python
import re
from collections import Counter

def get_y_count(score):
    # Determine the count of each classification occurring in the data
    return len([r for r in reviews if r[1] == str(score)])

# We will use these counts for smoothing when computing the prediction
positive_review_count = get_y_count(1)
negative_review_count = get_y_count(-1)

# These are the class probabilities
prob_positive = positive_review_count / len(reviews)
prob_negative = negative_review_count / len(reviews)

def make_class_prediction(text, counts, class_prob, class_count):
    prediction = 1
    text_counts = Counter(re.split("\s+", text))
    for word in text_counts:
        # For every word in the text, we get the number of times that word occurred in the reviews for a given class, add 1 to smooth the value, and divide by the total number of words in the class (plus the class_count, also to smooth the denominator)
        # Smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist in the training data
        # We also smooth the denominator counts to keep things even
        prediction *=  text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))
    # Now we multiply by the probability of the class existing in the documents
    return prediction * class_prob

# Now we can generate probabilities for the classes our reviews belong to
# The probabilities themselves aren't very useful, we make our classification decision based on which value is greater
print("Review: {0}".format(reviews[0][0]))
print("Negative prediction: {0}".format(make_class_prediction(reviews[0][0], negative_counts, prob_negative, negative_review_count)))
print("Positive prediction: {0}".format(make_class_prediction(reviews[0][0], positive_counts, prob_positive, positive_review_count)))
```

    Review: plot : two teen couples go to a church party drink and then drive . they get into an accident . one of the guys dies but his girlfriend continues to see him in her life and has nightmares . what's the deal ? watch the movie and " sorta " find out . . . critique : a mind-fuck movie for the teen generation that touches on a very cool idea but presents it in a very bad package . which is what makes this review an even harder one to write since i generally applaud films which attempt
    Negative prediction: 3.005053036235652e-221
    Positive prediction: 1.307170546690679e-226


### 3.2 Predicting the Test Set
Now that we can make predictions, let's predict the probabilities for the reviews in `test.csv`.


```python
import csv

def make_decision(text, make_class_prediction):
    # Compute the negative and positive probabilities
    negative_prediction = make_class_prediction(text, negative_counts, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, positive_counts, prob_positive, positive_review_count)

    # We assign a classification based on which probability is greater
    if negative_prediction > positive_prediction:
      return -1
    return 1

with open("test.csv", 'r') as file:
    test = list(csv.reader(file))

predictions = [make_decision(r[0], make_class_prediction) for r in test]
```

### 3.3 Computing Prediction Error

Now that we know the predictions, we'll compute error using the area under the `ROC` curve. This will tell us how "good" the model is; closer to 1 means that the model is better.  

Computing error is a very important measure of whether your model is "good," and when it's getting better or worse.


```python
actual = [int(r[1]) for r in test]

from sklearn import metrics

# Generate the ROC curve using scikits-learn
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

# Measure the area under the curve
# The closer to 1 it is, the "better" the predictions
print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))
```

    AUC of the predictions: 0.680701754385965


There are a lot of extensions we could add to this algorithm to make it perform better. We could look at `n-grams` instead of unigrams, for example. We could also remove punctuation and other non-characters. We could remove `stopwords`, or perform `stemming` or lemmatization.

# 4. Naive Bayes: The Scikit-Learn Way


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

# Generate counts from text using a vectorizer  
# We can choose from other available vectorizers, and set many different options
# This code performs our step of computing word counts
vectorizer = CountVectorizer(stop_words='english', max_df=.05)
train_features = vectorizer.fit_transform([r[0] for r in reviews])
test_features = vectorizer.transform([r[0] for r in test])

# Fit a Naive Bayes model to the training data
# This will train the model using the word counts we computed and the existing classifications in the training set
nb = MultinomialNB()
nb.fit(train_features, [int(r[1]) for r in reviews])

# Now we can use the model to predict classifications for our test features
predictions = nb.predict(test_features)

# Compute the error
# It's slightly different from our model because the internals of this process work differently from our implementation
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)
print("Multinomal naive bayes AUC: {0}".format(metrics.auc(fpr, tpr)))
```

    Multinomal naive bayes AUC: 0.635500515995872
