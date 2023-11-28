# Bank_Leave_Predictor_Project

## Company and Position
Worked as a **Data Analytics Intern** at **Takenmind**.

## Project
Goal of the project is to predict bank customer attrition.

## Table of Contents
- [Software](#software)
- [Intro](#intro)
- [Data](#data)
- [Results](#results)
- [Next](#next)
- [Acknowledgements](#acknowledgements)

## Software
1) VSC Code
2) python3
3) spaCy

## Intro
The importance of Machine Learning has risen tremendously in these past few years. Machine Learning, in simple terms, is the ability of machines to learn without being explicitly programmed. Machine Learning finds applications in various domains and one such area is Banking. Banks are troubled due to the uncertainty in the behaviour of their customers. They are unsure whether a person will stay in the bank and if a person is known to be leaving, they can try to figure out ways to make such a person stay longer.

In this article, we will look at some ways to figure out whether a customer will leave the bank or not. I have used Google Colab programming environment for solving the problem.

Let’s Start!

## Data
The first step in solving any machine learning problem is to get the data. So, I have obtained the dataset called ‘churn.csv’ (name changed for ease) from https://www.kaggle.com/shrutimechlearn/churn-modelling. This dataset is named ‘Churn_Modelling.csv’ and it contains various columns like ‘age’, ‘gender’, ‘location’, ‘bank balance’, etc. So, I have just downloaded this data from the site. It is just ~670 kb.

# Setting up the Programming Environment
Before starting programming in Google Colab, we need to give Colab access to our Google Drive so that we can load the ‘churn.csv’ for our work. We do it like below.

```python
from google.colab import drive
drive.mount('/content/drive')
```
It will open the page from where we choose the Google account whose Colab environment we are using. After this, we have to copy a line from another site (which soon opens) and paste it in our Colab notebook.

# Installing and Importing the Required Libraries
We will need a library called ‘eli5’ to measure feature importance. We will discuss it once the model is created and trained. Just install it for now like below.

```python
!pip install eli5
```
Now, we import the libraries that we require.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix , classification_report
```

# Loading the Data

We now load the data with pandas.

```python
data = pd.read_csv('/content/drive/MyDrive/Work/Projects/Bank Leave Predictor Project/churn.csv')
```

You will need to change the address of the ‘churn.csv’ as per your drive.

Now, let’s check it out!

<img width="1440" alt="Screenshot 2023-11-28 at 12 15 30 PM" src="https://github.com/sankalpsaoji98/Bank_Leave_Prediction/assets/26198596/d5860d1c-0a32-4a62-a522-ca5ac1bdf12e">

The data looks like above.

# Choosing the Features

For any machine learning model, we need to have some features chosen from the data based on which our model learns to predict whether a customer will leave the bank or not. So, when we look at the data we get a rough idea of which columns may have an effect on the customer leaving the bank. We surely know that ‘RowNumber’, ‘CustomerId’ and ‘Surname’ can have no effect on the result. So, we will choose columns like below.

```python
X = data.iloc[:,3:-1]
```
<img width="1443" alt="Screenshot 2023-11-28 at 12 16 30 PM" src="https://github.com/sankalpsaoji98/Bank_Leave_Prediction/assets/26198596/13750da3-9984-48b1-99a2-81099b2eb274">

# Encoding the Categorical Features
As you see in the set of features X, we have columns like ‘Geography’ and ‘Gender’ which contain words and not numbers. Also, these words can assume a limited number of values. These are called Categorical features. But, our ML model is a mathematical model and so, it will require numbers for computation. For this, we do something called Encoding. This is done as below.

```python
encoder = OrdinalEncoder()
value = encoder.fit_transform(X['Geography'].values.reshape(-1, 1))
X['Geography'] = value
encoder = OrdinalEncoder()
value = encoder.fit_transform(X['Gender'].values.reshape(-1, 1))
X['Gender'] = value
```

We have now encoded the Geography and Gender columns with OrdinalEncoder(). There are many encoders available in various libraries. You can check them out!

Now, X looks like,

<img width="1093" alt="Screenshot 2023-11-28 at 12 55 14 PM" src="https://github.com/sankalpsaoji98/Bank_Leave_Prediction/assets/26198596/0546e2d1-57dc-4ad5-b14e-462280821f7a">

# Getting the Target Column
Now, predicting whether a customer leaves the bank is a supervised learning problem. So, we have to train the model so as to be able to predict the right target variable which is a column of 0s and 1s created as a binary response to leaving the bank or not. So, this is present as the ‘Exited’ column in our dataset which we will select. We do this as below.


```python
y = data.iloc[:,len(data.columns)-1]
```

Now, this looks like,

<img width="500" alt="Screenshot 2023-11-28 at 12 56 24 PM" src="https://github.com/sankalpsaoji98/Bank_Leave_Prediction/assets/26198596/d006607e-a731-4cc9-9697-f1d6f746c6c0">

# Splitting the Data into Train and Test Sets

There is a function called train_test_split in sklearn which can be used to divide our data into training and testing sets. So, we do this as,

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
```

It is a general practice in ML to get 70% data as training and 30% for testing.

Now, we check the lengths of train and test sets.

```python
len(X_train), len(X_test), len(y_train), len(y_test)
```
Running this we get training set size as 7000 and testing set size as 3000.

## Models

# Implementing a Random Forest Classifier Model

We will first try out a Random Forest Classifier model for getting the customers who will leave the bank.

```python
RF = RandomForestClassifier(n_estimators = 100, max_depth = 2, random_state = 0)
RF.fit(X_train, y_train)
```
The above way shows how we train the model. I have set n_estimators as 100 and max_depth as 2 for now but you can experiment with other variations.

# Performance Check for Random Forest Classifier Model
We will check the scores on training and testing set now.

```python
round(RF.score(X_train, y_train), 4)
```

We get 0.8093 average accuracy. It indicates 80.93% accuracy. For testing set, do,

```python
round(RF.score(X_test, y_test), 4)
```

We get 0.8237 average accuracy. It indicates 82.37% accuracy.

# Checking Feature Importance for Random Forest Classifier Model

Now, eli5, which we downloaded will come into picture. We get feature importance as below.

```python
perm = PermutationImportance(RF, random_state = 42, n_iter = 10).fit(X, y)
eli5.show_weights(perm, feature_names = X.columns.tolist())
```

n_iter is 10 for now but can be changed as required. We get the following output for our features.

<img width="601" alt="Screenshot 2023-11-28 at 1 00 04 PM" src="https://github.com/sankalpsaoji98/Bank_Leave_Prediction/assets/26198596/5c034617-8f55-49ba-b4c4-c37a7f24724e">

This indicates that ‘NumOfProducts’, ‘Age’ and ‘Balance’ are our top features.

# Implementing a MLP Classifier Model

Now, we create another training and testing set for another model.

```python
X_train_new, X_test_new, y_train_new, y_test_new =  train_test_split(X, y, test_size = 0.30, random_state = 42)
```
Now, as we already tried with a Random Forest Classifier Model, we will check out what happens with MLP Classifier Model. We do this as below.

```python
clf = MLPClassifier(random_state = 1, max_iter = 100).fit(X_train_new, y_train_new)
```
Here, you can try out different max_iter values.

# Performance Check for MLP Classifier Model
We will check the scores of this model on training and testing set now.

```python
clf.score(X_train_new, y_train_new)
```

We get 0.755 average accuracy. It indicates 75.50% accuracy. For testing set, do,

```python
clf.score(X_test_new, y_test_new)
```

We get 0.761 average accuracy. It indicates 76.10% accuracy.

# Checking Feature Importance for MLP Classifier Model

For this model, we get feature importance as below.

```python
perm = PermutationImportance(clf, random_state = 42, n_iter = 10).fit(X, y)
eli5.show_weights(perm, feature_names = X.columns.tolist())
```

n_iter is 10 here also but can be changed as required. We get the following output for our features.

<img width="675" alt="Screenshot 2023-11-28 at 1 03 12 PM" src="https://github.com/sankalpsaoji98/Bank_Leave_Prediction/assets/26198596/70d61ac4-334f-4dc9-bdca-3ded58f76a4a">

This model gives different importance to features. So, this indicates that ‘Balance’, ‘EstimatedSalary’ and ‘Age’ are our top features.

# Implementing a Neural Network Model and Checking the Performance

Now, after trying out a Random Forest Classifier model and a MLP Classifier model, we will try out a Neural Network Model. We use keras to get the model done.

```python
model = keras.Sequential([
keras.layers.Dense(10, input_shape = (10,), activation = 'relu'),
keras.layers.Dense(25, activation = 'relu'),
keras.layers.Dense(1, activation = 'sigmoid')
])
model.compile(optimizer = 'adam',
loss = 'binary_crossentropy',
metrics = ['accuracy'])
```
We are having a 25 node hidden layer. This can be changed as per requirement. You can tweak and try out other combinations. We are using the ‘adam’ optimizer and ‘binary_crossentropy’ loss.

We fit and train the model as below. We will do with 50 epochs.

```python
model.fit(X_train, y_train, epochs = 50)
```
After training, with 50 epochs, I got 73.09% accuracy. After checking on test data, like,

```python
model.evaluate(X_test, y_test)
```
The model gives 78.50% accuracy on this test data.

Now, we will print the classification report and check the performance. We do it as below.

```python
yp = model.predict(X_test)
y_pred = []
for element in yp:
if element > 0.5:
y_pred.append(1)
else:
y_pred.append(0)
print(classification_report(y_test, y_pred))
```
We get the final classification report as,

<img width="867" alt="Screenshot 2023-11-28 at 1 05 17 PM" src="https://github.com/sankalpsaoji98/Bank_Leave_Prediction/assets/26198596/6d3bfb71-b52b-4438-92f6-fcd01dd5a869">

Based on the above metrics, we see that precision and recall are less with the above model for class 1 but good for class 0. So, steps can be taken to increase the values for class 1. This is the good thing about classification report that it gives very detailed depiction of every class and shows all the metrics.

So, now, you know how to create a classification model with sklearn and how to use it for bank churn problem. You can move into more detailed approaches now!

## Acknowledgements
The work was done as a self-motivated project.
