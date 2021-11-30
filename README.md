# Neural_Network_Charity_Analysis

## Overview

A Loan Prediction Risk Analysis for Alphabet Soup.

### Purpose

To design a binary classification model that was capable of predicting whether applicants will be successful if funded by Alphabet Soup. Then the model was optimized by modifying the input data and training to achieve a target predictive accuracy higher than 75%.

## Dataset

From Alphabet Soup’s business team, a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years was analyzed. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

### Deliverable 1 Preprocessing Data for a Neural Network Model

Using knowledge of Pandas and the Scikit-Learn’s StandardScaler(), the dataset was preprocessed in order to compile, train, and evaluate the neural network model later in Deliverable 2. 

### Deliverable 2 Compile, Train, and Evaluate the Model

Using TensorFlow, a neural network or deep learning model was designed to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. Then compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy.

See [AlphabetSoupCharity.ipynb](https://github.com/KimberlyCrawford/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb) file and [AlphabetSoupCharity.h5](https://github.com/KimberlyCrawford/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.h5) for Deliverable 1 and 2 code.

### Deliverable 3 Optimize the Model

Using TensorFlow, the model was optimized in order to achieve a target predictive accuracy higher than 75%. 

See [TensorFlow Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.10587&showTestData=false&discretize=true&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&discretize_hide=true&regularization_hide=true&learningRate_hide=true&regularizationRate_hide=true&percTrainData_hide=true&showTestData_hide=true&noise_hide=true&batchSize_hide=true) for more information.

## Results

### Deliverable 4 

A written report on the performance of the deep learning model created for AlphabetSoup. The following questions were answered after completing Deliverables 1-3 above.

#### Data Preprocessing

1) What variable(s) are considered the target(s) for your model?
2) What variable(s) are considered to be the features for your model?
3) What variable(s) are neither targets nor features, and should be removed from the input data?

#### Compiling, Training, and Evaluating the Model

4) How many neurons, layers, and activation functions did you select for your neural network model, and why?
5) Were you able to achieve the target model performance?
6) What steps did you take to try and increase model performance?

## Summary

Summarize the overall results of the deep learning model. 
Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.