# Neural_Network_Charity_Analysis

## Overview

A Loan Prediction Risk Analysis for the Nonprofit Foundation, Alphabet Soup.

### Purpose

The purpose of this project was to design a binary classification deep learning model that was capable of predicting (with accuracy higher than 75%) whether applicants were successful if funded by Alphabet Soup. 

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

See [AlphabetSoupCharity.ipynb](https://github.com/KimberlyCrawford/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb) file for Deliverable 1 and 2 code.

### Deliverable 3 Optimize the Model

Using TensorFlow, the model was optimized in order to achieve a target predictive accuracy higher than 75%. 

See [TensorFlow Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.10587&showTestData=false&discretize=true&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&discretize_hide=true&regularization_hide=true&learningRate_hide=true&regularizationRate_hide=true&percTrainData_hide=true&showTestData_hide=true&noise_hide=true&batchSize_hide=true) for more information.

## Results

### Deliverable 4 

A written report on the performance of the deep learning model created for AlphabetSoup. The following questions were answered after completing Deliverables 1-3 above.

#### Data Preprocessing

1) What variable(s) are considered the target(s) for your model? The IS_SUCCESSFUL variable was the target and answers the question whether the money received from the foundation was used effectively or not. The variable had two values: Yes = 1, No = 0. 

2) What variable(s) are considered to be the features for your model?

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

3) What variable(s) are neither targets nor features, and should be removed from the input data? The EIN and NAME variables were removed as they were identification columns.

#### Compiling, Training, and Evaluating the Model

4) How many neurons, layers, and activation functions did you select for your neural network model, and why? The number_input_features = len(X_train[0]), hidden_nodes_layer1 =  80, and hidden_nodes_layer2 = 30. Why???? 

5) Were you able to achieve the target model performance? No, the model designed when completing Deliverables 1 and 2 above resulted in predictive accuracy of 73%; therefore, the model design and parameters were changed to improve the performance as noted below. 

6) What steps did you take to try and increase model performance?

19.2.6 The most straightforward means of improving neural network performance is tweaking the model design and parameters. When it comes to tweaking a neural network model, a little can go a long way. If we tweak too many design aspects and parameters at once, we can cause a model to become less effective without a means of understanding why. To avoid trapping ourselves in endless optimization iterations, we can use characteristics of our input data to determine what parameters should be changed. There are a few means of optimizing a neural network:

- Check out your input dataset.
- Add more neurons to a hidden layer.
- Add additional hidden layers.
- Use a different activation function for the hidden layers.
- Add additional epochs to the training regimen.


## Summary

Summarize the overall results of the deep learning model. 
Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.
