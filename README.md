# Alphabet Soup Charity - Neural Network Analysis

## Overview
The Analyst was tasked with helping Beks, a data scientist for the non-profit company, Alphabet Soup, create a binary classifier that can predict whether applicants will be successful if funded by company donations.  To create the classifier, The Analyst built a neural network predictive model using python's tensorflow library to create and train the model based on previous loan data.

## Results

### Data Preprocessing
- The Analyst determined that the target variable in the data set was the "IS_SUCCESSFUL" column.  The column provides a determination of if the donation money was used effectively by the receiving company.
- The 'STATUS', 'ASK_AMT', 'IS_SUCCESSFUL', 'APPLICATION_TYPE', 'CLASSIFICATION', 'INCOME_AMT', 'AFFILIATION', 'SPECIAL_CONSIDERATIONS', columns were used as features in the model.
- The Analyst dropped the 'EIN', 'USE_CASE', 'ORGANIZATION' and 'NAME' columns from the dataset, as those columns did not offer relevant data that would benefit the model's prediction.

### Compiling, Training, and Evaluating the Model
- The Analyst set a baseline by using the model to create a prediction based on the data preprocessing methods described above.  To set up the model The Analyst used the following neurons, layers, activation functions, and epochs:

    - Hidden layers: 2
    - Neurons: 80 (layer 1), 30 (layer 2)
    - Activation functions: relu (hidden layers 1 & 2), sigmoid (output layer)
    - Epochs: 50

    ![attempt1_setup.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/attempt1_setup.png)

    ![attempt1_epochs.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/attempt1_epochs.png)

    The model resulted in an accuracy of 69.14% and was not able to achieve the target accuracy of 75%.

    ![attempt1_results.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/attempt1_results.png)

After the baseline prediction was created, The Analyst progressed with multiple attempts to optimize the model by changing the set-up parameters in an effort to reach the desired accuracy of 75%.

- Attempt 2: Increase the hidden layers & neurons

    - Hidden layers: 3
    - Neurons: 100 (layer 1), 50 (layer 2), 25 (layer 3)
    - Activation functions: relu (hidden layers 1, 2, & 3), sigmoid (output layer)
    - Epochs: 50

    ![attempt2_setup.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/attempt2_setup.png)

    ![attempt2_epochs.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/attempt2_epochs.png)

    The model resulted in an accuracy of 53.26%, less than the base line, and was not able to achieve the target accuracy of 75%.

    ![attempt2_results.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/attempt2_results.png)

- Attempt 3: Increase the hidden layers & neurons, different activation functions

    - Hidden layers: 3
    - Neurons: 100 (layer 1), 50 (layer 2), 25 (layer 3)
    - Activation functions: sigmoid (hidden layers 1, 2, & 3, output layer)
    - Epochs: 50

    ![attempt3_setup.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/attempt3_setup.png)

    ![attempt3_epochs.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/attempt3_epochs.png)

    The model resulted in an accuracy of 53.24%, less than the base line, very similar to attempt 2, and was not able to achieve the target accuracy of 75%.

    ![attempt3_results.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/attempt3_results.png)

- Attempt 4: Increase the hidden layers & neurons, different activation functions, increase epochs

    - Hidden layers: 4
    - Neurons: 200 (layer 1), 100 (layer 2), 50 (layer 3), 25 (layer 3)
    - Activation functions: sigmoid (hidden layers 1, output layer), tanh (hidden layers 2 & 3)
    - Epochs: 100

    ![attempt4_setup.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/attempt4_setup.png)

    ![attempt4_epochs.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/attempt4_epochs.png)

    The model resulted in an accuracy of 53.24%, less than the base line, the same accuracy as attempt 2, and was not able to achieve the target accuracy of 75%.

    ![attempt4_results.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/attempt4_results.png)

- Attempt 5: Decrease hidden layers, set neurons, activation functions, and epochs back to baseline. 

    - Hidden layers: 1
    - Neurons: 80 (layer 1)
    - Activation functions: relu (hidden layer 1), sigmoid (output layer)
    - Epochs: 50

    ![attempt5_setup.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/attempt5_setup.png)

    ![attempt5_epochs.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/attempt5_epochs.png)

    The model resulted in an accuracy of 68.62%, similar than the base line, yet higher accuracy as attempt 2.  Still this model was not able to achieve the target accuracy of 75%.

    ![attempt5_results.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/attempt5_results.png)

## Summary
None of The Analyst's attempts at optimizing the neural network model were successful as all of the results from the 5 iterations below 75% accuracy.  To try to determine an optimal set up for the neural network model, The Analyst ran a kerastuner function to find the best set up parameters for the predictive model and provide the highest resulting accuracy from those parameters.  The code to set up the kerastuner function is below:

![kerastuner.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/kerastuner.png)

The function provided an output showing the ideal setup parameters and highest accuracy that can be achieved from the current dataset.

![kerastuner_results.png](https://github.com/hillmanj1995/Alphabet-Soup---Neural-Network-Analysis/blob/main/Resources/kerastuner_results.png)

The kerastuner functions shows that after 500 trails, the highest accuracy that could be achieved from the current dataset was only 72.52%.  As this is lower than the target accuracy of 75%, The Analyst should go back to the original dataset and reassess which columns benefit the model's prediction performance.  Those columns should be dropped/added and the prediction attempts should be performed again.

