# Neural_Network_Charity_Analysis

## Overview
The purpose of this project is to create a machine learning model predicting whether applicants will be successful if funded by Alphabet Soup.

## Results

### Data Preprocessing
- The target variable in this model: "IS_SUCCESSFUL" column
- The features variables in this model:
    - APPLICATION_TYPE—Alphabet Soup application type
    - AFFILIATION—Affiliated sector of industry
    - CLASSIFICATION—Government organization classification
    - USE_CASE—Use case for funding
    - ORGANIZATION—Organization type
    - STATUS—Active status
    - INCOME_AMT—Income classification
    - SPECIAL_CONSIDERATIONS—Special consideration for application
    - ASK_AMT—Funding amount requested
- The variables are removed from the input data: "EIN" and "NAME" columns
### Compiling, Training, and Evaluating the Model
- Define the model

The initial deep learning model is using 80 neurons in first hidden layer, 30 neurons in second hidden layer, both are using ReLU activation function. For output layer, we use sigmoid activation function to keep the output between 0 and 1.
```python
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_feature = len(X_train[0])
hidden_nodes_layer1 = 80
hidden_nodes_layer2 = 30

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_feature,activation='relu')
      )

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation='relu'))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```

- Model structure
```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)              (None, 80)                3520      
_________________________________________________________________
dense_4 (Dense)              (None, 30)                2430      
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 31        
=================================================================
Total params: 5,981
Trainable params: 5,981
Non-trainable params: 0
_________________________________________________________________
```
- Model performance

The target of predictive accuracy for Alphabet Soup company is 75%. Unfortunately, the model performance is within 72.41% accuracy. It means that the model we designed above is not going to support for Alphabet Soup to predict if applicants will be successful funded or not.

```python
# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```
```
268/268 - 0s - loss: 0.5901 - accuracy: 0.7241
Loss: 0.5900665521621704, Accuracy: 0.7240816354751587
```

- Model optimization



## Summary
Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.
