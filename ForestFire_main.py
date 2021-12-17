'''
***** Final Project 2021 - Software Carpentry *****
Contributors - Mahin Gadkari , Rashi Sultania
Aim:- The aim of the code is to predict the likelyhood of forest fires
given the weather conditions.
'''

import functions as F

print("************         Welcome to the Forest Fire Predictor        ************")
print("This predictor works on various machine learning classification\n" +
      "algorithms such as Decission Tree, Support Vector Machine, Multi Level Perceptron \n" +
      "and Gaussian Naive Bayes.\n")
print("To select which alogrithm to use  for the prediction use the given legend and \n" +
      "input the code when propted for model type:\n")
print("Model: Decision Tree Classifier                                    Code: DT\n" +
      "Model: Support Vector Classifier                                   Code: SVC\n" +
      "Model: Multi Level Perceptron Classifier                           Code: MLP\n" +
      "Model: Gaussian Naive Bayes Classifier                             Code: GNB\n" +
      "Model: The concensus between all models gives us the prediction    Code: ALL\n")
# model_type = get_model_type()
model_type = "DT"
inputs = [[29.0, 57.0, 18.0, 0, 65.7, 3.4, 7.6, 1.3, 3.4, 0.5]]
# while True:
#     Train = input("Do you want to retrain the model?(Y/N) : ")
#     if Train == "Y":
#         Model_train(model_type)
#         break
#     elif Train == "N":
#         break
#     else:
#         print("Enter a valid answer!")
# while True:
#     Score = input("Do you want to know the accuracy of the model?(Y/N) : ")
#     if Score == "Y":
#         Model_score(model_type)
#         break
#     elif Score == "N":
#         break
#     else:
#         print("Enter a valid answer!")
# while True:
#     Predict = input("Do you want to make a prediction?(Y/N) : ")
#     if Score == "Y":
#         print("Enter the required data as requested")
#         # inputs = get_inputs()
#         prediction = Model_predict(model_type, X=inputs)
#         break
#     elif Score == "N":
#         break
#     else:
#         print("Enter a valid answer!")

F.Model_score(model_type)
prediction = F.Model_predict(model_type, X=inputs)
print(prediction)
