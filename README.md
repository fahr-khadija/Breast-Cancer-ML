# Breast-Cancer-ML
Breast Cancer Classification & Prediction using Machine Learning Framework
## ![Breast-Cancer](https://tse4.mm.bing.net/th?id=OIP.2ickM6j-W8NcEtf5LYG8JQHaEK&pid=Api&P=0&h=900)

# Project Objective 
Generate a good machine Learning model that can be used to classify tumor and predict patient with Breast Cancer from any input preprocessing data .

# Why we choose this topic:

Breast cancer, affecting 2.2 million individuals annually, is the most prevalent cancer worldwide. Early 
diagnosis significantly improves survival rates, and a key challenge lies in distinguishing between benign and malignant tumors. 
Machine learning (ML) techniques have demonstrated the potential to dramatically enhance diagnostic accuracy, with research revealing a 97% success rate compared to 79% by experienced doctors.
By integrating advanced computational models, we seek to  achieve early and highly accurate diagnosis of breast cancer patients, thereby reducing waiting times and minimizing human and technical errors in the diagnostic process. This predictive model aims to assist doctors in efficiently diagnosing breast cancer and enable patients to receive early treatment.

 ### General Description
Breast cancer is a cancer that develops from breast tissue.
The external signs of breast cancer may include a lump in the breast, a change in breast shape, dimpling of the skin, milk rejection, fluid coming from the nipple, a newly inverted nipple, or a red or scaly patch of skin .
The internal  signs affect the size ,the form  and the characteristic of the tumor.
They are  a pictures showing the difference between a Benign and Malignant Tumors 
![tumor](https://github.com/fahr-khadija/Breast-Cancer-ML/blob/main/img/image.png)  
![Alt text]( https://github.com/fahr-khadija/Breast-Cancer-ML/blob/main/img/image-5.png)
for more information you can visit this health links https://en.wikipedia.org/wiki/Breast_cancer
https://en.wikipedia.org/wiki/Breast_cancer#Multiple_primary_tumours
### Analyse breast tumours by Machine Learning 

   #### Our workflow ensures an approach to breast tumor analysis, from data collection to deploying a predictive model for real-time predictions on new data. We divided on 7 steps :

         1-Dataset Collection (we use the data retrieve from SQL)
         2-Data Preprocessing DP (we developp a fct that preprocess any new data'preprocess_data')
         3-Exploratory Data Analysis  EDA
         4-Classification of prediction on 3 models SVM,NN,KNN
         5-Performance Evaluation (Accuracy , lost and confusion matrix) 
         6-Best Model Generated (Compare the 3 models SVM,NN,KNN and choose the best one)
         7-Predict on new Data (use the fct 'split-csv'to generate new data from original data )
   
   #### Schematic workflow diagram
  ![Alt text](https://github.com/fahr-khadija/Breast-Cancer-ML/blob/main/img/image-2.png)

### Flow Implementation 
     The implementation is doing on the final code  and it s deeply explained , 
     you can just load your csv file in final code and launch it in collab  
     the final code contain the generation of the best model and  prediction of  new data 
   
   All Librariries version used are listed in this text file 
### Librarires version  
 https://github.com/fahr-khadija/Breast-Cancer-ML/blob/main/Breast-cancer-project/Libraries.txt

### split function 
https://github.com/fahr-khadija/Breast-Cancer-ML/blob/main/Breast-cancer-project/split-csv.py

### Preprocess function 
https://github.com/fahr-khadija/Breast-Cancer-ML/blob/main/Breast-cancer-project/preprocessing_functions.py
###  The data used is breast_cancer_data_load.csv  to generate the best ML model 
 https://github.com/fahr-khadija/Breast-Cancer-ML/blob/main/Breast-cancer-project/breast_cancer_data_load.csv
###  The new data used is breast_cancer_data_new.csv to predict with the best ML model already generated 
 https://github.com/fahr-khadija/Breast-Cancer-ML/blob/main/Breast-cancer-project/breast_cancer_data_new.csv
### the finale code collab  (generation of best model ,prediction new data with it )
https://github.com/fahr-khadija/Breast-Cancer-ML/blob/main/Breast-cancer-project/breast_cancer_final.ipynb
### the best model generated 
https://github.com/fahr-khadija/Breast-Cancer-ML/blob/main/Breast-cancer-project/best_svm_model.pkl

### the code for prediction new data with the best model already generated (best_svm_model.pkl)
https://github.com/fahr-khadija/Breast-Cancer-ML/blob/main/Breast-cancer-project/predict-new-data.ipynb

### Results Analysis And Decision
In our work , we used NN, SVM, and KNN three machine learning techniques to early prevention and detection of breast cancer to find which method performs better that we can used to predict the new data. 

  #### 1-Analyze the performance and evaluation 
 Tableau for accuracy 

![Alt text](https://github.com/fahr-khadija/Breast-Cancer-ML/blob/main/img/image-3.png)

  
   #### • Machines learning techniques performance 

   ###  Neural Network (NN):
         -------------------------------------------------------------------------------
         Results
         Before Optimization:
         Test Set:        Accuracy: 94.32%,Loss: 0.1303
         Training Set: Accuracy: 98.47% ,Loss: 0.0669

         #optimisation 1
         number_input_features = len( X_train_scaled[0])
         hidden_nodes_layer1=80
         hidden_nodes_layer2=30
         hidden_nodes_layer3=21
         model2 = tf.keras.models.Sequential()
         # First hidden layer
         model2.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation='relu'))
         # Second hidden layer
         model2.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation='relu'))
         # Output layer
         model2.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
         # Check the structure of the model
         model2.summary()
         # Compile the model
         model1.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
         # Train the model
          fit_model=model1.fit(X_train_scaled,y_train,validation_split=0.40, epochs=30)

        After Optimization:
        Test Set:       Accuracy: 92.05%,Loss: 0.3176
        Training Set: Accuracy: 100.00%,Loss: 0.0026
      ********   
      
   ##### Analysis:
    The neural network performs well on the test and training sets before optimization.
    After optimization, there's a decrease in test accuracy, indicating potential overfitting 
    during optimization. 
    However, the training accuracy reaches 100%, suggesting the model might be overfitting to the training data.
-------------------------------------------------------------------------------
 
 ###  Support Vector Machine (SVM):
      Before Optimization:
      Test Set:        Accuracy: 93.18%
      Training Set: Accuracy: 98.47% 
      ************************88
          #Optimisation
          #Define the parameter grid
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['linear', 'rbf', 'poly']}
          # Create a GridSearchCV object
          grid_search = GridSearchCV(svm_model, param_grid, cv=3, verbose=3)
          # Fit the GridSearchCV object to the training data
          grid_search.fit(X_train_scaled, y_train)
          # Get the best parameters and the best model from the grid search
          best_params = grid_search.best_params_
          best_model = grid_search.best_estimator_

    
       After Optimization:
       Test Set:        Accuracy: 94.32%
      Training Set: Accuracy: 99.24%
     ********
 Analysis:
    The SVM performs well on both the test and training sets.
    Optimization results in a slight increase in test accuracy, indicating improvement.
    So we can use the optimized model to predict the new Data 

 ###  K Nearest Neighbors (KNN):
        Test Set:Precision: 92%,Recall: 92%,F1-score: 92%,Accuracy: 92%
        Training Set:Precision: 100%,Recall: 100%,F1-score: 100%,Accuracy: 100%
        ********
 Analysis:
      KNN performs well on both the test and training sets, achieving high precision, recall, and accuracy.
   
#### • Machines learning techniques evaluation 
Neural Network (NN) consider the potential overfitting after optimization.we should evaluate 
 other metrics and consider tuning hyperparameters to improve it .
Support Vector Machine (SVM) appears to be a strong performer, especially after optimization.
The model's accuracy on the test and training sets is relatively high.
K Nearest Neighbors (KNN) performs consistently well, showing high precision, recall, and accuracy.  

   #### -Decision:
SVM and KNN both demonstrate strong performance, but SVM exhibits superior accuracy on both the test and training sets. 
Therefore, we have chosen the optimized SVM model for predicting in the new dataset. 

## • Predict on new Data Analyze
      
      Applying the trained model to real-world cases for diagnosis and decision-making.  
      New dataset =Breast_cancer_new
      if you launch the code breast_cancer_final.ipynb in collab enter the new dataset 
      if you want to  launch the file independently of the trainned model  use the code predict-new-data.ipynb   and just integrate your new data instead of ours 'breast_cancer_data_new.csv' ,

             # Read the CSV file from the Resources folder into a Pandas DataFrame
             breast_cancer_new_df = pd.read_csv('breast_cancer_data_new.csv')
             #preprocessing new data with our function preprocess_data
             processed_data = preprocess_data(breast_cancer_new_df)
             # Review the DataFrame
             breast_cancer_new_df.head(10)     
      
            https://github.com/fahr-khadija/Breast-Cancer-ML/blob/main/Breast-cancer-project/predict-new-data.ipynb
      
      
   ###  Accuracy on new dataset: 0.9497716894977168 
   ![Alt text](https://github.com/fahr-khadija/Breast-Cancer-ML/blob/main/img/image-4.png)

  ### Report Analysis      
 Our Result indicate that our model performs well with the new dataset,  achieving a commendable accuracy of approximately 95%. However, a notable concern arises when examining the confusion matrix, revealing 11 instance (5%) in which the model produces both true positives and true negatives. 
This situation implies that the model fails to identify cases where cancer is present or incorrectly predicts the absence of cancer.  

Given the critical nature of this issue and its potential impact on patient outcomes, it is imperative to optimize our model further to mitigate and minimize such occurrences

### Data Analysis Report 
https://github.com/fahr-khadija/Breast-Cancer-ML/blob/main/technical%20report%20analysis.pdf



