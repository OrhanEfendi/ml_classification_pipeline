# ml_classification_pipeline

# Machine Learning Binary classification end-to-end pipeline

The sinking of the Titanic is one of the most infamous shipwrecks in history.On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
![titanic](https://user-images.githubusercontent.com/100521892/190914869-4fef700e-56a6-4fe6-bd35-f0e42ab61d3e.jpg)
# Overview
![mod](https://user-images.githubusercontent.com/100521892/192150061-a8b95c1a-c124-46f7-93b9-d91cd1423b9f.PNG)



# Results

Since we have an imnbalanced data set. I have found the best models with imbalanced data sets. Since it is an unbalanced dataset, I used balanced_accuracy, matthew score, f1 score, precison and recall as scores since accuracy score did not give accurate results. I used RandomForesClassifier in this dataset. To improve its parameters, I used a hyperparameter optimization. For this I used HalvingGridSearch, HalvingRandomsearch, Optuna

# Project Structure

├── data                   
├── docs                    
├── featureselection                   
├── imbalance                    
├── impute                  
├── Modelselection                  
├── Preprocessing                  
├── LICENSE

└── README
