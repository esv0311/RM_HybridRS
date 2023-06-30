# RM Hybrid Movie Recommendation System
## Data Acquisition
In this project, the dataset was acquired from the Kaggle MovieLens 100k dataset.  The data was collected over a period of seven months, from September 19th, 1997 to April 22nd, 1998, through the MovieLens website. The dataset provided includes various files with information about movie ratings, users, items (movies), genres, and occupations:
- u.data : the full dataset with 100,000 ratings by 943 users on 1,682 items. It includes information such as user ID, item ID, rating, and timestamp.
- u.item : the movies along with the information, including movie ID, title, release date, IMDb URL, and genre information. The genre information is represented by binary values (1 or 0) indicating whether a movie belongs to a particular genre.
- u.user : This file contains demographic information about the users, including user ID, age, gender, occupation, and zip code.

## Data Preprocessing
Prior to analysis, the dataset underwent a preprocessing stage, such as:
- Checking for missing values
- Checking for duplicated rows 
- Removing the 'timestamp' column from the ratings data frame 
These acquisition and preprocessing steps ensure the dataset is clean and ready for further analysis.

## Feature Engineering
Feature engineering refers to the process of transforming raw data into meaningful features that can be used for machine learning models. 
- Extracting movie genres from the data frame
- Upon closer examination, there are a few implicit feature engineering steps in the code, such as creating utility matrices for training and test data and computing genre similarities using cosine similarity 

## Training
To train the performance of our recommendation models, we split our dataset into two subsets: 60% training data which is used to train our model, and 40% test data to test and evaluate our model. After partitioning the dataset, we designed the 3 models (collaborative, content-based, and hybrid model) and fitted them:
- Content-based recommendation system using the KNN algorithm 
- Collaborative filtering recommendation system using SVD algorithm 
- Hybrid recommendation system by fitting both content-based and collaborative filtering algorithms 

## Evaluation of The Model and Results
- Generating predictions and calculating RMSE and FCP values for the content-based model
  The model's training RMSE of 0.9552 suggests it fits the training data quite well. The test RMSE, on the other hand, suggests a higher error level, with an RMSE of 1.1460. This may result from limitations in the content-based filtering approach, such as its reliance on movie attributes and the lack of user-item interactions. The model also obtained an FCP value of 0.590981 which indicates that the model only captures moderate concordance with user preferences when ranking the movie dataset based on the movie’s content similarities. In the testing set, the FCP value decreased to 0.440255, suggesting a lower ability to rank movies accurately for unseen user preferences.
- Generating predictions and calculating RMSE and FCP values for the collaborative filtering model
  The model’s training RMSE of 0.9006 shows that it effectively captures the underlying patterns in the training dataset with a relatively low inaccuracy. However, the test RMSE of 0.9446 suggests a slightly higher error level in predicting movies for unseen data, which could indicate a slight overfitting of the model to the training data. The model also obtained an FCP value of 0.733433 in the training data, indicating a higher level of concordance with user preferences than the CB model. In the testing set, the value slightly decreases to 0.701847, which implies that the model maintains relatively consistent ranking accuracy.
- Generating predictions and calculating RMSE and FCP values for the hybrid model
  The training RMSE of 0.7801, which is the lowest among the three models, suggests a good fit of the hybrid model to the training data, indicating a low error level. A little higher than the training RMSE, the test RMSE of 0.9977 demonstrates a respectably accurate prediction performance on the unobserved data. This is further emphasized by the FCP value of 0.751256 in the training set, which surpasses both CB and CF models, and the FCP value of 0.641928 in the testing set, which strikes a balance between those 2 models. The hybrid model achieves a trade-off between the 2 individual models, delivering moderate ranking accuracy and concordance in unseen scenarios.
- Plotting the RMSE and FCP values for the different models
  RMSE Plot:
  ![RMSE Values by Models](https://github.com/esv0311/RM_HybridRS/blob/main/RMSEmodels.png?raw=true)
  FCP Plot:
  ![FCP Values by Models](https://github.com/esv0311/RM_HybridRS/blob/main/FCPmodels.png?raw=true)
