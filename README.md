# Comparative Analysis of Naive Bayes, Logistic Regression, and Multi-Layer Neural Network for Predicting Type 2 Diabetes

This project explores the application of three machine learning models—Naive Bayes, Logistic Regression, and Multi-Layer Neural Networks—to predict the likelihood of type 2 diabetes. Using a dataset from the National Institute of Diabetes and Digestive and Kidney Diseases, the project aims to assess the performance of each model based on accuracy, precision, recall, and calibration plots.

## Project Overview:
The primary goal of the project is to evaluate and compare the performance of Naive Bayes, Logistic Regression, and Multi-Layer Neural Networks (MLNN) on a medical dataset. The dataset includes patient demographics and health indicators, such as cholesterol levels, BMI, and blood sugar levels. The project focuses on identifying which model is the most effective for binary classification tasks, particularly in medical diagnostics.

## Dataset:
The dataset used includes 390 entries and various attributes like cholesterol, glucose, BMI, systolic and diastolic blood pressure, waist-to-hip ratio, and the target variable indicating whether a patient has diabetes. It was sourced from the National Institute of Diabetes and Digestive and Kidney Diseases.  
**Dataset source**: [Kaggle - Predict Diabetes Based on Diagnostic Measures](https://www.kaggle.com/datasets/houcembenmansour/predict-diabetes-based-on-diagnostic-measures).

## Models Used:
### 1. Naive Bayes Classifier:
- Based on Bayes' Theorem, this model assumes independence among features and is suitable for small datasets. It uses a combination of categorical and Gaussian distributions.
- **Accuracy**: 92.3%  
- **Package**: [Mixed Naive Bayes](https://pypi.org/project/mixed-naive-bayes/).

### 2. Logistic Regression:
- A linear model that predicts the probability of diabetes based on input features. Logistic regression is well-suited for binary classification problems and is robust with small datasets.
- **Accuracy**: 91.0%  
- **Reference**: [Logistic Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).

### 3. Multi-Layer Neural Network (MLNN):
- This deep learning model consists of two hidden layers, utilizing the `tanh` activation function for hidden layers and the `sigmoid` function for the output. It is designed to handle complex, non-linear relationships.
- **Accuracy**: 92.3%  
- **Reference**: [A comprehensive review of machine learning techniques on diabetes detection](https://link.springer.com/article/10.1186/s42492-021-00097-7).

## Key Findings:
- All models achieved approximately 92% accuracy on the test set. However, Naive Bayes had the best calibration and interpretability, making it the most appropriate model for the dataset.
- Logistic Regression performed well but slightly underperformed compared to Naive Bayes in terms of calibration.
- Multi-Layer Neural Networks provided comparable accuracy but showed poor calibration, especially for midrange predicted probabilities.

## Performance Metrics:
- **Confusion Matrix**: All models demonstrated strong true negative and true positive rates, but the number of false positives and false negatives varied between models.
- **Calibration Plots**: Naive Bayes and Logistic Regression had better calibration curves, indicating that their predicted probabilities were more reliable. The neural network model struggled with calibration, overestimating the likelihood of diabetes in some cases.

## Conclusion:
The Naive Bayes Classifier was found to be the most effective model for this dataset, balancing accuracy and calibration. The logistic regression model offered comparable performance but slightly lower accuracy. The deep learning model, while powerful, was less interpretable and showed poor calibration, making it less suited for small, structured datasets like this one.

## Future Work:
Future work could include expanding the dataset to improve model generalization and exploring additional models, such as Support Vector Machines (SVM) and Random Forests, to compare their performance in medical diagnostics.

## Source Code:
- [Naive Bayes Colab](https://github.com/anamika8/Binary-Classification-for-Diabetes-Detection/blob/main/naivebayes.ipynb)
- [Logistic Regression Colab](https://github.com/anamika8/Binary-Classification-for-Diabetes-Detection/blob/main/logisticregression.ipynb)
- [Multi-Layer Neural Network Colab](https://github.com/anamika8/Binary-Classification-for-Diabetes-Detection/blob/main/multilayer_neural_network.ipynb)

