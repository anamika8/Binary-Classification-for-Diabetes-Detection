# Comparative Analysis of Naive Bayes, Logistic Regression, and Multi-Layer Neural Network for Predicting Type 2 Diabetes

This project explores the application of three machine learning models—Naive Bayes, Logistic Regression, and Multi-Layer Neural Networks—to predict the likelihood of type 2 diabetes. Using a dataset from the National Institute of Diabetes and Digestive and Kidney Diseases, the project assesses the performance of each model based on accuracy, precision, recall, and calibration plots.

## Project Overview:
The primary goal is to evaluate and compare the performance of Naive Bayes, Logistic Regression, and Multi-Layer Neural Networks (MLNN) on a medical dataset. The dataset includes patient demographics and health indicators, such as cholesterol levels, BMI, and blood sugar levels.

### Dataset:
- The dataset includes 390 entries and various attributes like cholesterol, glucose, BMI, systolic and diastolic blood pressure, waist-to-hip ratio, and the target variable indicating whether a patient has diabetes.

### Models Used:
1. **Naive Bayes Classifier**:
   - Based on Bayes' Theorem, this model assumes independence among features and is suitable for small datasets.
   - Accuracy: 92.3%

2. **Logistic Regression**:
   - A linear model that predicts the probability of diabetes based on input features. Logistic regression is robust with small datasets.
   - Accuracy: 91.0%

3. **Multi-Layer Neural Network (MLNN)**:
   - A deep learning model with two hidden layers, using the `tanh` activation function for hidden layers and `sigmoid` for the output.
   - Accuracy: 92.3%

## Key Findings:
- All models achieved approximately 92% accuracy. **Naive Bayes** had the best calibration and interpretability.
- **Logistic Regression** performed well but slightly underperformed compared to Naive Bayes in terms of calibration.
- **Multi-Layer Neural Networks** had comparable accuracy but showed poor calibration, especially for midrange predicted probabilities.

## Performance Metrics:
- **Confusion Matrix**: All models demonstrated strong true negative and true positive rates, but the number of false positives and false negatives varied between models.
- **Calibration Plots**: Naive Bayes and Logistic Regression showed better calibration curves. The neural network model struggled with calibration.

## Conclusion:
The **Naive Bayes Classifier** proved to be the most effective model, balancing accuracy and calibration. The logistic regression model offered comparable performance but with slightly lower accuracy. The deep learning model, while powerful, was less interpretable and showed poor calibration.

## Future Work:
Future work could include expanding the dataset and exploring additional models, such as Support Vector Machines (SVM) and Random Forests, to compare their performance in medical diagnostics.

## Source Code:
- [Naive Bayes Colab](https://github.com/anamika8/Binary-Classification-for-Diabetes-Detection/blob/main/naivebayes.ipynb)
- [Logistic Regression Colab](https://github.com/anamika8/Binary-Classification-for-Diabetes-Detection/blob/main/logisticregression.ipynb)
- [Multi-Layer Neural Network Colab](https://github.com/anamika8/Binary-Classification-for-Diabetes-Detection/blob/main/multilayer_neural_network.ipynb)

