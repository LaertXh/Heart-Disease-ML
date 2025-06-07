# Healthcare Risk Prediction- Multi-Model Analysis

## Project Overview

This project applies multiple machine learning models to predict heart disease using the Framingham dataset. It includes structured code for data preprocessing, model training, and evaluation.

## Dataset

- **Source:** Framingham Heart Study
- **Features:** Various cardiovascular risk factors
- **Target Variable:** Presence or absence of heart disease

## Project Structure

```
Heart-Disease-ML/
│── data/                # Dataset storage
│── src/                 # Source code for models
    |── all_models.ipynb # notebook file with all models
    |── specific-models/
        │── imports.py           # All required imports
        │── data_cleaning.py     # Data preprocessing & train-test split
        │── logisticregression_training.py  # Logistic Regression training
        |── kneighborsclassifier_training.py  # KNN model training
        │── randomforestclassifier_training.py  # Random Forest model training
        |── svc_training.py # SVC model training
│── results/             # Model evaluation results
│── requirements.txt  # Required dependencies
│── README.md          # Project documentation
```

## Installation & Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/LaertXh/Heart-Disease-ML.git
   cd Heart-Disease-ML
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project**

   You can simply run the all_models file in the src folder to execute every model and their performances.
   Or you can go to each specific model file within the src/specific-models folder and run them seperately

## Model Performance

| Model                     | Accuracy |
| ------------------------- | -------- |
| Logistic Regression       | 85.2%    |
| K-Nearest Neighbors       | 84.4%    |
| Random Forest             | 85.2%    |
| Support Vector Classifier | 83.3%    |

## Conclusion

The highest accuracy models were **Logistic Regression** and **Random Forest** , both achieving **85.2% accuracy** . The project demonstrates the effectiveness of various classifiers for heart disease prediction. Please note that the model results were limited by the data set given. It was meant so that default parameters would yield low initial results. This lead to me messing around with parameters to improve it, which demonstrates the power of training under the correct parameters.

## Contact

**Laert Xhumari**

[LaertXhumari@gmail.com](mailto:LaertXhumari@gmail.com)

[GitHub Profile](https://github.com/LaertXh)
