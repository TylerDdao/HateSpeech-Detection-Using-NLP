
# Hate Speech Detection on Twitter comments using NLP library

## Overview

This project involves developing a machine learning model to detect hate speech on Twitter. The goal is to classify tweets as hate speech (HS=1) or non-hate speech (HS=0) using a logistic regression model. The dataset used for training and evaluation contains labeled tweets.

## Project Structure

The repository contains the following files:

- **TrainingData.csv**: The dataset containing labeled tweets.
- **model_training.py**: Script for preprocessing data, training the logistic regression model, and evaluating it.
- **hate_speech_model.pkl**: Saved trained model.
- **vectorizer.pkl**: Saved vectorizer for transforming text data during inference.
- **README.md**: Project documentation.

## Requirements

Ensure you have Python 3.x installed and set up with the following libraries:

```
pip install pandas numpy scikit-learn seaborn joblib
```

## Data Preprocessing

### Lowercasing and Punctuation Removal:
- All tweets are converted to lowercase.
- Punctuation is removed using regex.

### Label Mapping:
- Class labels are mapped from 1 and 2 to 1 (Hate Speech) and 0 (Not Hate Speech).

### Text Vectorization:
- **CountVectorizer** is used to convert text data into a numerical format using a bag-of-words model.

## Model Training

- **Logistic Regression** was chosen for its simplicity and efficiency in text classification tasks.
- The dataset was split into 80% training and 20% testing.
- The model was evaluated using metrics such as Accuracy, Precision, Recall, and F1-Score.

### Results
- **Accuracy**: 95%
- The model showed a higher recall and F1-score for hate speech detection.
- It performed well in identifying hate speech, but non-hate speech had slightly lower scores due to class imbalance.

## How to Run

1. Clone the repository:
   ```
   git clone <repository_url>
   cd hate_speech_detection
   ```

2. Run the model training script:
   ```
   python model_training.py
   ```

3. View the classification results.
   - The model and vectorizer are saved as .pkl files for future use.

## Future Improvements

- Implement data balancing techniques to address class imbalance.
- Test other algorithms like SVM or Neural Networks.
- Perform hyperparameter tuning for better results.

## Author

Tyler

Feel free to reach out for any questions or further clarifications!
