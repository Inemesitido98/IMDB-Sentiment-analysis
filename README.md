# Project Title: IMDB-Sentiment-analysis

## Overview
This project focuses on sentiment analysis, aiming to classify text data into positive, and negative sentiments. I experimented with various machine learning models to evaluate their performance on this task.

## Objectives
- To preprocess and clean text data for sentiment analysis.
- To compare the performance of different machine learning models for text classification.
- 
- To evaluate the models based on accuracy, precision, recall, and F1 score.

## Models Used
- Logistic Regression
- Support Vector Classifier (SVC)
- Random Forest
- Naive Bayes 

## Approach
1. **Data Preprocessing**:
   - Removed stopwords, punctuation, and special characters.
   - Tokenized and vectorized the text using techniques such as CountVectorizer and TF-IDF.
2. **Model Training and Evaluation**:
   - Trained and tested multiple models on the dataset.
   - Evaluated models using accuracy, precision, recall, and F1-score metrics.
3. **Insights**:
   - Analyzed the strengths and weaknesses of each model and identified the best-performing one.

## Results
- The **Linear Support Vector Classifier model** achieved the highest accuracy of 0.8674.
- The **Logistic Regression model** achieved the second to the highest accuracy of 0.8658.
- The **Random Forest** model is the least performing model across most metrics.
- The **Naive Bayes** was interpretable but had lower accuracy than other models.
  
  N.B: is the least performing model across most metrics
## File Structure
- `sentiment_analysis.ipynb`: Jupyter notebook containing all the code and detailed explanations in markdown cells.



