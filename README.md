# Multi-task Sentiment Analysis Model

This project demonstrates a multi-task deep learning model built using TensorFlow and Keras for performing sentiment analysis on textual data. The model jointly performs two tasks:

Classification Task: Predicts whether a given text has a positive or negative sentiment (binary classification).
Regression Task: Predicts a continuous sentiment score, ranging from 0 to 1, for the given text.
The model utilizes a shared LSTM layer to capture the underlying textual features and then branches into two separate outputs for classification and regression.

Dataset
The dataset contains customer comments with two corresponding labels:

sentiment_label: A binary label (0 for negative sentiment, 1 for positive sentiment).
sentiment_score: A continuous score between 0 and 1 representing the intensity of the sentiment.
An example of the dataset format (sample_comments.csv):

csv
Copy code
text,sentiment_label,sentiment_score
"This product is amazing!",1,0.95
"I hate this product.",0,0.1
...
Model Architecture
The model architecture includes:

Embedding Layer: To convert the input sequences of tokens into dense vector representations.
LSTM Layer: A Long Short-Term Memory layer that captures temporal dependencies in the input sequences.
Dropout Layer: To prevent overfitting by randomly setting input units to 0 with a frequency of 50%.
Classification Output: A Dense layer with sigmoid activation to predict binary sentiment.
Regression Output: A Dense layer with linear activation to predict the sentiment score.
