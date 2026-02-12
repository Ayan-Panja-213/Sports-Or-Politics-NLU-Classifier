# Sports-Or-Politics-NLU-Classifier

Overview---------------------------------
This project implements a Natural Language Understanding (NLU) based text classification system that classifies news articles into Sports or Politics using machine learning techniques.

Problem Statement-------------------------
Given a text document consisting of a title and description, the task is to classify the document as either Sports or Politics.

Dataset Description-----------------------
The dataset contains news articles with the following columns:
Class Index
Title
Description
Only two classes are used in this project:
Class 1: Politics (News)
Class 2: Sports
The dataset was reduced from the original version to keep the file size small and suitable for GitHub upload.
Dataset location:
data/sports-or-politics.csv

Feature Representation Techniques------------
The following text feature extraction methods are used:
Bag of Words
TF-IDF
N-grams (unigram and bigram)

Machine Learning Models Used------------------
The following classifiers are implemented and compared:
Multinomial Naive Bayes
Logistic Regression
AdaBoost Classifier
Gradient Boosting Classifier

Experimental Setup-----------------------------
Input text is created by combining Title and Description
Dataset split:
90 percent training data
10 percent testing data
Evaluation metric used:
Accuracy
Results
The accuracy of each classifier is printed for different feature representations.
The results show that Logistic Regression,Naive Bayes and Gradient Boosting perform better compared to AdaBoost in most cases.

Project Structure--------------------------------
Sports-Or-Politics-NLU-Classifier
│
├── src
│ └── M23MA2004_prob4.py
│
├── data
│ └── sports-or-politics.csv
│
└── README.md

How to Run the Code--------------------------------
Install required libraries
pandas
scikit-learn
Run the program
python src/M23MA2004_prob4.py

Limitations------------------------------------------
Only classical machine learning models are used
No deep learning or semantic embeddings are applied
Performance depends on dataset size and quality

Future Scope-----------------------------------------
Use word embeddings such as Word2Vec or GloVe
Apply deep learning models like LSTM or BERT
Perform hyperparameter tuning
Add evaluation metrics such as precision, recall, and F1-score

Author
Ayan Panja
MSc–MTech Dual Degree in Mathematics and Data Computational Scienece
IIT Jodhpur
