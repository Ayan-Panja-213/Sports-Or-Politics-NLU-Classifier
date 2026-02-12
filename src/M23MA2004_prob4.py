import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# FUNCTION for DATA PROCESSING
def data_process():
    data=pd.read_csv("sports-or-politics.csv")
    print("columns exists in data set : ",data.columns.tolist())
    
    # NORMALIZE COLUMN NAMES
    data.columns=[col.strip().lower().replace(" ","_") for col in data.columns]
    
    # NUMERIC to CLASS NAME MAPPING
    data["label"]=data["class_index"].map({1:"politics",2:"sports"})
    
    # COMBINING TITLE & DESCRIPTION INTO SINGLE TEXT
    data["full_text"]=(data["title"].astype(str)+" "+data["description"].astype(str))
    texts=data["full_text"]
    labels=data["label"]
    return texts,labels

# FUNCTION for EVALUATING in DIFFERENT ML models                                                   MODELS
def models_4_clsif(X_train,X_test,Y_train,Y_test,feature_name):
    models={
        "Naive Bayes": MultinomialNB(),                                                          # Naive Bayes
        "Logistic Regression": LogisticRegression(max_iter=999),                                 # Logistic Regression
        "AdaBoost": AdaBoostClassifier(n_estimators=111,random_state=42),                        # AdaBoost
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)                         # Gradient Boosting
    }
    print("\n    ")
    print(f"Feature Representation: {feature_name}")
    
    for name, model in models.items():
        model.fit(X_train,Y_train)
        pred=model.predict(X_test)
        acc=accuracy_score(Y_test,pred)
        print(f"{name:<20} Accuracy: {acc:.4f}")
        
# FUNCTION for FEATURE EXTRACTION & EVALUATION        
def clsifr():
    texts,labels=data_process()
    
    # TRAIN & TEST SPLIT 75% TRAIN 25% TEST
    x_train,x_test,y_train,y_test=train_test_split(texts,labels,test_size=0.25,random_state=42)
    
    #                   TF-IDF
    tf_idf=TfidfVectorizer(stop_words="english")
    x_train_tfidf=tf_idf.fit_transform(x_train)
    x_test_tdidf=tf_idf.transform(x_test)
    models_4_clsif(x_train_tfidf,x_test_tdidf,y_train,y_test,"TF-IDF")
    
    #                  n-grams(1,2)
    ngram=TfidfVectorizer(stop_words="english",ngram_range=(1,2))
    x_train_ngram=ngram.fit_transform(x_train)
    x_test_ngram=ngram.transform(x_test)
    models_4_clsif(x_train_ngram, x_test_ngram, y_train, y_test, "n-grams (1,2)")
    
    #                 Bag of Words 
    bag=CountVectorizer(stop_words="english")
    x_train_bag=bag.fit_transform(x_train)
    x_test_bag=bag.transform(x_test)
    models_4_clsif(x_train_bag,x_test_bag,y_train,y_test,"Bag of Words")
    
if __name__=="__main__":
    clsifr()
