import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def data_process():
    data=pd.read_csv("data/sports-or-politics.csv")
    print("columns exists in data set : ",data.columns.tolist())
    data.columns=[col.strip().lower().replace(" ","_") for col in data.columns]
    
    data=data[data["class_index"].isin([1,2])]
    data["label"]=data["class_index"].map({1:"news",2:"sports"})
    
    data["full_text"]=(data["title"].astype(str)+" "+data["description"].astype(str))
    texts=data["full_text"]
    labels=data["label"]
    return texts,labels

def models_4_clsif(X_train,X_test,Y_train,Y_test,feature_name):
    models={
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=999),
        "AdaBoost": AdaBoostClassifier(n_estimators=111,random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    print("\n    ")
    print(f"Feature Representation: {feature_name}")
    
    for name, model in models.items():
        model.fit(X_train,Y_train)
        pred=model.predict(X_test)
        acc=accuracy_score(Y_test,pred)
        print(f"{name:<20} Accuracy: {acc:.4f}")
        
def clsifr():
    texts,labels=data_process()
    
    x_train,x_test,y_train,y_test=train_test_split(texts,labels,test_size=0.1,random_state=42)
    
    tf_idf=TfidfVectorizer(stop_words="english")
    x_train_tfidf=tf_idf.fit_transform(x_train)
    x_test_tfidf = tf_idf.transform(x_test)
    models_4_clsif(x_train_tfidf, x_test_tfidf, y_train, y_test, "TF-IDF")

    
    ngram=TfidfVectorizer(stop_words="english",ngram_range=(1,2))
    x_train_ngram=ngram.fit_transform(x_train)
    x_test_ngram=ngram.transform(x_test)
    models_4_clsif(x_train_ngram, x_test_ngram, y_train, y_test, "n-grams (1,2)")
    
    bag=CountVectorizer(stop_words="english")
    x_train_bag=bag.fit_transform(x_train)
    x_test_bag=bag.transform(x_test)
    models_4_clsif(x_train_bag,x_test_bag,y_train,y_test,"Bag of Words")
    
if __name__=="__main__":
    clsifr()
