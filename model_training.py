# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # headless backend, avoids GUI memory leaks


from sklearn.feature_extraction.text import TfidfVectorizer

class ScamDetectorModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.vectorizer = None   # store tfidf vectorizer
    
    def prepare_data(self, df):
    	"""Prepare data using TF-IDF on available text column"""
    	# Change this based on your actual column name
    	texts = df['question'].fillna("")  
    
    	self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    	X = self.vectorizer.fit_transform(texts)

    	# Adjust based on actual label column
    	y = (df['scam_score'] > 50).astype(int)  
    
    	return X, y

    
    def train_model(self, X, y, model_type='logistic'):
        print(f"Training {model_type} model...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                class_weight='balanced',
                solver="liblinear",
                max_iter=1000
            )
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2']
            }
        else:
            raise ValueError("Only logistic is supported in this text mode")
        
        grid = GridSearchCV(self.model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        
        print("Best parameters:", grid.best_params_)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
        
        return X_test, y_test, y_pred, y_pred_proba
    
    def predict_scam_score(self, text):
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)
        return proba[0,1]
    
    def save_model(self, filepath="scam_detector_model.pkl"):
        joblib.dump({
            "model": self.model,
            "vectorizer": self.vectorizer
        }, filepath)
        print(f"Model + vectorizer saved to {filepath}")
    
    def load_model(self, filepath="scam_detector_model.pkl"):
        data = joblib.load(filepath)
        self.model = data["model"]
        self.vectorizer = data["vectorizer"]
        print(f"Loaded model + vectorizer from {filepath}")

# Usage
df = pd.read_csv('scam_detection_dataset.csv')
detector = ScamDetectorModel()

X, y = detector.prepare_data(df)
X_test, y_test, y_pred, y_pred_proba = detector.train_model(X, y)

detector.save_model()

# Reload model + test prediction
detector.load_model()
test_text = "How can I make $10000 in 30 days with guaranteed returns?"
print("Scam score:", detector.predict_scam_score(test_text))

# Test prediction
test_text = "How can I make $10000 in 30 days with guaranteed returns?"
score = detector.predict_scam_score(test_text)
print(f"Scam score: {score}")

import joblib

# After fitting vectorizer
joblib.dump(detector.vectorizer, "vectorizer.pkl")
print("Vectorizer saved to vectorizer.pkl")
