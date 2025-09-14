# feature_engineering.py
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.scam_indicators = [
            r'\b(?:guaranteed?|guarantee)\b',
            r'\b(?:risk[- ]?free|no[- ]?risk)\b',
            r'\b(?:easy[- ]?money|quick[- ]?cash)\b',
            r'\b(?:double|triple|quadruple).{0,20}money\b',
            r'\b(?:\d+%|\d+\s*percent).{0,20}(?:return|profit|gain)\b',
            r'\b(?:limited[- ]?time|act[- ]?now|hurry)\b',
            r'\b(?:insider|secret|exclusive).{0,20}(?:tip|info|deal)\b',
            r'\b(?:binary[- ]?option|forex[- ]?robot|crypto[- ]?bot)\b',
            r'\b(?:ponzi|pyramid|mlm|network[- ]?marketing)\b',
            r'\b(?:work[- ]?from[- ]?home|passive[- ]?income)\b'
        ]
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_scam_indicators(self, text):
        """Extract scam indicator features"""
        if pd.isna(text):
            text = ""
        
        text = text.lower()
        features = {}
        
        # Count scam indicator patterns
        for i, pattern in enumerate(self.scam_indicators):
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            features[f'scam_indicator_{i}'] = matches
        
        # Total scam indicators
        features['total_scam_indicators'] = sum(features.values())
        
        # Money-related patterns
        money_patterns = [r'\$\d+', r'\d+\s*(?:dollars?|bucks?)', r'\d+k', r'\d+m']
        features['money_mentions'] = sum([len(re.findall(p, text)) for p in money_patterns])
        
        # Urgency indicators
        urgency_words = ['now', 'today', 'immediately', 'urgent', 'hurry', 'fast', 'quick']
        features['urgency_score'] = sum([text.count(word) for word in urgency_words])
        
        # Question marks (desperation indicator)
        features['question_marks'] = text.count('?')
        
        return features
    
    def extract_linguistic_features(self, text):
        """Extract linguistic features"""
        if pd.isna(text):
            return {}
        
        blob = TextBlob(text)
        tokens = word_tokenize(text.lower())
        
        features = {}
        features['text_length'] = len(text)
        features['word_count'] = len(tokens)
        features['avg_word_length'] = np.mean([len(word) for word in tokens]) if tokens else 0
        features['sentence_count'] = len(blob.sentences)
        features['polarity'] = blob.sentiment.polarity
        features['subjectivity'] = blob.sentiment.subjectivity
        
        # Capital letter ratio
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Exclamation marks
        features['exclamation_count'] = text.count('!')
        
        return features
    
    def create_features(self, df):
        """Create all features for the dataset"""
        print("Creating features...")
        
        # Clean text
        df['clean_question'] = df['question'].apply(self.clean_text)
        
        # Extract scam indicators
        scam_features = df['question'].apply(self.extract_scam_indicators)
        scam_df = pd.DataFrame(list(scam_features))
        
        # Extract linguistic features
        ling_features = df['question'].apply(self.extract_linguistic_features)
        ling_df = pd.DataFrame(list(ling_features))
        
        # Combine all features
        feature_df = pd.concat([df, scam_df, ling_df], axis=1)
        
        # TF-IDF features
        tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_features = tfidf.fit_transform(feature_df['clean_question'])
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        final_df = pd.concat([feature_df, tfidf_df], axis=1)
        
        return final_df, tfidf

# Usage
fe = FeatureEngineer()
df = pd.read_csv('scam_detection_dataset.csv')
feature_df, tfidf_vectorizer = fe.create_features(df)
feature_df.to_csv('featured_dataset.csv', index=False)