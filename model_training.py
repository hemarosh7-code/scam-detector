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

class ScamDetectorModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.tfidf_vectorizer = None
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Select feature columns (exclude text and target columns)
        exclude_cols = ['question', 'clean_question', 'is_scam', 'scam_score', 
                       'timestamp', 'user_id', 'category', 'source']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns]
        
        # Handle missing values
        X = X.fillna(0)
        
        # Convert scam_score to binary classification
        y = (df['scam_score'] > 50).astype(int)
        
        return X, y
    
    def train_model(self, X, y, model_type='logistic'):
        """Train the scam detection model"""
        print(f"Training {model_type} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select and train model
        if model_type == 'logistic':
            self.model = LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
            
            # Grid search for best parameters
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                class_weight='balanced',
                random_state=42
            )
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            
        elif model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(random_state=42)
            
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, 
            scoring='roc_auc', n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc'
        )
        print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return X_test, y_test, y_pred, y_pred_proba
    
    def plot_results(self, y_test, y_pred, y_pred_proba):
        """Plot model results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[0,1].plot(fpr, tpr, label=f'ROC-AUC = {roc_auc_score(y_test, y_pred_proba):.4f}')
        axes[0,1].plot([0, 1], [0, 1], 'k--')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        
        # Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns[:20],  # Top 20 features
                'importance': self.model.feature_importances_[:20]
            }).sort_values('importance', ascending=False)
            
            sns.barplot(data=feature_importance, x='importance', y='feature', ax=axes[1,0])
            axes[1,0].set_title('Top 20 Feature Importances')
        
        # Probability Distribution
        axes[1,1].hist(y_pred_proba[y_test == 0], alpha=0.7, label='Legitimate', bins=30)
        axes[1,1].hist(y_pred_proba[y_test == 1], alpha=0.7, label='Scam', bins=30)
        axes[1,1].set_xlabel('Predicted Probability')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Probability Distribution')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_scam_score(self, text):
        """Predict scam score for a given text"""
        if not self.model or not self.scaler:
            raise ValueError("Model not trained yet!")
        
        # Create temporary dataframe
        temp_df = pd.DataFrame({'question': [text]})
        
        # Extract features
        fe = FeatureEngineer()
        feature_df, _ = fe.create_features(temp_df)
        
        # Prepare features
        X = feature_df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        scam_probability = self.model.predict_proba(X_scaled)[0, 1]
        scam_score = int(scam_probability * 100)
        
        return scam_score
    
    def save_model(self, filepath='scam_detector_model.pkl'):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='scam_detector_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        print(f"Model loaded from {filepath}")

# Usage
df = pd.read_csv('scam_detection_dataset.csv')
detector = ScamDetectorModel()

X, y = detector.prepare_data(df)
X_test, y_test, y_pred, y_pred_proba = detector.train_model(X, y, 'logistic')
detector.plot_results(y_test, y_pred, y_pred_proba)
detector.save_model()

# Test prediction
test_text = "How can I make $10000 in 30 days with guaranteed returns?"
score = detector.predict_scam_score(test_text)
print(f"Scam score: {score}")

import joblib

# Save model
joblib.dump(model, "scam_detector_model.pkl")
print("✅ Model saved as scam_detector_model.pkl")
