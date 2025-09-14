# retraining_pipeline.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from feature_engineering import FeatureEngineer
from model_training import ScamDetectorModel

class ModelRetrainingPipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fe = FeatureEngineer()
        
    def collect_feedback_data(self, days_back=30):
        """Collect feedback data from the last N days"""
        from app import db, Question
        
        since = datetime.utcnow() - timedelta(days=days_back)
        
        # Get questions with human feedback
        feedback_data = db.session.query(Question)\
            .filter(Question.timestamp >= since)\
            .filter(Question.human_verified.isnot(None))\
            .all()
        
        data = []
        for q in feedback_data:
            data.append({
                'question': q.content,
                'predicted_score': q.scam_score,
                'human_label': q.human_verified,
                'category': q.category
            })
        
        return pd.DataFrame(data)
    
    def evaluate_current_model(self, test_data):
        """Evaluate current model performance"""
        model_data = joblib.load('scam_detector_model.pkl')
        current_model = model_data['model']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        
        # Prepare test data
        feature_df, _ = self.fe.create_features(test_data)
        X_test = feature_df[feature_columns].fillna(0)
        X_test_scaled = scaler.transform(X_test)
        y_test = test_data['human_label']
        
        # Make predictions
        y_pred = current_model.predict(X_test_scaled)
        y_pred_proba = current_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'samples': len(test_data)
        }
        
        return metrics, y_pred_proba
    
    def should_retrain(self, current_metrics, threshold_f1=0.85):
        """Decide if model should be retrained"""
        if current_metrics['f1'] < threshold_f1:
            self.logger.info(f"Model F1 score {current_metrics['f1']:.3f} below threshold {threshold_f1}")
            return True
        
        if current_metrics['samples'] < 100:
            self.logger.info("Insufficient feedback samples for reliable evaluation")
            return False
        
        return False
    
    def retrain_model(self, training_data):
        """Retrain the model with new data"""
        self.logger.info("Starting model retraining...")
        
        # Combine with existing training data
        existing_data = pd.read_csv('featured_dataset.csv')
        combined_data = pd.concat([existing_data, training_data], ignore_index=True)
        
        # Train new model
        detector = ScamDetectorModel()
        X, y = detector.prepare_data(combined_data)
        detector.train_model(X, y, 'logistic')
        
        # Save new model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_path = f'scam_detector_model_{timestamp}.pkl'
        detector.save_model(new_model_path)
        
        # Backup current model
        import shutil
        shutil.copy('scam_detector_model.pkl', f'scam_detector_model_backup_{timestamp}.pkl')
        
        # Replace current model
        shutil.copy(new_model_path, 'scam_detector_model.pkl')
        
        self.logger.info(f"Model retrained and saved as {new_model_path}")
        return new_model_path
    
    def run_pipeline(self):
        """Run the complete retraining pipeline"""
        try:
            # Collect feedback data
            feedback_data = self.collect_feedback_data()
            
            if len(feedback_data) < 50:
                self.logger.info("Insufficient feedback data for retraining")
                return
            
            # Evaluate current model
            current_metrics, _ = self.evaluate_current_model(feedback_data)
            self.logger.info(f"Current model metrics: {current_metrics}")
            
            # Decide if retraining is needed
            if self.should_retrain(current_metrics):
                # Prepare training data
                feature_df, _ = self.fe.create_features(feedback_data)
                
                # Retrain model
                new_model_path = self.retrain_model(feature_df)
                
                # Validate new model
                new_metrics, _ = self.evaluate_current_model(feedback_data)
                self.logger.info(f"New model metrics: {new_metrics}")
                
                # Deploy if improved
                if new_metrics['f1'] > current_metrics['f1']:
                    self.logger.info("New model performs better - deploying")
                    # Deployment logic here
                else:
                    self.logger.warning("New model doesn't improve performance - reverting")
                    # Revert to backup
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")

# Automated retraining scheduler
def schedule_retraining():
    """Schedule automatic retraining"""
    import schedule
    import time
    
    pipeline = ModelRetrainingPipeline()
    
    # Schedule daily evaluation, weekly retraining if needed
    schedule.every().day.at("02:00").do(pipeline.run_pipeline)
    
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour

if __name__ == "__main__":
    schedule_retraining()