# maintenance.py
import os
import pandas as pd
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine
import joblib

class MaintenanceService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_url = os.getenv('DATABASE_URL')
        self.engine = create_engine(self.db_url)
    
    def cleanup_old_data(self, days_to_keep=90):
        """Clean up old data from database"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        with self.engine.connect() as conn:
            # Delete old questions
            result = conn.execute(
                "DELETE FROM questions WHERE timestamp < %s AND flagged = FALSE",
                (cutoff_date,)
            )
            self.logger.info(f"Deleted {result.rowcount} old questions")
            
            # Keep flagged questions longer (6 months)
            flagged_cutoff = datetime.utcnow() - timedelta(days=180)
            result = conn.execute(
                "DELETE FROM questions WHERE timestamp < %s AND flagged = TRUE",
                (flagged_cutoff,)
            )
            self.logger.info(f"Deleted {result.rowcount} old flagged questions")
    
    def optimize_database(self):
        """Optimize database performance"""
        with self.engine.connect() as conn:
            # Update statistics
            conn.execute("ANALYZE questions")
            
            # Reindex if needed
            conn.execute("REINDEX INDEX idx_questions_timestamp")
            conn.execute("REINDEX INDEX idx_questions_scam_score")
            
            self.logger.info("Database optimization completed")
    
    def backup_model_data(self):
        """Backup model and training data"""
        import shutil
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"/backup/models/{timestamp}"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup model files
        for model_file in ['scam_detector_model.pkl', 'featured_dataset.csv']:
            if os.path.exists(model_file):
                shutil.copy(model_file, backup_dir)
        
        # Backup recent predictions for analysis
        with self.engine.connect() as conn:
            df = pd.read_sql(
                "SELECT * FROM questions WHERE timestamp > NOW() - INTERVAL '7 days'",
                conn
            )
            df.to_csv(f"{backup_dir}/recent_predictions.csv", index=False)
        
        self.logger.info(f"Model backup completed: {backup_dir}")
    
    def generate_performance_report(self):
        """Generate weekly performance report"""
        with self.engine.connect() as conn:
            # Get statistics for the last week
            stats = pd.read_sql("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total_questions,
                    AVG(scam_score) as avg_score,
                    COUNT(CASE WHEN scam_score >= 70 THEN 1 END) as high_risk,
                    COUNT(CASE WHEN scam_score BETWEEN 40 AND 69 THEN 1 END) as medium_risk,
                    COUNT(CASE WHEN scam_score < 40 THEN 1 END) as low_risk
                FROM questions 
                WHERE timestamp > NOW() - INTERVAL '7 days'
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, conn)
        
        # Generate report
        report = {
            'period': '7 days',
            'generated_at': datetime.utcnow().isoformat(),
            'summary': {
                'total_questions': int(stats['total_questions'].sum()),
                'avg_scam_score': float(stats['avg_score'].mean()),
                'high_risk_count': int(stats['high_risk'].sum()),
                'medium_risk_count': int(stats['medium_risk'].sum()),
                'low_risk_count': int(stats['low_risk'].sum())
            },
            'daily_stats': stats.to_dict('records')
        }
        
        # Save report
        report_file = f"reports/performance_report_{datetime.now().strftime('%Y%m%d')}.json"
        os.makedirs('reports', exist_ok=True)
        
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance report generated: {report_file}")
        return report
    
    def check_model_drift(self):
        """Check for model drift and alert if detected"""
        with self.engine.connect() as conn:
            # Compare recent predictions with historical baseline
            recent_stats = pd.read_sql("""
                SELECT AVG(scam_score) as recent_avg
                FROM questions 
                WHERE timestamp > NOW() - INTERVAL '7 days'
            """, conn)
            
            historical_stats = pd.read_sql("""
                SELECT AVG(scam_score) as historical_avg
                FROM questions 
                WHERE timestamp BETWEEN NOW() - INTERVAL '30 days' AND NOW() - INTERVAL '7 days'
            """, conn)
        
        recent_avg = recent_stats['recent_avg'].iloc[0]
        historical_avg = historical_stats['historical_avg'].iloc[0]
        
        if recent_avg is not None and historical_avg is not None:
            drift_threshold = 0.15  # 15% change
            relative_change = abs(recent_avg - historical_avg) / historical_avg
            
            if relative_change > drift_threshold:
                self.logger.warning(f"Model drift detected: {relative_change:.2%} change in average score")
                # Send alert (implement notification logic)
                return True
        
        return False
    
    def run_maintenance(self):
        """Run all maintenance tasks"""
        self.logger.info("Starting maintenance tasks...")
        
        try:
            self.cleanup_old_data()
            self.optimize_database()
            self.backup_model_data()
            self.generate_performance_report()
            
            if self.check_model_drift():
                self.logger.warning("Model drift detected - consider retraining")
            
            self.logger.info("Maintenance tasks completed successfully")
            
        except Exception as e:
            self.logger.error(f"Maintenance failed: {str(e)}")

# Scheduled maintenance
def schedule_maintenance():
    """Schedule regular maintenance tasks"""
    import schedule
    import time
    
    maintenance = MaintenanceService()
    
    # Daily cleanup and optimization
    schedule.every().day.at("01:00").do(maintenance.cleanup_old_data)
    schedule.every().day.at("01:30").do(maintenance.optimize_database)
    
    # Weekly reporting and backup
    schedule.every().sunday.at("02:00").do(maintenance.backup_model_data)
    schedule.every().sunday.at("03:00").do(maintenance.generate_performance_report)
    
    # Check for drift daily
    schedule.every().day.at("04:00").do(maintenance.check_model_drift)
    
    while True:
        schedule.run_pending()
        time.sleep(3600)

if __name__ == "__main__":
    schedule_maintenance()