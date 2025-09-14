# monitoring.py
import logging
import time
from datetime import datetime, timedelta
from flask import Flask
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil
import joblib

# Metrics
REQUEST_COUNT = Counter('scam_detector_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('scam_detector_request_duration_seconds', 'Request latency')
SCAM_SCORES = Histogram('scam_detector_scores', 'Distribution of scam scores', buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
MODEL_PREDICTIONS = Counter('scam_detector_predictions_total', 'Total predictions', ['risk_level'])
SYSTEM_CPU = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
SYSTEM_MEMORY = Gauge('system_memory_usage_bytes', 'Memory usage in bytes')

class MonitoringService:
    def __init__(self, app):
        self.app = app
        self.setup_logging()
        self.setup_metrics()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scam_detector.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_metrics(self):
        @self.app.before_request
        def before_request():
            REQUEST_COUNT.labels(
                method=request.method, 
                endpoint=request.endpoint or 'unknown'
            ).inc()
        
        @self.app.after_request
        def after_request(response):
            # Update system metrics
            SYSTEM_CPU.set(psutil.cpu_percent())
            SYSTEM_MEMORY.set(psutil.virtual_memory().used)
            return response
    
    def log_prediction(self, content, score, risk_level, user_id=None):
        """Log prediction for analysis"""
        self.logger.info(f"Prediction - Score: {score}, Risk: {risk_level}, User: {user_id}")
        SCAM_SCORES.observe(score)
        MODEL_PREDICTIONS.labels(risk_level=risk_level).inc()
    
    def check_model_performance(self):
        """Monitor model performance and alert if degraded"""
        # Check prediction distribution
        # Alert if too many high-risk predictions (possible model drift)
        pass
    
    def health_check(self):
        """Comprehensive health check"""
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'healthy',
            'checks': {}
        }
        
        # Check model availability
        try:
            joblib.load('scam_detector_model.pkl')
            health_status['checks']['model'] = 'ok'
        except Exception as e:
            health_status['checks']['model'] = f'error: {str(e)}'
            health_status['status'] = 'unhealthy'
        
        # Check database connection
        try:
            from app import db
            db.session.execute('SELECT 1')
            health_status['checks']['database'] = 'ok'
        except Exception as e:
            health_status['checks']['database'] = f'error: {str(e)}'
            health_status['status'] = 'unhealthy'
        
        # Check system resources
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        if cpu_usage > 80:
            health_status['checks']['cpu'] = f'high: {cpu_usage}%'
            health_status['status'] = 'degraded'
        else:
            health_status['checks']['cpu'] = f'ok: {cpu_usage}%'
        
        if memory_usage > 80:
            health_status['checks']['memory'] = f'high: {memory_usage}%'
            health_status['status'] = 'degraded'
        else:
            health_status['checks']['memory'] = f'ok: {memory_usage}%'
        
        return health_status

# Prometheus metrics endpoint
@app.route('/metrics')
def metrics():
    return generate_latest()

@app.route('/health')
def health():
    monitoring = MonitoringService(app)
    return jsonify(monitoring.health_check())