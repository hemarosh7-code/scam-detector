# security.py
from functools import wraps
from flask import request, jsonify, current_app
import hashlib
import hmac
import time
import redis
import ipaddress
from datetime import datetime, timedelta

class SecurityMiddleware:
    def __init__(self, app=None, redis_client=None):
        self.app = app
        self.redis_client = redis_client or redis.Redis()
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        app.before_request(self.before_request)
        app.after_request(self.after_request)
    
    def before_request(self):
        # Rate limiting
        client_ip = self.get_client_ip()
        if not self.check_rate_limit(client_ip):
            return jsonify({'error': 'Rate limit exceeded'}), 429
        
        # IP whitelist/blacklist
        if not self.check_ip_allowed(client_ip):
            return jsonify({'error': 'Access denied'}), 403
        
        # Validate request size
        if request.content_length and request.content_length > 1024 * 1024:  # 1MB limit
            return jsonify({'error': 'Request too large'}), 413
    
    def after_request(self, response):
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        return response
    
    def get_client_ip(self):
        """Get real client IP address"""
        if request.headers.get('X-Forwarded-For'):
            return request.headers.get('X-Forwarded-For').split(',')[0].strip()
        elif request.headers.get('X-Real-IP'):
            return request.headers.get('X-Real-IP')
        return request.remote_addr
    
    def check_rate_limit(self, client_ip, requests_per_minute=60):
        """Check if client is within rate limits"""
        key = f"rate_limit:{client_ip}"
        current = self.redis_client.get(key)
        
        if current is None:
            self.redis_client.setex(key, 60, 1)
            return True
        elif int(current) < requests_per_minute:
            self.redis_client.incr(key)
            return True
        else:
            return False
    
    def check_ip_allowed(self, client_ip):
        """Check if IP is allowed (whitelist/blacklist)"""
        try:
            ip = ipaddress.ip_address(client_ip)
            
            # Block known malicious IP ranges
            blocked_ranges = [
                '10.0.0.0/8',      # Private networks (if needed)
                '192.168.0.0/16',  # Private networks (if needed)
                '172.16.0.0/12',   # Private networks (if needed)
            ]
            
            # Check against blacklist in Redis
            if self.redis_client.sismember('blacklisted_ips', client_ip):
                return False
            
            return True
            
        except ValueError:
            return False
    
    def validate_api_key(self, api_key):
        """Validate API key for authenticated endpoints"""
        if not api_key:
            return False
        
        # Check against valid API keys in database/Redis
        return self.redis_client.sismember('valid_api_keys', api_key)

def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        # Validate API key (implement your validation logic)
        security = SecurityMiddleware()
        if not security.validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

# Input validation
def validate_question_input(data):
    """Validate question input data"""
    if not isinstance(data, dict):
        return False, "Invalid data format"
    
    question = data.get('question', '')
    if not question or not isinstance(question, str):
        return False, "Question is required and must be a string"
    
    if len(question) > 5000:  # 5000 character limit
        return False, "Question too long (max 5000 characters)"
    
    if len(question) < 10:
        return False, "Question too short (min 10 characters)"
    
    # Check for potential injection attempts
    suspicious_patterns = ['<script', 'javascript:', 'onload=', 'onerror=']
    question_lower = question.lower()
    for pattern in suspicious_patterns:
        if pattern in question_lower:
            return False, "Suspicious content detected"
    
    return True, "Valid"