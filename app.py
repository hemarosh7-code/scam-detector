from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import joblib
import pandas as pd
import os
from feature_engineering import FeatureEngineer
import json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///scam_detector.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    scam_score = db.Column(db.Integer)
    user_id = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    category = db.Column(db.String(50), default='general')
    flagged = db.Column(db.Boolean, default=False)
    upvotes = db.Column(db.Integer, default=0)

class TrendingScam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pattern = db.Column(db.String(200))
    count = db.Column(db.Integer)
    avg_score = db.Column(db.Float)
    first_seen = db.Column(db.DateTime)
    last_seen = db.Column(db.DateTime)

# Initialize database
with app.app_context():
    db.create_all()

# Load ML model
model_data = joblib.load('scam_detector_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
feature_columns = model_data['feature_columns']
fe = FeatureEngineer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_question():
    """Analyze a question for scam indicators"""
    try:
        data = request.json
        question_text = data.get('question', '')
        user_id = data.get('user_id', 'anonymous')
        category = data.get('category', 'general')
        
        if not question_text:
            return jsonify({'error': 'Question text is required'}), 400
        
        # Create temporary dataframe for feature extraction
        temp_df = pd.DataFrame({'question': [question_text]})
        feature_df, _ = fe.create_features(temp_df)
        
        # Prepare features
        X = feature_df[feature_columns].fillna(0)
        X_scaled = scaler.transform(X)
        
        # Predict scam probability
        scam_probability = model.predict_proba(X_scaled)[0, 1]
        scam_score = int(scam_probability * 100)
        
        # Determine risk level
        if scam_score >= 70:
            risk_level = 'HIGH'
            action = 'BLOCK'
        elif scam_score >= 40:
            risk_level = 'MEDIUM'
            action = 'REVIEW'
        else:
            risk_level = 'LOW'
            action = 'ALLOW'
        
        # Save to database
        question = Question(
            content=question_text,
            scam_score=scam_score,
            user_id=user_id,
            category=category,
            flagged=(scam_score >= 70)
        )
        db.session.add(question)
        db.session.commit()
        
        # Update trending patterns
        update_trending_patterns(question_text, scam_score)
        
        response = {
            'scam_score': scam_score,
            'risk_level': risk_level,
            'action': action,
            'question_id': question.id,
            'timestamp': question.timestamp.isoformat(),
            'recommendations': get_recommendations(scam_score)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trending', methods=['GET'])
def get_trending_scams():
    """Get trending scam patterns"""
    try:
        days = request.args.get('days', 7, type=int)
        since = datetime.utcnow() - timedelta(days=days)
        
        # Get recent high-score questions
        trending = db.session.query(Question)\
            .filter(Question.timestamp >= since)\
            .filter(Question.scam_score >= 70)\
            .order_by(Question.timestamp.desc())\
            .limit(50).all()
        
        trends = []
        for q in trending:
            trends.append({
                'id': q.id,
                'content': q.content[:200] + '...' if len(q.content) > 200 else q.content,
                'scam_score': q.scam_score,
                'timestamp': q.timestamp.isoformat(),
                'category': q.category
            })
        
        return jsonify({
            'trending_scams': trends,
            'total_count': len(trends),
            'timeframe_days': days
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get platform statistics"""
    try:
        total_questions = Question.query.count()
        flagged_questions = Question.query.filter(Question.flagged == True).count()
        
        # Recent activity (last 24 hours)
        since_24h = datetime.utcnow() - timedelta(hours=24)
        recent_questions = Question.query.filter(Question.timestamp >= since_24h).count()
        recent_flagged = Question.query.filter(
            Question.timestamp >= since_24h,
            Question.flagged == True
        ).count()
        
        # Average scores by category
        category_stats = db.session.query(
            Question.category,
            db.func.avg(Question.scam_score).label('avg_score'),
            db.func.count(Question.id).label('count')
        ).group_by(Question.category).all()
        
        categories = {}
        for cat, avg_score, count in category_stats:
            categories[cat] = {
                'avg_scam_score': round(avg_score, 2) if avg_score else 0,
                'total_questions': count
            }
        
        return jsonify({
            'total_questions': total_questions,
            'flagged_questions': flagged_questions,
            'flagged_percentage': round((flagged_questions / total_questions) * 100, 2) if total_questions > 0 else 0,
            'recent_24h': {
                'questions': recent_questions,
                'flagged': recent_flagged
            },
            'categories': categories
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def update_trending_patterns(question_text, scam_score):
    """Update trending scam patterns"""
    if scam_score >= 70:  # Only track high-risk patterns
        # Extract key phrases (simple implementation)
        words = question_text.lower().split()
        patterns = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
        for pattern in patterns:
            trend = TrendingScam.query.filter_by(pattern=pattern).first()
            if trend:
                trend.count += 1
                trend.last_seen = datetime.utcnow()
                trend.avg_score = (trend.avg_score + scam_score) / 2
            else:
                trend = TrendingScam(
                    pattern=pattern,
                    count=1,
                    avg_score=scam_score,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow()
                )
                db.session.add(trend)
        
        db.session.commit()

def get_recommendations(scam_score):
    """Get recommendations based on scam score"""
    if scam_score >= 70:
        return [
            "This content shows high scam indicators - consider blocking",
            "Review user history for pattern of suspicious activity",
            "Alert moderators for immediate review"
        ]
    elif scam_score >= 40:
        return [
            "Content requires manual review",
            "Consider additional verification steps",
            "Monitor user's future posts closely"
        ]
    else:
        return [
            "Content appears legitimate",
            "Continue normal processing",
            "Regular monitoring sufficient"
        ]

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback on predictions"""
    try:
        data = request.json
        question_id = data.get('question_id')
        is_correct = data.get('is_correct', True)
        user_feedback = data.get('feedback', '')
        
        question = Question.query.get(question_id)
        if not question:
            return jsonify({'error': 'Question not found'}), 404
        
        # Store feedback for model retraining (implement as needed)
        # This could be stored in a separate feedback table
        
        return jsonify({'message': 'Feedback received successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)