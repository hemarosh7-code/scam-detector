# fastapi_app.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
import joblib
import pandas as pd
import os
from typing import List, Optional

app = FastAPI(title="Scam Detector API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./scam_detector.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class QuestionDB(Base):
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    scam_score = Column(Integer)
    user_id = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)
    category = Column(String(50), default="general")
    flagged = Column(Boolean, default=False)
    upvotes = Column(Integer, default=0)

Base.metadata.create_all(bind=engine)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    user_id: Optional[str] = "anonymous"
    category: Optional[str] = "general"

class QuestionResponse(BaseModel):
    scam_score: int
    risk_level: str
    action: str
    question_id: int
    timestamp: str
    recommendations: List[str]

class TrendingResponse(BaseModel):
    id: int
    content: str
    scam_score: int
    timestamp: str
    category: str

# Load ML model
model_data = joblib.load('scam_detector_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
feature_columns = model_data['feature_columns']

from feature_engineering import FeatureEngineer
fe = FeatureEngineer()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/analyze", response_model=QuestionResponse)
async def analyze_question(request: QuestionRequest, db: Session = Depends(get_db)):
    """Analyze a question for scam indicators"""
    try:
        # Create temporary dataframe for feature extraction
        temp_df = pd.DataFrame({'question': [request.question]})
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
        question_db = QuestionDB(
            content=request.question,
            scam_score=scam_score,
            user_id=request.user_id,
            category=request.category,
            flagged=(scam_score >= 70)
        )
        db.add(question_db)
        db.commit()
        db.refresh(question_db)
        
        return QuestionResponse(
            scam_score=scam_score,
            risk_level=risk_level,
            action=action,
            question_id=question_db.id,
            timestamp=question_db.timestamp.isoformat(),
            recommendations=get_recommendations(scam_score)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trending")
async def get_trending_scams(days: int = 7, db: Session = Depends(get_db)):
    """Get trending scam patterns"""
    try:
        since = datetime.utcnow() - timedelta(days=days)
        
        trending = db.query(QuestionDB)\
            .filter(QuestionDB.timestamp >= since)\
            .filter(QuestionDB.scam_score >= 70)\
            .order_by(QuestionDB.timestamp.desc())\
            .limit(50).all()
        
        trends = []
        for q in trending:
            trends.append(TrendingResponse(
                id=q.id,
                content=q.content[:200] + '...' if len(q.content) > 200 else q.content,
                scam_score=q.scam_score,
                timestamp=q.timestamp.isoformat(),
                category=q.category
            ))
        
        return {
            'trending_scams': trends,
            'total_count': len(trends),
            'timeframe_days': days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}