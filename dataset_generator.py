import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class ScamDatasetGenerator:
    def __init__(self):
        self.scam_keywords = [
            'guaranteed returns', 'risk-free', 'get rich quick', 'easy money',
            'double your money', 'insider trading', 'pump and dump', 'ponzi',
            'pyramid scheme', 'binary options', 'forex robot', 'cryptocurrency scam',
            'investment opportunity', 'limited time offer', 'exclusive deal',
            'no experience needed', 'work from home', 'passive income',
            'mlm', 'network marketing', 'affiliate marketing'
        ]
        
        self.legitimate_topics = [
            'fundamental analysis', 'technical analysis', 'portfolio diversification',
            'risk management', 'dividend investing', 'value investing',
            'growth investing', 'etf investing', 'mutual funds', 'bonds',
            'market research', 'financial planning', 'retirement planning',
            'tax strategies', 'asset allocation', 'dollar cost averaging'
        ]
        
        self.question_templates = {
            'scam': [
                "How can I make {} in {} days with this {} strategy?",
                "Is this {} investment that promises {}% returns legitimate?",
                "Someone offered me a {} opportunity, should I invest?",
                "Can I really earn {} per day with this {} system?",
                "Is {} trading bot that guarantees {}% profit real?"
            ],
            'legitimate': [
                "What are the best practices for {}?",
                "How do I start learning about {}?",
                "What books do you recommend for {}?",
                "Can someone explain {} in simple terms?",
                "What are the risks involved in {}?"
            ]
        }
    
    def generate_scam_question(self):
        template = random.choice(self.question_templates['scam'])
        keyword = random.choice(self.scam_keywords)
        amount = random.choice(['$1000', '$5000', '$10000', 'quick money'])
        timeframe = random.choice(['7', '30', '90'])
        percentage = random.choice(['20', '50', '100', '200'])
        
        question = template.format(amount, timeframe, keyword, amount, keyword, 
                                 keyword, percentage, keyword, percentage)
        return question
    
    def generate_legitimate_question(self):
        template = random.choice(self.question_templates['legitimate'])
        topic = random.choice(self.legitimate_topics)
        return template.format(topic)
    
    def generate_dataset(self, size=10000):
        data = []
        
        # Generate 70% legitimate, 30% scam questions
        scam_count = int(size * 0.3)
        legit_count = size - scam_count
        
        # Generate scam questions
        for i in range(scam_count):
            question = self.generate_scam_question()
            # Add noise to scam scores (60-95 range)
            scam_score = np.random.normal(80, 10)
            scam_score = max(60, min(95, scam_score))
            
            data.append({
                'question': question,
                'scam_score': scam_score,
                'is_scam': 1,
                'timestamp': datetime.now() - timedelta(days=random.randint(0, 365)),
                'user_id': f'user_{random.randint(1000, 9999)}',
                'upvotes': random.randint(0, 5),
                'category': 'investment'
            })
        
        # Generate legitimate questions
        for i in range(legit_count):
            question = self.generate_legitimate_question()
            # Add noise to legitimate scores (0-40 range)
            scam_score = np.random.normal(15, 8)
            scam_score = max(0, min(40, scam_score))
            
            data.append({
                'question': question,
                'scam_score': scam_score,
                'is_scam': 0,
                'timestamp': datetime.now() - timedelta(days=random.randint(0, 365)),
                'user_id': f'user_{random.randint(1000, 9999)}',
                'upvotes': random.randint(0, 20),
                'category': random.choice(['stocks', 'bonds', 'etfs', 'general'])
            })
        
        return pd.DataFrame(data)

# Generate and save dataset
generator = ScamDatasetGenerator()
df = generator.generate_dataset(15000)
df.to_csv('scam_detection_dataset.csv', index=False)
print(f"Generated dataset with {len(df)} samples")
print(df['is_scam'].value_counts())