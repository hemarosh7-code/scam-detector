# data_collector.py
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

class DataCollector:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def collect_reddit_data(self, subreddits=['investing', 'stocks', 'SecurityAnalysis']):
        """Collect data from Reddit using pushshift API"""
        data = []
        
        for subreddit in subreddits:
            url = f"https://api.pushshift.io/reddit/search/submission"
            params = {
                'subreddit': subreddit,
                'size': 1000,
                'fields': 'title,selftext,score,created_utc'
            }
            
            try:
                response = requests.get(url, params=params)
                posts = response.json()['data']
                
                for post in posts:
                    if post.get('selftext') and len(post['selftext']) > 50:
                        data.append({
                            'question': post['title'] + ' ' + post['selftext'],
                            'score': post.get('score', 0),
                            'timestamp': post['created_utc'],
                            'source': f'reddit_{subreddit}'
                        })
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error collecting from {subreddit}: {e}")
        
        return pd.DataFrame(data)
    
    def manual_labeling_interface(self, df, sample_size=500):
        """Simple interface for manual labeling"""
        sample_df = df.sample(n=min(sample_size, len(df))).reset_index()
        labeled_data = []
        
        print("Manual labeling interface - Enter 1 for scam, 0 for legitimate, s to skip:")
        
        for idx, row in sample_df.iterrows():
            print(f"\n{idx+1}/{len(sample_df)}")
            print("Question:", row['question'][:200] + "...")
            
            while True:
                label = input("Label (0/1/s): ").strip()
                if label in ['0', '1', 's']:
                    break
                print("Please enter 0, 1, or s")
            
            if label == 's':
                continue
            
            labeled_data.append({
                'question': row['question'],
                'is_scam': int(label),
                'source': 'manual_labeled'
            })
        
        return pd.DataFrame(labeled_data)

# Usage
collector = DataCollector()
reddit_data = collector.collect_reddit_data()
print(f"Collected {len(reddit_data)} posts from Reddit")