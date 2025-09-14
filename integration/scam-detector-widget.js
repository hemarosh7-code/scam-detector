// scam-detector-widget.js
class ScamDetectorWidget {
    constructor(options = {}) {
        this.apiUrl = options.apiUrl || 'https://api.scamdetector.com';
        this.apiKey = options.apiKey;
        this.threshold = options.threshold || 70;
        this.container = options.container || document.body;
        this.autoCheck = options.autoCheck || false;
        
        this.init();
    }
    
    init() {
        this.createWidget();
        if (this.autoCheck) {
            this.observeContent();
        }
    }
    
    createWidget() {
        // Create floating widget
        this.widget = document.createElement('div');
        this.widget.id = 'scam-detector-widget';
        this.widget.innerHTML = `
            <div style="position: fixed; bottom: 20px; right: 20px; z-index: 10000; 
                        background: white; border: 2px solid #007cba; border-radius: 8px; 
                        padding: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); max-width: 300px;">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="font-weight: bold; color: #007cba;">🛡️ Scam Detector</span>
                    <button id="minimize-widget" style="margin-left: auto; background: none; border: none; font-size: 18px;">−</button>
                </div>
                <div id="widget-content">
                    <textarea id="content-input" placeholder="Paste content to check for scams..." 
                             style="width: 100%; height: 80px; margin-bottom: 10px; padding: 8px; border: 1px solid #ddd; border-radius: 4px;"></textarea>
                    <button id="analyze-btn" style="background: #007cba; color: white; border: none; 
                                                   padding: 8px 16px; border-radius: 4px; cursor: pointer; width: 100%;">
                        Analyze Content
                    </button>
                    <div id="results" style="margin-top: 10px; display: none;"></div>
                </div>
            </div>
        `;
        
        document.body.appendChild(this.widget);
        this.attachEvents();
    }
    
    attachEvents() {
        document.getElementById('minimize-widget').addEventListener('click', () => {
            const content = document.getElementById('widget-content');
            content.style.display = content.style.display === 'none' ? 'block' : 'none';
        });
        
        document.getElementById('analyze-btn').addEventListener('click', () => {
            const content = document.getElementById('content-input').value;
            if (content.trim()) {
                this.analyzeContent(content);
            }
        });
    }
    
    async analyzeContent(content) {
        const resultsDiv = document.getElementById('results');
        const analyzeBtn = document.getElementById('analyze-btn');
        
        analyzeBtn.textContent = 'Analyzing...';
        analyzeBtn.disabled = true;
        
        try {
            const response = await fetch(`${this.apiUrl}/api/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.apiKey
                },
                body: JSON.stringify({
                    question: content,
                    category: 'general'
                })
            });
            
            const data = await response.json();
            this.displayResults(data, resultsDiv);
            
        } catch (error) {
            resultsDiv.innerHTML = '<div style="color: red;">Error analyzing content</div>';
        } finally {
            analyzeBtn.textContent = 'Analyze Content';
            analyzeBtn.disabled = false;
            resultsDiv.style.display = 'block';
        }
    }
    
    displayResults(data, container) {
        const riskColor = data.scam_score >= 70 ? '#dc3545' : 
                         data.scam_score >= 40 ? '#ffc107' : '#28a745';
        
        container.innerHTML = `
            <div style="text-align: center; padding: 10px; background: ${riskColor}20; border-radius: 4px;">
                <div style="font-size: 24px; font-weight: bold; color: ${riskColor};">
                    ${data.scam_score}
                </div>
                <div style="font-size: 12px; color: #666;">
                    Risk Level: ${data.risk_level}
                </div>
                <div style="font-size: 12px; margin-top: 5px;">
                    Action: ${data.action}
                </div>
            </div>
        `;
    }
    
    observeContent() {
        // Auto-check forms and textareas
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            form.addEventListener('submit', (e) => {
                const textInputs = form.querySelectorAll('textarea, input[type="text"]');
                textInputs.forEach(async (input) => {
                    if (input.value.length > 50) {
                        const result = await this.quickAnalyze(input.value);
                        if (result.scam_score >= this.threshold) {
                            if (!confirm(`Warning: This content has a high scam score (${result.scam_score}). Continue anyway?`)) {
                                e.preventDefault();
                            }
                        }
                    }
                });
            });
        });
    }
    
    async quickAnalyze(content) {
        try {
            const response = await fetch(`${this.apiUrl}/api/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.apiKey
                },
                body: JSON.stringify({
                    question: content,
                    category: 'general'
                })
            });
            return await response.json();
        } catch (error) {
            return { scam_score: 0 };
        }
    }
}

// Usage
// new ScamDetectorWidget({
//     apiUrl: 'https://your-api.com',
//     apiKey: 'your-api-key',
//     threshold: 70,
//     autoCheck: true
// });