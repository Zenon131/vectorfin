"""
VectorFin Web Interface

A simple Flask web application that provides a user interface for interacting
with a trained VectorFin model and viewing LLM-enhanced interpretations.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify

# Add parent directory to path so we can import the vectorfin package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import VectorFin components
from vectorfin.src.models.vectorfin import VectorFinSystem
from vectorfin.src.data.data_loader import MarketData, FinancialTextData

# Import the LLM interpreter
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_enhanced_interpreter import LLMFinancialInterpreter

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Global variables
MODEL = None
INTERPRETER = None


def initialize_model(models_dir="./trained_models"):
    """Initialize the VectorFin model."""
    global MODEL
    
    if MODEL is None:
        print(f"Loading model from {models_dir}...")
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create system and load models
        MODEL = VectorFinSystem.load_models(
            VectorFinSystem,  # Pass the class as first parameter
            directory=models_dir,
            vector_dim=128,
            sentiment_dim=16,
            fusion_dim=128,
            device=device
        )
        
        print("Model loaded successfully!")
    
    return MODEL


def initialize_interpreter(models_dir="./trained_models", provider="openai"):
    """Initialize the LLM interpreter."""
    global INTERPRETER
    
    if INTERPRETER is None:
        # Get API key from environment if available
        api_key = os.environ.get(f"{provider.upper()}_API_KEY")
        
        INTERPRETER = LLMFinancialInterpreter(
            model_path=models_dir,
            llm_provider=provider,
            api_key=api_key
        )
        
        print(f"Interpreter initialized with {provider} provider")
    
    return INTERPRETER


def fetch_market_data(tickers, days=30):
    """Fetch recent market data for the given tickers."""
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Fetch data
    market_data = MarketData.fetch_market_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )
    
    return market_data


def process_news_input(news_text):
    """Process free-form news text input into a DataFrame."""
    # Split news text into lines
    lines = [line.strip() for line in news_text.strip().split('\n') if line.strip()]
    
    # Parse into news items
    news_items = []
    
    for line in lines:
        # Try to extract date if it exists
        if ':' in line and line.split(':', 1)[0].strip().replace('/', '-').replace('.', '-').count('-') == 2:
            # Date is likely at the beginning
            date_text, headline = line.split(':', 1)
            try:
                # Try different date formats
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%m-%d-%Y', '%m/%d/%Y']:
                    try:
                        date = datetime.strptime(date_text.strip(), fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # If no format matched, use today's date
                    date = datetime.now()
            except:
                date = datetime.now()
        else:
            # No date found, use today's date
            date = datetime.now()
            headline = line
        
        news_items.append({
            'date': date,
            'headline': headline.strip(),
            'source': 'user_input'
        })
    
    # Create DataFrame
    if news_items:
        news_data = pd.DataFrame(news_items)
        news_data['date'] = pd.to_datetime(news_data['date'])
        return news_data
    else:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['date', 'headline', 'source'])


def make_prediction(model, news_data, market_data, prediction_horizon=5):
    """Make a prediction using the model."""
    # Get the most recent date in the market data
    latest_date = max(df['date'].max() for df in market_data.values())
    
    # Get texts from the news
    texts = news_data['headline'].tolist()
    
    # Process texts through the text vectorizer
    text_vectors = model.process_text(texts)
    
    # Prepare the market data (most recent data point)
    latest_market_data = {}
    for ticker, df in market_data.items():
        latest_market_data[ticker] = df.iloc[-1:].copy()
    
    # Align the latest market data
    aligned_data = MarketData.align_market_data(latest_market_data)
    
    # Process market data through the numerical vectorizer
    market_vectors, _ = model.process_market_data(aligned_data)
    
    # Create unified representation
    unified_vector = model.create_unified_representation(text_vectors, market_vectors)
    
    # Make prediction
    predictions = model.predict(unified_vector, prediction_types=['direction', 'magnitude', 'volatility'])
    
    # Format results
    results = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'prediction_horizon': prediction_horizon,
        'predictions': {
            'direction': float(predictions['direction'].item()),  # Probability of upward movement
            'magnitude': float(predictions['magnitude'].item()),  # Expected percentage change
            'volatility': float(predictions['volatility'].item())  # Expected volatility
        },
        'confidence_score': 0.75  # Placeholder confidence score
    }
    
    return results


def format_market_context(market_data):
    """Format market data for the LLM interpreter."""
    context_lines = []
    
    for ticker, data in market_data.items():
        df = data.copy()
        if len(df) > 5:
            # Calculate some basic stats
            last_price = df['close'].iloc[-1]
            change_5d = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100
            avg_volume = df['volume'].iloc[-5:].mean()
            prev_avg_volume = df['volume'].iloc[-30:-5].mean() if len(df) > 30 else df['volume'].mean()
            vol_change = (avg_volume / prev_avg_volume - 1) * 100
            
            context_lines.append(f"- {ticker}: Last closing price ${last_price:.2f}, {'up' if change_5d > 0 else 'down'} {abs(change_5d):.1f}% over the past 5 days, with average volume {'+' if vol_change > 0 else ''}{vol_change:.0f}% compared to the previous period.")
    
    # Add broader market context 
    context_lines.append("- Market Context: This analysis focuses solely on the specified tickers and does not include broader market indicators.")
    
    return "\n".join(context_lines)


def format_news_context(news_data):
    """Format news data for the LLM interpreter."""
    # Sort by date descending
    recent_news = news_data.sort_values('date', ascending=False)
    
    context_lines = ["Recent news highlights:"]
    
    # Add up to 5 most recent news items
    for i, (_, news) in enumerate(recent_news.head(5).iterrows()):
        date_str = news['date'].strftime('%b %d, %Y')
        context_lines.append(f"- {date_str}: \"{news['headline']}\" (Source: {news.get('source', 'unknown')})")
    
    return "\n".join(context_lines)


@app.route('/')
def home():
    """Render the home page."""
    # Make sure model is initialized
    initialize_model()
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get data from the request
        data = request.get_json()
        tickers = data['tickers'].split(',')
        news_text = data['news']
        prediction_horizon = int(data['predictionHorizon'])
        use_llm = data.get('useLlm', False)
        llm_question = data.get('llmQuestion', '')
        
        # Clean tickers
        tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
        
        if not tickers:
            return jsonify({'error': 'No valid tickers provided.'}), 400
        
        # Initialize model if needed
        model = initialize_model()
        
        # Fetch market data
        market_data = fetch_market_data(tickers, days=30)
        
        # Process news input
        news_data = process_news_input(news_text)
        
        # Make prediction
        prediction_results = make_prediction(model, news_data, market_data, prediction_horizon)
        
        response = {
            'prediction': prediction_results,
            'llm_interpretation': None
        }
        
        # Get LLM interpretation if requested
        if use_llm:
            # Initialize interpreter if needed
            interpreter = initialize_interpreter()
            
            # Format context for LLM
            market_context = format_market_context(market_data)
            news_context = format_news_context(news_data)
            
            # Get interpretation
            interpretation = interpreter.interpret_prediction(
                prediction_results,
                market_context,
                news_context,
                user_question=llm_question if llm_question else None
            )
            
            response['llm_interpretation'] = interpretation
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VectorFin - Financial Prediction Interface</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 20px; }
        .prediction-card { margin-top: 20px; }
        .news-textarea { height: 200px; }
        .interpretation { white-space: pre-wrap; }
        .loading { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">VectorFin Financial Prediction Interface</h1>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Input Parameters</div>
                    <div class="card-body">
                        <form id="predictionForm">
                            <div class="mb-3">
                                <label for="tickers" class="form-label">Stock Tickers (comma-separated)</label>
                                <input type="text" class="form-control" id="tickers" placeholder="AAPL,MSFT,GOOGL" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="news" class="form-label">Recent News (one per line, include dates if available)</label>
                                <textarea class="form-control news-textarea" id="news" placeholder="2023-05-15: Apple announces new AI features&#10;2023-05-14: Microsoft expands cloud services"></textarea>
                            </div>
                            
                            <div class="mb-3">
                                <label for="predictionHorizon" class="form-label">Prediction Horizon (days)</label>
                                <input type="number" class="form-control" id="predictionHorizon" value="5" min="1" max="30">
                            </div>
                            
                            <div class="mb-3 form-check">
                                <input type="checkbox" class="form-check-input" id="useLlm">
                                <label class="form-check-label" for="useLlm">Use LLM for interpretation</label>
                            </div>
                            
                            <div class="mb-3 llm-question" style="display:none;">
                                <label for="llmQuestion" class="form-label">Question for LLM (optional)</label>
                                <input type="text" class="form-control" id="llmQuestion" placeholder="How should I interpret this for a short-term strategy?">
                            </div>
                            
                            <button type="submit" class="btn btn-primary">Generate Prediction</button>
                            <div class="loading mt-3">
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                                <span class="ms-2">Processing...</span>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card prediction-card">
                    <div class="card-header">Prediction Results</div>
                    <div class="card-body">
                        <div id="results">
                            <p class="text-muted">Submit the form to generate a prediction.</p>
                        </div>
                    </div>
                </div>
                
                <div class="card prediction-card llm-card" style="display:none;">
                    <div class="card-header">LLM Interpretation</div>
                    <div class="card-body">
                        <div id="llmResults" class="interpretation">
                            <p class="text-muted">Enable LLM interpretation to see analysis.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('useLlm').addEventListener('change', function() {
            const llmQuestion = document.querySelector('.llm-question');
            const llmCard = document.querySelector('.llm-card');
            
            if (this.checked) {
                llmQuestion.style.display = 'block';
                llmCard.style.display = 'block';
            } else {
                llmQuestion.style.display = 'none';
                llmCard.style.display = 'none';
            }
        });
        
        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading indicator
            document.querySelector('.loading').style.display = 'block';
            document.getElementById('results').innerHTML = '<p>Processing prediction...</p>';
            document.getElementById('llmResults').innerHTML = '<p>Waiting for prediction results...</p>';
            
            const formData = {
                tickers: document.getElementById('tickers').value,
                news: document.getElementById('news').value,
                predictionHorizon: document.getElementById('predictionHorizon').value,
                useLlm: document.getElementById('useLlm').checked,
                llmQuestion: document.getElementById('llmQuestion').value
            };
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                document.querySelector('.loading').style.display = 'none';
                
                if (data.error) {
                    document.getElementById('results').innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                    return;
                }
                
                // Format prediction results
                const pred = data.prediction.predictions;
                const directionText = pred.direction > 0.5 ? 'Upward' : 'Downward';
                const directionConfidence = Math.abs((pred.direction - 0.5) * 2) * 100;
                
                let resultsHtml = `
                    <h5>Prediction for ${data.prediction.date} (${data.prediction.prediction_horizon} days ahead)</h5>
                    <div class="table-responsive">
                        <table class="table table-bordered">
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                                <th>Interpretation</th>
                            </tr>
                            <tr>
                                <td>Direction</td>
                                <td>${(pred.direction * 100).toFixed(1)}%</td>
                                <td>${directionText} movement (${directionConfidence.toFixed(1)}% confidence)</td>
                            </tr>
                            <tr>
                                <td>Magnitude</td>
                                <td>${(pred.magnitude * 100).toFixed(2)}%</td>
                                <td>Expected ${pred.magnitude >= 0 ? 'gain' : 'loss'} of ${Math.abs(pred.magnitude * 100).toFixed(2)}%</td>
                            </tr>
                            <tr>
                                <td>Volatility</td>
                                <td>${(pred.volatility * 100).toFixed(2)}%</td>
                                <td>${pred.volatility < 0.01 ? 'Low' : pred.volatility < 0.03 ? 'Moderate' : 'High'} expected volatility</td>
                            </tr>
                        </table>
                    </div>
                `;
                
                document.getElementById('results').innerHTML = resultsHtml;
                
                // Display LLM interpretation if available
                if (data.llm_interpretation) {
                    document.getElementById('llmResults').innerHTML = data.llm_interpretation
                        .replace(/\\n/g, '<br>')
                        .replace(/\n/g, '<br>');
                }
            })
            .catch(error => {
                document.querySelector('.loading').style.display = 'none';
                document.getElementById('results').innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            });
        });
    </script>
</body>
</html>
        """)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5002)
