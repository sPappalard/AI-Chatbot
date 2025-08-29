import os
import json
import random
from datetime import datetime, timedelta
import time
from collections import defaultdict
from numpy.core.multiarray import scalar
import numpy as np

#Framework web to create API REST 
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

#Natural Language Processing
import nltk
#deep learning
import torch
import torch.nn as nn
#Pre-training models (DistilBERT)
from transformers import AutoTokenizer, AutoModel

import requests
#local database
import sqlite3
from contextlib import contextmanager

import uuid
from dotenv import load_dotenv


#load environment variables from .env
load_dotenv()

#Download NLTK data (tokenizer, dict)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

#create FLASK APP
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fallbackKEY')
#enable CORS for calls from different domains 
CORS(app)

#API Keys from .env
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '')

#Initialize tokenizer and model (DistilBERT)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = AutoModel.from_pretrained('distilbert-base-uncased')

#to manage usage limits external API to avoid exeeding them
class APIRateLimiter:
    def __init__(self):
        # counters for each API (reset every day)
        self.daily_counters = {
            'weather': {'count': 0, 'reset_date': datetime.now().date(), 'limit': 1000},
            'news': {'count': 0, 'reset_date': datetime.now().date(), 'limit': 100},
            'stocks': {'count': 0, 'reset_date': datetime.now().date(), 'limit': 25}
        }
        
        # Rate limiting per minute (Alpha Vantage: 5 calls/minute)
        self.minute_counters = defaultdict(list)
    
    #Check if we can make a request or not
    def can_make_request(self, api_type):
        """Controlla se possiamo fare una richiesta"""
        today = datetime.now().date()
        
        # counters if it is a new day
        if self.daily_counters[api_type]['reset_date'] != today:
            self.daily_counters[api_type]['count'] = 0
            self.daily_counters[api_type]['reset_date'] = today
        
        # check daily limit
        if self.daily_counters[api_type]['count'] >= self.daily_counters[api_type]['limit']:
            return False, f"Limite giornaliero raggiunto per {api_type} ({self.daily_counters[api_type]['limit']} chiamate)"
        
        # check minute limit (only for stocks/Alpha Vantage)
        if api_type == 'stocks':
            now = time.time()
            minute_ago = now - 60
            
            # Remove requests older than 1 minute
            self.minute_counters[api_type] = [t for t in self.minute_counters[api_type] if t > minute_ago]
            
            if len(self.minute_counters[api_type]) >= 5:
                return False, "Limite di 5 chiamate al minuto raggiunto per le azioni"
            
            # add current timestamp
            self.minute_counters[api_type].append(now)
        
        return True, "OK"
    
    #to increase counter after 1 success call
    def increment_counter(self, api_type):
        """Incrementa il contatore dopo una chiamata riuscita"""
        self.daily_counters[api_type]['count'] += 1
    
    #return API stats (used, limit, remaining)
    def get_stats(self):
        """Restituisce statistiche uso API"""
        today = datetime.now().date()
        stats = {}
        
        for api_type, data in self.daily_counters.items():
            if data['reset_date'] != today:
                stats[api_type] = {'used': 0, 'limit': data['limit'], 'remaining': data['limit']}
            else:
                remaining = data['limit'] - data['count']
                stats[api_type] = {'used': data['count'], 'limit': data['limit'], 'remaining': remaining}
        
        return stats

# Initialize rate limiter
rate_limiter = APIRateLimiter()

#---------------------------------------------
#Neural Netword Model for intent classification:
#-BASE: DistilBERT (pre-trained)(768 dimensions)
#-1. Layer: 768->256 + BatchNorm + Dropout + GELU 
#-2. Layer: 256->128 + BatchNorm + Dropout + GELU
#-3. Layer: 128->num_classed (final output)
#---------------------------------------------
class ImprovedChatbotModel(nn.Module):
    def __init__(self, bert_hidden_size, num_classes, hidden_size=256, dropout_rate=0.3):
        super().__init__()
        self.bert = bert_model
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)
        
        self.fc1 = nn.Linear(bert_hidden_size, hidden_size)  # 768  -> 256
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # 256 -> 128
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)  # 128 -> num_classes
        
        self.gelu = nn.GELU()
        
        # Fine-tuning Configuration: unfreeze only last 2 layer 
        #freeze
        for param in self.bert.parameters():
            param.requires_grad = False
        #unfreeze
        for param in self.bert.transformer.layer[-2:].parameters():
            param.requires_grad = True

    #Method called when you passes data to the model        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #CLS TOKEN
        pooled_output = bert_output.last_hidden_state[:, 0, :]  
        
        #classification layers 
        #1 layer
        x = self.dropout1(pooled_output)
        x = self.gelu(self.bn1(self.fc1(x)))
        #2 layer
        x = self.dropout2(x)
        x = self.gelu(self.bn2(self.fc2(x)))
        #3 layer
        x = self.dropout3(x)
        x = self.fc3(x)
        return x


#to manage conversation memory and create ToDo list table
class ConversationMemory:
    def __init__(self, db_path='conversations.db'):
        self.db_path = db_path
        self.init_db()

    #inizialize db
    def init_db(self):
        with self.get_db() as conn:
            #create conversation table 
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    timestamp DATETIME,
                    message TEXT,
                    response TEXT,
                    intent TEXT
                )
            ''')
            #create ToDo list table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS todos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    task TEXT,
                    completed BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    priority INTEGER DEFAULT 1
                )
            ''')
            conn.commit()

    @contextmanager
    #to open a connection to Database
    def get_db(self):
        #connect to SQlite database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            #return connection
            yield conn
        finally:
            #close connection
            conn.close()

    #to add new conversation to the DB
    def add_conversation(self, user_id, message, response, intent):
        #open a connection
        with self.get_db() as conn:
            #insert to DB
            conn.execute('''
                INSERT INTO conversations (id, user_id, timestamp, message, response, intent)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (str(uuid.uuid4()), user_id, datetime.now(), message, response, intent))
            #confirm the transation
            conn.commit()

    #to retrieve the recent context of a specific user
    def get_recent_context(self, user_id, limit=5):
        with self.get_db() as conn:
            #select user's last conversation (order by most recent, limited to 5)
            cursor = conn.execute('''
                SELECT message, response, intent FROM conversations
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
            #return all rows resulting from query 
            return cursor.fetchall()

#Chatbot main class  
class EnhancedChatbotAssistant:
    def __init__(self, intents_path):
        #path of intents file
        self.intents_path = intents_path
        #model
        self.model = None
        #list of intents
        self.intents = []
        #dict for complete data intents
        self.intents_data = {}
        #mapping for conversions
        self.label_to_idx = {}
        self.idx_to_label = {}
        #instance to memorize conversation
        self.memory = ConversationMemory()
        
        #load intents and model
        self.load_intents()
        self.load_model()

            #dictonary mapping company names to their stocl symbol
        self.available_stocks = {
            #tech
            'apple': 'AAPL', 'aapl': 'AAPL',
            'microsoft': 'MSFT', 'msft': 'MSFT',
            'google': 'GOOGL', 'googl': 'GOOGL', 'alphabet': 'GOOGL',
            'meta': 'META', 'facebook': 'META', 'meta platforms': 'META',
            'amazon': 'AMZN', 'amzn': 'AMZN',
            'netflix': 'NFLX', 'nflx': 'NFLX',
            'tesla': 'TSLA', 'tsla': 'TSLA',
            'nvidia': 'NVDA', 'nvda': 'NVDA',
            'amd': 'AMD', 'advanced micro devices': 'AMD',
            'intel': 'INTC', 'intc': 'INTC',
            
            #finance
            'jpmorgan': 'JPM', 'jp morgan': 'JPM', 'jpm': 'JPM',
            'bank of america': 'BAC', 'bac': 'BAC',
            'wells fargo': 'WFC', 'wfc': 'WFC',
            'goldman sachs': 'GS', 'gs': 'GS',
            'morgan stanley': 'MS', 'ms': 'MS',
            'visa': 'V', 'v': 'V',
            'mastercard': 'MA', 'ma': 'MA',
            'paypal': 'PYPL', 'pypl': 'PYPL',
            
            #health
            'johnson & johnson': 'JNJ', 'jnj': 'JNJ', 'johnson johnson': 'JNJ',
            'pfizer': 'PFE', 'pfe': 'PFE',
            'moderna': 'MRNA', 'mrna': 'MRNA',
            'abbott': 'ABT', 'abt': 'ABT',
            'merck': 'MRK', 'mrk': 'MRK',
            
            #energy
            'exxon': 'XOM', 'exxon mobil': 'XOM', 'xom': 'XOM',
            'chevron': 'CVX', 'cvx': 'CVX',
            'conocophillips': 'COP', 'cop': 'COP',
            
            #consumer goods
            'coca cola': 'KO', 'coca-cola': 'KO', 'ko': 'KO',
            'pepsi': 'PEP', 'pep': 'PEP', 'pepsico': 'PEP',
            'procter gamble': 'PG', 'pg': 'PG', 'procter & gamble': 'PG',
            'nike': 'NKE', 'nke': 'NKE',
            'adidas': 'ADDYY', 'addyy': 'ADDYY',
            
            #industrial
            'boeing': 'BA', 'ba': 'BA',
            'caterpillar': 'CAT', 'cat': 'CAT',
            '3m': 'MMM', 'mmm': 'MMM',
            'general electric': 'GE', 'ge': 'GE',
            'lockheed martin': 'LMT', 'lmt': 'LMT',
            
            # Retail e Services
            'walmart': 'WMT', 'wmt': 'WMT',
            'home depot': 'HD', 'hd': 'HD',
            'mcdonalds': 'MCD', 'mcd': 'MCD', "mcdonald's": 'MCD',
            'starbucks': 'SBUX', 'sbux': 'SBUX',
            'disney': 'DIS', 'dis': 'DIS', 'walt disney': 'DIS',
            
            # Telecom
            'verizon': 'VZ', 'vz': 'VZ',
            'at&t': 'T', 'att': 'T', 'at t': 'T',
            't-mobile': 'TMUS', 'tmus': 'TMUS', 'tmobile': 'TMUS',
            
            # other 
            'berkshire hathaway': 'BRK.B', 'berkshire': 'BRK.B', 'brk': 'BRK.B',
            'warren buffett': 'BRK.B',  
            'ibm': 'IBM',
            'oracle': 'ORCL', 'orcl': 'ORCL',
            'salesforce': 'CRM', 'crm': 'CRM',
            'zoom': 'ZM', 'zm': 'ZM',
            'slack': 'WORK', 'work': 'WORK',
            'shopify': 'SHOP', 'shop': 'SHOP',
            'spotify': 'SPOT', 'spot': 'SPOT',
            'uber': 'UBER',
            'lyft': 'LYFT',
            'airbnb': 'ABNB', 'abnb': 'ABNB',
            'square': 'SQ', 'sq': 'SQ', 'block': 'SQ',
            'palantir': 'PLTR', 'pltr': 'PLTR',
            'snowflake': 'SNOW', 'snow': 'SNOW'
        }
        
        # create a sorted list (without duplicates) for the display
        self.stock_symbols_display = sorted(list(set(self.available_stocks.values())))
    
    #to exstract stock symbol from user's message
    def extract_stock_symbol(self, message):
        """Estrae il simbolo dello stock dal messaggio"""
        message_lower = message.lower()
        
        # search for keyword in the message, if it find one-> return corrispective symbol
        for keyword, symbol in self.available_stocks.items():
            if keyword in message_lower:
                return symbol
        
        # Check if a symbol has been inserted directly
        words = message_lower.split()
        for word in words:
            word_upper = word.upper()
            if word_upper in self.stock_symbols_display:
                return word_upper
        
        return None

    #to generate an help message about best available stocks (group by categories)
    def get_stocks_help_message(self):
        """Restituisce messaggio di aiuto per le azioni"""
        categories = {
            "🔥 **Tech Giants**": ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NFLX", "TSLA", "NVDA"],
            "🏦 **Finanziari**": ["JPM", "BAC", "WFC", "GS", "V", "MA", "PYPL"],
            "💊 **Salute/Farmaceutico**": ["JNJ", "PFE", "MRNA", "ABT", "MRK"],
            "⚡ **Energia**": ["XOM", "CVX", "COP"],
            "🛍️ **Beni Consumo**": ["KO", "PEP", "PG", "NKE", "WMT", "HD", "MCD", "SBUX", "DIS"],
            "🏭 **Industriali**": ["BA", "CAT", "MMM", "GE", "LMT"],
            "📱 **Telecom**": ["VZ", "T", "TMUS"],
            "💎 **Altri Popolari**": ["BRK.B", "IBM", "ORCL", "CRM", "ZM", "SHOP", "UBER", "ABNB"]
        }
        
        result = "📊 **Azioni Disponibili:**\n\n"
        
        for category, symbols in categories.items():
            result += f"{category}:\n"
            display_symbols = symbols[:8]
            result += f"   {' • '.join(display_symbols)}"
            if len(symbols) > 8:
                result += f" *+{len(symbols)-8} altri*"
            result += "\n\n"
        
        #example of good prompts
        result += "💡 **Esempi di utilizzo:**\n"
        result += "• 'Prezzo azioni Apple' o 'Quanto vale AAPL?'\n"
        result += "• 'Azioni Tesla oggi' o 'Azioni TSLA'\n"
        result += "• 'Microsoft stock' o 'MSFT prezzo'\n\n"
        result += f"📈 *Totale: {len(self.stock_symbols_display)} azioni disponibili*"
        
        return result

    #to load intents.json and to populate lists and dicts with intents data
    def load_intents(self):
        with open(self.intents_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for intent in data['intents']:
                self.intents.append(intent['tag'])
                self.intents_data[intent['tag']] = intent

    #to load trained model
    def load_model(self):
        if not os.path.exists('chatbot_distilbert_robust.pth'):
            raise FileNotFoundError("❌ File chatbot_distilbert_robust.pth non trovato. Devi addestrare e salvare il modello prima.")

        checkpoint = torch.load('chatbot_distilbert_robust.pth', map_location='cpu',weights_only=False)
        num_classes = checkpoint.get('num_classes', len(self.intents))
        
        self.model = ImprovedChatbotModel(768, num_classes, hidden_size=256, dropout_rate=0.3)
        #load weights of the trained model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        #load label mappings 
        self.label_to_idx = checkpoint.get('label_to_idx', {tag: i for i, tag in enumerate(self.intents)})
        self.idx_to_label = checkpoint.get('idx_to_label', {i: tag for i, tag in enumerate(self.intents)})

        best_acc = checkpoint.get('best_accuracy', None)
        print(f"✅ Modello chatbot_distilbert_robust.pth caricato con successo" +
            (f" (best acc: {best_acc:.2f}%)" if best_acc is not None else ""))

        #set the model in EVAL mode (no training)
        self.model.eval()

    #ENCODE TEXT: tokenize text using DistilBERT's tokenizer and prepare input for ML model
    def encode_text(self, text):
        encoding = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']

#------------------------------------
#REAL API SERVICES(with rate limiting)
#------------------------------------

    # to return real weather (API: OpenWeatherMap)
    def get_weather_real(self, city="Roma"):
        """API OpenWeatherMap con rate limiting"""
        # check rate limit
        can_request, message = rate_limiter.can_make_request('weather')
        if not can_request:
            return f"⚠️ {message}. Riprova domani!"
        
        if not OPENWEATHER_API_KEY:
            return self.get_weather_fallback(city)
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': city,
                'appid': OPENWEATHER_API_KEY,
                'units': 'metric',
                'lang': 'it'
            }
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if response.status_code == 200:
                #UPTADE rate limiter
                rate_limiter.increment_counter('weather')
                
                temp = data['main']['temp']
                feels_like = data['main']['feels_like']
                humidity = data['main']['humidity']
                description = data['weather'][0]['description']
                
                # add limits info
                stats = rate_limiter.get_stats()['weather']
                
                return f"🌤️ **Meteo {city}:**\n" \
                       f"🌡️ Temperatura: {temp}°C (percepita {feels_like}°C)\n" \
                       f"☁️ Condizioni: {description.title()}\n" \
                       f"💧 Umidità: {humidity}%\n\n" \
                       f"📊 *API calls rimanenti oggi: {stats['remaining']}/{stats['limit']}*"
            else:
                return self.get_weather_fallback(city)
                
        except Exception as e:
            print(f"Weather API error: {e}")
            return self.get_weather_fallback(city)

    #return simulated weather (only if API failed)
    def get_weather_fallback(self, city):
        """Meteo simulato se API non disponibile o limite superato"""
        temps = [18, 19, 20, 21, 22, 23, 24, 25]
        conditions = ["soleggiato", "parzialmente nuvoloso", "nuvoloso"]
        
        return f"🌤️ **Meteo {city}** (simulato):\n" \
               f"🌡️ Temperatura: {random.choice(temps)}°C\n" \
               f"☁️ Condizioni: {random.choice(conditions)}\n" \
               f"💧 Umidità: {random.randint(45, 75)}%"

    # to return real stock prices (API: Alpha Vantage)
    def get_stock_prices_real(self, symbol=None):
        """API Alpha Vantage con rate limiting per stock specifici"""
        # check rate limit
        can_request, message = rate_limiter.can_make_request('stocks')
        if not can_request:
            return f"⚠️ {message}. Riprova tra qualche minuto!"
        
        if not ALPHA_VANTAGE_API_KEY:
            return self.get_stock_prices_fallback(symbol)
        
        try:
            # If a stock is not specified, use AAPL as the default
            if not symbol:
                symbol = 'AAPL'
            symbol = symbol.upper()
            
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': ALPHA_VANTAGE_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            # Debug: print API response
            print(f"Stock API response for {symbol}: {response.status_code}")
            print(f"Stock API data: {data}")
            
            if 'Global Quote' in data and data['Global Quote']:
                # update rate limiter
                rate_limiter.increment_counter('stocks')
                
                quote = data['Global Quote']
                price = float(quote.get('05. price', 0))
                change = float(quote.get('09. change', 0))
                change_pct = float(quote.get('10. change percent', '0').replace('%', ''))
                
                emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                
                #add limits info
                stats = rate_limiter.get_stats()['stocks']
                
                return f"📊 **Prezzo Azione {symbol}:**\n" \
                    f"💰 **{symbol}**: ${price:.2f} {emoji} {change_pct:+.2f}%\n\n" \
                    f"📊 *API calls rimanenti: {stats['remaining']}/{stats['limit']} (giorno), 4/5 (minuto)*"
            else:
                error_msg = data.get('Note', data.get('Information', 'Errore sconosciuto'))
                print(f"Stock API error for {symbol}: {error_msg}")
                return self.get_stock_prices_fallback(symbol)
                
        except Exception as e:
            print(f"Stock API exception for {symbol}: {e}")
            return self.get_stock_prices_fallback(symbol)

    #to return (randomly) simulated stock prices (only if API failed or stock not available)
    def get_stock_prices_fallback(self, symbol=None):
        """Prezzi stock simulati - SOLO quando API non disponibile o azione non supportata"""
        popular_stocks = {
            'AAPL': random.uniform(180, 190),
            'MSFT': random.uniform(370, 380), 
            'GOOGL': random.uniform(135, 145),
            'META': random.uniform(320, 340),
            'AMZN': random.uniform(140, 160),
            'TSLA': random.uniform(240, 260),
            'NVDA': random.uniform(490, 510),
            'NFLX': random.uniform(450, 470),
            'JPM': random.uniform(145, 155),
            'V': random.uniform(240, 250),
            'JNJ': random.uniform(160, 170),
            'KO': random.uniform(58, 62),
            'DIS': random.uniform(95, 105),
            'BA': random.uniform(200, 220)
        }
        
        if symbol:
            symbol = symbol.upper()
            if symbol in self.stock_symbols_display:
                price = popular_stocks.get(symbol, random.uniform(50, 500))
                change = random.choice(["📈", "📉", "➡️"])
                pct = random.uniform(-5, 5)
                return f"📊 **Prezzo Azione {symbol}** (simulato - API non disponibile):\n💰 **{symbol}**: ${price:.2f} {change} {pct:+.2f}%"
            else:
                return f"❌ **Azione '{symbol}' non supportata.**\n\n{self.get_stocks_help_message()}"
        
        # if stock not specificated 
        result = "📊 **Top Azioni** (simulato - API non disponibile):\n\n"
        display_stocks = dict(list(popular_stocks.items())[:8])  # shows first 8 stocks
        
        for symbol, price in display_stocks.items():
            change = random.choice(["📈", "📉", "➡️"])
            pct = random.uniform(-3, 3)
            result += f"• **{symbol}**: ${price:.2f} {change} {pct:+.2f}%\n"
        
        result += f"\n💡 *Per vedere un'azione specifica scrivi: 'prezzo [nome azione]'*\n"
        result += f"📋 *{len(self.stock_symbols_display)} azioni disponibili - scrivi 'azioni disponibili' per la lista completa*"
        
        return result
    
    # to return real news from US (API: NewsAPI)
    def get_news_real(self):
        """API NewsAPI con rate limiting"""
        # check rate limit
        can_request, message = rate_limiter.can_make_request('news')
        if not can_request:
            return f"⚠️ {message}. Riprova domani!"
        
        if not NEWS_API_KEY:
            return self.get_news_fallback()
        
        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                'country': 'us',
                'category': 'general',
                'pageSize': 5,
                'apiKey': NEWS_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if response.status_code == 200 and data['articles']:
                #update rate limiter
                rate_limiter.increment_counter('news')
                
                news_text = "📰 **Ultime Notizie dagli Stati Uniti:**\n\n"
                
                for i, article in enumerate(data['articles'][:5], 1):
                    title = article['title']
                    source = article['source']['name']
                    
                    if len(title) > 80:
                        title = title[:77] + "..."
                    
                    news_text += f"**{i}.** {title}\n"
                    news_text += f"   *Fonte: {source}*\n\n"
                
                # add limit info
                stats = rate_limiter.get_stats()['news']
                news_text += f"📊 *API calls rimanenti oggi: {stats['remaining']}/{stats['limit']}*"
                
                return news_text
            else:
                return self.get_news_fallback()
                
        except Exception as e:
            print(f"News API error: {e}")
            return self.get_news_fallback()

    #to return simulated news (only if API failed)
    def get_news_fallback(self):
        """Notizie simulate"""
        fake_news = [
            "🚀 Nuova missione spaziale italiana lanciata con successo",
            "💻 Importante aggiornamento di sicurezza per tutti i dispositivi",
            "🏆 L'Italia vince un prestigioso premio internazionale",
            "🌱 Nuova tecnologia verde sviluppata da startup italiana",
            "📱 Rilasciata nuova versione dell'app di messaggistica più popolare"
        ]
        
        result = "📰 **Ultime Notizie** (simulate):\n\n"
        for i, news in enumerate(random.sample(fake_news, 3), 1):
            result += f"**{i}.** {news}\n\n"
        
        return result

    # to return real crypto prices (API: CoinGecko) (no rate limiting needded)
    def get_crypto_prices(self):
        """API CoinGecko gratuita (no rate limiting necessario)"""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': 'bitcoin,ethereum,cardano,polkadot,chainlink',
                'vs_currencies': 'usd',
                'include_24hr_change': 'true'
            }
            
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if response.status_code == 200:
                result = "₿ **Prezzi Criptovalute:**\n\n"
                
                crypto_names = {
                    'bitcoin': 'Bitcoin (BTC)',
                    'ethereum': 'Ethereum (ETH)',
                    'cardano': 'Cardano (ADA)',
                    'polkadot': 'Polkadot (DOT)',
                    'chainlink': 'Chainlink (LINK)'
                }
                
                for crypto_id, crypto_data in data.items():
                    name = crypto_names.get(crypto_id, crypto_id)
                    price = crypto_data['usd']
                    change = crypto_data.get('usd_24h_change', 0)
                    
                    emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    result += f"• **{name}**: ${price:,.2f} {emoji} {change:+.2f}%\n"
                
                return result
            else:
                return "❌ Errore nel recupero prezzi crypto"
                
        except Exception as e:
            print(f"Crypto API error: {e}")
            return "❌ Servizio crypto temporaneamente non disponibile"

    #to terurn current time 
    def get_current_time(self):
        """Ora attuale"""
        now = datetime.now()
        return f"🕐 **Ora attuale:** {now.strftime('%H:%M:%S')}\n" \
               f"📅 **Data:** {now.strftime('%d/%m/%Y')}\n" \
               f"📆 **Giorno:** {now.strftime('%A')}"

    #to return a random joke
    def get_joke(self):
        """Barzellette casuali"""
        jokes = [
            "Perché i programmatori preferiscono il buio? Perché la luce attira i bug! 🐛",
            "Come chiami un pesce con due ginocchia? Un pesce-ginocchio! 🐟",
            "Cosa dice un AI quando è felice? 'Sono a 1 e 0 dalla felicità!' 🤖",
            "Perché l'AI non si ammala mai? Perché ha sempre l'antivirus aggiornato! 💻",
            "Cosa fa un bot quando è triste? Si auto-debugga! 🔧"
        ]
        return random.choice(jokes)

    #to do management
    def manage_todo(self, user_id, action, task=None, priority=1):
        if action == 'add' and task:
            # Extract priority from text if present and set it
            if any(word in task.lower() for word in ['importante', 'urgente', 'priorità']):
                priority = 3
            elif any(word in task.lower() for word in ['bassa', 'quando possibile']):
                priority = 1
            else:
                priority = 2
            
            #insert task into db (with priority)
            with self.memory.get_db() as conn:
                conn.execute('INSERT INTO todos (user_id, task, priority) VALUES (?, ?, ?)', 
                           (user_id, task, priority))
                conn.commit()
            
            priority_text = {1: "bassa", 2: "media", 3: "alta"}[priority]
            return f"✅ Aggiunto '{task}' con priorità {priority_text}."

        #show task list order by priority    
        elif action == 'list':
            with self.memory.get_db() as conn:
                cursor = conn.execute('''
                    SELECT id, task, priority FROM todos 
                    WHERE user_id=? AND completed=0 
                    ORDER BY priority DESC, created_at ASC
                ''', (user_id,))
                todos = cursor.fetchall()
            
            if not todos:
                return "📝 La tua lista è vuota! 🎉"
            
            result = "📝 **Le tue cose da fare:**\n\n"
            priority_emojis = {1: "🔵", 2: "🟡", 3: "🔴"}
            
            for todo in todos:
                emoji = priority_emojis.get(todo['priority'], "⚪")
                result += f"{emoji} **{todo['id']}:** {todo['task']}\n"
            
            result += "\n💡 *Scrivi 'completa [numero]' per completare un task*"
            return result

        #complete task (set as completed)     
        elif action == 'complete' and task:
            try:
                task_id = int(task)
                with self.memory.get_db() as conn:
                    cursor = conn.execute('SELECT task FROM todos WHERE id=? AND user_id=? AND completed=0', 
                                        (task_id, user_id))
                    todo = cursor.fetchone()
                    
                    if todo:
                        conn.execute('UPDATE todos SET completed=1 WHERE id=? AND user_id=?', 
                                   (task_id, user_id))
                        conn.commit()
                        return f"🎉 Task completato: '{todo['task']}'!"
                    else:
                        return "❌ Task non trovato o già completato."
            except:
                return "❌ ID task non valido."
                
        return "❓ Comando todo non riconosciuto. Prova: 'aggiungi [task]', 'mostra lista', 'completa [id]'"
    
    #-------------------
    # Message processing
    #-------------------
    def process_message(self, message, user_id):
        input_ids, attention_mask = self.encode_text(message)
        
        #tokenize message and send it to the model
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            confidence_scores = torch.softmax(outputs, dim=1)
            #obtain prediction and confidence score
            predicted_idx = torch.argmax(outputs, dim=1).item()
            confidence = confidence_scores[0][predicted_idx].item()
        
        # Use idx_to_label if available, otherwise fallback
        if self.idx_to_label:
            #return predicted intent
            intent = self.idx_to_label[predicted_idx]
        else:
            #return predicted intent
            intent = self.intents[predicted_idx] if predicted_idx < len(self.intents) else 'unknown'
        
        # Confidence threshold for asking clarification
        if confidence < 0.5:
            help_message = self.get_help_message()
            response = f"🤔 Non sono sicuro di aver capito la tua richiesta. {help_message}"
        
        #------------------------
        # Handle specific intents
        #------------------------

        # CASE 1: stocks
        elif intent == 'stocks':
            if 'crypto' in message.lower() or 'bitcoin' in message.lower():
                response = self.get_crypto_prices()
            else:
                # Check if it asks for the list of available actions
                if any(phrase in message.lower() for phrase in ['disponibili', 'lista', 'elenco', 'quali azioni', 'che azioni']):
                    response = self.get_stocks_help_message()
                else:
                    # Extract the stock symbol from the message
                    stock_symbol = self.extract_stock_symbol(message)
                    
                    if stock_symbol:
                        # Action found and supported - try real API first
                        if stock_symbol in self.stock_symbols_display:
                            response = self.get_stock_prices_real(stock_symbol)
                        else:
                            # Action recognized but not supported by APIs
                            response = f"❌ **Azione '{stock_symbol}' non supportata dalle nostre API.**\n\n{self.get_stocks_help_message()}"
                    else:
                        # No actions specified or not recognized
                        if any(word in message.lower() for word in ['prezzo', 'quanto', 'valore', 'quotazione', 'stock', 'azione']):
                            # User asked for a price but without specifying the unrecognized stock or stock
                            response = f"🤔 **Dimmi quale azione ti interessa!**\n\n{self.get_stocks_help_message()}"
                        else:
                            # Generic request for stocks - show help
                            response = f"📊 **Servizio Azioni Disponibile!**\n\n{self.get_stocks_help_message()}"
                        
        # CASE 2: weather
        elif intent == 'weather':
            # Extract city from message if present
            city = "Roma"  # Default
            italian_cities = ['roma', 'milano', 'napoli', 'torino', 'palermo', 'genova', 'bologna', 'firenze', 'bari', 'catania']
            for c in italian_cities:
                if c in message.lower():
                    city = c.title()
                    break
            response = self.get_weather_real(city)
            
        # CASE 3: news
        elif intent == 'news':
            response = self.get_news_real()
        
        # CASE 4: time
        elif intent == 'time':
            response = self.get_current_time()

        # CASE 5: joke
        elif intent == 'joke':
            response = self.get_joke()
        
        # CASE 6: to do
        elif intent == 'todo':
            #extract task from message 
            if 'aggiungi' in message.lower():
                task = message.lower().replace('aggiungi', '').strip()
                task = task.replace('alla lista', '').replace('todo', '').strip()
                if task:
                    #add task 
                    response = self.manage_todo(user_id, 'add', task)
                else:
                    response = "❓ Cosa devo aggiungere alla lista?"
            #show to do list
            elif any(word in message.lower() for word in ['mostra', 'lista', 'elenco', 'visualizza']):
                response = self.manage_todo(user_id, 'list')
            #complete task
            elif 'completa' in message.lower():
                for word in message.split():
                    if word.isdigit():
                        response = self.manage_todo(user_id, 'complete', word)
                        break
                else:
                    #which task do you want to complete?
                    response = "❓ Specifica il numero del task da completare (es: 'completa 1')"
            else:
                #add task
                response = self.manage_todo(user_id, 'list')
      
        # CASE 7: greeting
        elif intent == 'greeting':
            responses = self.intents_data.get(intent, {}).get('responses', [])
            if responses:
                response = random.choice(responses) + " " + self.get_help_message()
            else:
                response = "Ciao! " + self.get_help_message()

        # CASE 8: thanks
        elif intent == 'thanks':
            responses = self.intents_data.get(intent, {}).get('responses', [])
            if responses:
                response = random.choice(responses)
            else:
                response = "Di niente! Sono qui per aiutarti. 😊"

        # CASE 9: help
        elif intent == 'help':
            response = self.get_help_message()        
        
        # CASE 10:greeting - mood - name - goodbye - thanks
        else:
            # First try the standard responses of intent 
            responses = self.intents_data.get(intent, {}).get('responses', [])
            # if there are available responses: use a random response: among those predefined for that intent 
            if responses:
                response = random.choice(responses)
            else:
                # show error message
                help_message = self.get_help_message()
                response = f"❓ Non ho riconosciuto la tua richiesta. {help_message}"
       
        # Save conversation
        self.memory.add_conversation(user_id, message, response, intent)
        return response, intent, confidence
    
    # to return hel message
    def get_help_message(self):
        """Restituisce un messaggio di aiuto con esempi di comandi"""
        help_examples = [
            "🌤️ **Meteo**: 'Che tempo fa a Roma?', 'Meteo Milano'",
            "📰 **Notizie**: 'Ultime notizie', 'News di oggi'",
            "📊 **Azioni**: 'Prezzo azioni Apple', 'Quota Microsoft', 'Azioni Tesla'",
            "₿ **Crypto**: 'Prezzo Bitcoin', 'Crypto oggi'",
            "📝 **Todo List**: 'Aggiungi comprare latte', 'Mostra lista', 'Completa 1'",
            "😄 **Barzellette**: 'Raccontami una barzelletta', 'Fai ridere'",
            "🕐 **Orario**: 'Che ore sono?', 'Data di oggi'"
        ]
        
        return (
            "Ecco cosa posso fare per te:\n\n" +
            "\n".join(f"• {example}" for example in help_examples) +
            "\n\n💡 *Prova uno di questi comandi!*"
        )

# Initialize assistant
assistant = EnhancedChatbotAssistant('intents.json')

#main route (web page)
@app.route('/')
def index():
    #create a unique ID user 
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template('index.html')

# endpoint /chat: used to send messages (only POST) 
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    #return the message
    message = data.get('message', '').strip()
    #return user id
    user_id = session.get('user_id', 'anonymous')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    #process message using AI Chatbot
    response, intent, confidence = assistant.process_message(message, user_id)
    
    #return a json response with (response, intent, confidence, timestamp and user id)
    return jsonify({
        'response': response,
        'intent': intent,
        'confidence': round(confidence * 100, 1),
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id
    })

# endpoint /chat: used to return last 10 user's conversation
@app.route('/history', methods=['GET'])
def history():
    user_id = session.get('user_id', 'anonymous')
    #read from DB 
    history = assistant.memory.get_recent_context(user_id, limit=10)
    #convert to DICT for the JSON serialization
    return jsonify({'history': [dict(h) for h in history]})

@app.route('/api-stats', methods=['GET'])
def api_stats():
    """Endpoint per vedere statistiche uso API"""
    stats = rate_limiter.get_stats()
    return jsonify({
        'api_usage': stats,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/reset-limits', methods=['POST'])
def reset_limits():
    """Endpoint per resettare i limiti (solo per admin/development)"""
    # In produzione potresti voler aggiungere autenticazione qui
    admin_key = request.json.get('admin_key') if request.json else None
    expected_key = os.environ.get('ADMIN_KEY', 'admin123')  # Cambia in produzione!
    
    if admin_key != expected_key:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Reset contatori
    today = datetime.now().date()
    for api_type in rate_limiter.daily_counters:
        rate_limiter.daily_counters[api_type]['count'] = 0
        rate_limiter.daily_counters[api_type]['reset_date'] = today
    
    # Clear minute counters
    rate_limiter.minute_counters.clear()
    
    return jsonify({
        'message': 'Rate limits reset successfully',
        'new_stats': rate_limiter.get_stats(),
        'timestamp': datetime.now().isoformat()
    })



@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': assistant.model is not None,
        'api_keys_configured': {
            'weather': bool(OPENWEATHER_API_KEY),
            'news': bool(NEWS_API_KEY),
            'stocks': bool(ALPHA_VANTAGE_API_KEY)
        },
        'api_usage': rate_limiter.get_stats(),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 AI Assistant started on port {port}")
    print(f"🔑 Weather API: {'✅' if OPENWEATHER_API_KEY else '❌'}")
    print(f"🔑 News API: {'✅' if NEWS_API_KEY else '❌'}")
    print(f"🔑 Stock API: {'✅' if ALPHA_VANTAGE_API_KEY else '❌'}")
    print(f"🛡️ Rate limiting attivo: Weather (1000/giorno), News (1000/giorno), Stocks (500/giorno, 5/minuto)")
    app.run(host='0.0.0.0', port=port, debug=False)