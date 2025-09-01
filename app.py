import os
import json
import random
from datetime import datetime
import time
from contextlib import contextmanager
from typing import Dict, Tuple
import logging
import sqlite3
import threading
from zoneinfo import ZoneInfo 

os.makedirs('/app/data', exist_ok=True)

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
    
    def __init__(self, db_path: str = "/app/data/rate_limits.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        self.API_LIMITS = {
            'weather': {'daily_limit': 1000, 'minute_limit': None, 'fallback_enabled': True},
            'news': {'daily_limit': 100, 'minute_limit': None, 'fallback_enabled': True},
            'stocks': {'daily_limit': 25, 'minute_limit': 5, 'fallback_enabled': True}
        }
        
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600
    
    #initialize tables for rate limiting
    def init_database(self):
        with self._get_db_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_limits (
                    api_type TEXT PRIMARY KEY,
                    count INTEGER DEFAULT 0,
                    reset_date TEXT,
                    last_updated REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS minute_limits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_type TEXT,
                    timestamp REAL,
                    UNIQUE(api_type, timestamp)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS api_usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_type TEXT,
                    timestamp REAL,
                    success BOOLEAN,
                    fallback_used BOOLEAN,
                    user_session TEXT
                )
            ''')
            conn.commit()
    
    #connection to rate limiting DB
    @contextmanager
    def _get_db_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    #check if is it possible to do an API request
    def can_make_request(self, api_type: str, user_session: str = "anonymous") -> Tuple[bool, str, bool]:
        if api_type not in self.API_LIMITS:
            return False, f"API type '{api_type}' not supported", True
        
        with self.lock:
            self._cleanup_old_records()
            
            daily_ok, daily_msg = self._check_daily_limit(api_type)
            if not daily_ok:
                self._log_usage(api_type, False, True, user_session)
                return False, daily_msg, True
            
            minute_ok, minute_msg = self._check_minute_limit(api_type)
            if not minute_ok:
                self._log_usage(api_type, False, True, user_session)
                return False, minute_msg, True
            
            return True, "OK", False
    
    #check daily limit
    def _check_daily_limit(self, api_type: str) -> Tuple[bool, str]:
        today = datetime.now().date().isoformat()
        daily_limit = self.API_LIMITS[api_type]['daily_limit']
        
        with self._get_db_connection() as conn:
            cursor = conn.execute('SELECT count, reset_date FROM daily_limits WHERE api_type = ?', (api_type,))
            row = cursor.fetchone()
            
            if row is None:
                conn.execute('INSERT INTO daily_limits (api_type, count, reset_date, last_updated) VALUES (?, 0, ?, ?)', 
                           (api_type, today, time.time()))
                conn.commit()
                return True, "OK"
            
            current_count = row['count']
            reset_date = row['reset_date']
            
            if reset_date != today:
                conn.execute('UPDATE daily_limits SET count = 0, reset_date = ?, last_updated = ? WHERE api_type = ?', 
                           (today, time.time(), api_type))
                conn.commit()
                return True, "OK"
            
            if current_count >= daily_limit:
                remaining_hours = 24 - datetime.now().hour
                return False, f"Limite giornaliero raggiunto per {api_type} ({current_count}/{daily_limit}). Reset in {remaining_hours}h"
            
            return True, "OK"
    
    #check minute limit (only for stocks API)
    def _check_minute_limit(self, api_type: str) -> Tuple[bool, str]:
        minute_limit = self.API_LIMITS[api_type].get('minute_limit')
        if minute_limit is None:
            return True, "OK"
        
        now = time.time()
        minute_ago = now - 60
        
        with self._get_db_connection() as conn:
            conn.execute('DELETE FROM minute_limits WHERE api_type = ? AND timestamp < ?', (api_type, minute_ago))
            cursor = conn.execute('SELECT COUNT(*) as count FROM minute_limits WHERE api_type = ? AND timestamp >= ?', 
                                (api_type, minute_ago))
            current_count = cursor.fetchone()['count']
            
            if current_count >= minute_limit:
                return False, f"Limite di {minute_limit} chiamate al minuto raggiunto per {api_type}"
            
            conn.commit()
            return True, "OK"
    
    #inrement counter after a succcess call
    def increment_counter(self, api_type: str, user_session: str = "anonymous"):
        with self.lock:
            now = time.time()
            
            with self._get_db_connection() as conn:
                conn.execute('UPDATE daily_limits SET count = count + 1, last_updated = ? WHERE api_type = ?', 
                           (now, api_type))
                
                if self.API_LIMITS[api_type].get('minute_limit'):
                    conn.execute('INSERT OR IGNORE INTO minute_limits (api_type, timestamp) VALUES (?, ?)', 
                               (api_type, now))
                
                conn.commit()
            
            self._log_usage(api_type, True, False, user_session)
    
    def _log_usage(self, api_type: str, success: bool, fallback_used: bool, user_session: str):
        try:
            with self._get_db_connection() as conn:
                conn.execute('''INSERT INTO api_usage_log (api_type, timestamp, success, fallback_used, user_session)
                               VALUES (?, ?, ?, ?, ?)''', (api_type, time.time(), success, fallback_used, user_session))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Errore nel logging utilizzo API: {e}")
    
    #to obtain all the API statistics
    def get_stats(self) -> Dict:
        stats = {}
        today = datetime.now().date().isoformat()
        
        with self._get_db_connection() as conn:
            for api_type, config in self.API_LIMITS.items():
                cursor = conn.execute('SELECT count, reset_date FROM daily_limits WHERE api_type = ?', (api_type,))
                row = cursor.fetchone()
                
                used = 0 if row is None or row['reset_date'] != today else row['count']
                limit = config['daily_limit']
                remaining = max(0, limit - used)
                
                stats[api_type] = {
                    'used': used, 'limit': limit, 'remaining': remaining,
                    'percentage_used': round((used / limit) * 100, 1),
                    'fallback_enabled': config['fallback_enabled']
                }
                
                if config.get('minute_limit'):
                    minute_ago = time.time() - 60
                    cursor = conn.execute('SELECT COUNT(*) as minute_count FROM minute_limits WHERE api_type = ? AND timestamp >= ?', 
                                        (api_type, minute_ago))
                    minute_used = cursor.fetchone()['minute_count']
                    stats[api_type].update({
                        'minute_used': minute_used,
                        'minute_limit': config['minute_limit'],
                        'minute_remaining': max(0, config['minute_limit'] - minute_used)
                    })
        
        return stats
    
    def _cleanup_old_records(self):
        if time.time() - self._last_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = time.time()
        cutoff_time = time.time() - (7 * 24 * 3600)
        
        with self._get_db_connection() as conn:
            conn.execute('DELETE FROM minute_limits WHERE timestamp < ?', (time.time() - 120,))
            conn.execute('DELETE FROM api_usage_log WHERE timestamp < ?', (cutoff_time,))
            conn.commit()


# Initialize rate limiter
rate_limiter = APIRateLimiter('/app/data/rate_limits.db')

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
    def __init__(self, db_path=None):
        #create a dict in the memory to isolate data for each session
        self._in_memory_storage = {}
        #to understand if we are in shared docker envirenment (Huggin Face) (true) or local (false)
        self.is_shared_environment = os.environ.get('DEPLOYMENT_ENV') == 'huggingface'
        
        if db_path is None:
            if self.is_shared_environment:
                #shared environment: use temporary memory
                self.db_path = ':memory:'
            else: 
                #local environment: use persistency file
                self.db_path = os.environ.get('DB_PATH', '/app/data/conversations.db')
        else:
            self.db_path = db_path

        if not self.is_shared_environment: 
           os.makedirs(os.path.dirname(self.db_path) if self.db_path != ':memory:' else '/tmp', exist_ok=True)
       
        self.init_db()

    #to create a unique key fot the session
    def _get_session_key(self, user_id, session_id=None):
        if session_id:
            return f"{user_id}_{session_id}"
        return user_id
    
    #to be sure that storage exist for this session
    def _ensure_session_storage(self, session_key):
        if self.is_shared_environment:
            if session_key not in self._in_memory_storage:
                self._in_memory_storage[session_key] = {
                    'conversations': [],
                    'todos': []
                }
    
    #inizialize db
    def init_db(self):
        with self.get_db() as conn:
            #create conversation table 
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_id TEXT,
                    timestamp DATETIME,
                    message TEXT,
                    response TEXT,
                    intent TEXT
                )
            ''')

            #MIGRATION
            try:
                conn.execute('ALTER TABLE conversations ADD COLUMN session_id TEXT')
                print("✅ Aggiunta colonna session_id alla tabella conversations")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print("ℹ️ Colonna session_id già presente")
                else:
                    print(f"❌ Errore aggiunta colonna: {e}")

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

            #MIGRATION
            try:
                conn.execute('ALTER TABLE todos ADD COLUMN session_id TEXT')
                print("✅ Aggiunta colonna session_id alla tabella todos")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print("ℹ️ Colonna session_id già presente in todos")
                else:
                    print(f"❌ Errore aggiunta session_id a todos: {e}")

            # table for tracking sessions
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP
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
    def add_conversation(self, user_id, message, response, intent, session_id = None):
        if self.is_shared_environment:
            # Use in-memory storage for shared environment (Hugging Face)
            if not session_id:
                session_id = session.get('session_id', str(uuid.uuid4()))
                
            session_key = self._get_session_key(user_id, session_id)
            self._ensure_session_storage(session_key)
            
            conversation = {
                'id': str(uuid.uuid4()),
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': datetime.now(),
                'message': message,
                'response': response,
                'intent': intent
            }
            self._in_memory_storage[session_key]['conversations'].append(conversation)

        else:
            # Use database for local environment
            if not session_id:
                session_id = session.get('session_id', str(uuid.uuid4()))
            
            #open a connection
            with self.get_db() as conn:
                #insert to DB
                conn.execute('''
                    INSERT INTO conversations (id, user_id, session_id, timestamp, message, response, intent)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (str(uuid.uuid4()), user_id, session_id, datetime.now(), message, response, intent))
                
                #update session activity
                conn.execute('''
                    INSERT OR REPLACE INTO sessions (session_id, user_id, last_activity)
                    VALUES (?, ?, ?)
                ''', (session_id, user_id, datetime.now()))
                
                #confirm the transation
                conn.commit()

    #to retrieve the recent context of a specific user
    def get_recent_context(self, user_id, limit=5, session_id = None):
        if self.is_shared_environment:
            # Use in-memory storage for shared environment
            session_key = self._get_session_key(user_id, session_id)
            self._ensure_session_storage(session_key)
            
            conversations = self._in_memory_storage[session_key]['conversations']
            # Return the most recent conversations (most recent first)
            recent = conversations[-limit:] if conversations else []
            return [{'message': c['message'], 'response': c['response'], 'intent': c['intent']} 
                for c in reversed(recent)]
        else:
            #use database for locla environment
            with self.get_db() as conn:
                #select user's last conversation (order by most recent, limited to 5)
                if session_id:
                    # Filter by specific session (shared environment)
                    cursor = conn.execute('''
                        SELECT message, response, intent FROM conversations
                        WHERE user_id = ? AND session_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (user_id, session_id, limit))
                else:
                    #use only user_id (local environment)
                    cursor = conn.execute('''
                        SELECT message, response, intent FROM conversations
                        WHERE user_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (user_id, limit))
                #return all rows resulting from query 
                return cursor.fetchall()
    
    #to delete user's conversations and, if we are in share environment, session's conversations 
    def clear_user_conversations(self, user_id, session_id=None):
        if self.is_shared_environment:
            # Delete only data from this specific session
            session_key = self._get_session_key(user_id, session_id)
            if session_key in self._in_memory_storage:
                self._in_memory_storage[session_key] = {
                    'conversations': [],
                    'todos': []
                }
        else:
            with self.get_db() as conn:
                if session_id:
                    conn.execute('DELETE FROM conversations WHERE user_id = ? AND session_id = ?', (user_id, session_id))
                    conn.execute('DELETE FROM todos WHERE user_id = ? AND session_id = ?', (user_id, session_id))
                else:
                    conn.execute('DELETE FROM conversations WHERE user_id = ?', (user_id,))
                    conn.execute('DELETE FROM todos WHERE user_id = ?', (user_id,))
                conn.commit()

    def add_todo(self, user_id, task, priority=1, session_id=None):
        if self.is_shared_environment:
            session_key = self._get_session_key(user_id, session_id)
            self._ensure_session_storage(session_key)
            
            # Find next available ID
            existing_todos = self._in_memory_storage[session_key]['todos']
            next_id = max([t['id'] for t in existing_todos], default=0) + 1
            
            todo = {
                'id': next_id,
                'user_id': user_id,
                'task': task,
                'completed': False,
                'created_at': datetime.now(),
                'priority': priority,
                'session_id': session_id
            }
            self._in_memory_storage[session_key]['todos'].append(todo)
            return next_id
        else:
            with self.get_db() as conn:
                cursor = conn.execute('INSERT INTO todos (user_id, task, priority, session_id) VALUES (?, ?, ?, ?)', 
                                    (user_id, task, priority, session_id))
                conn.commit()
                return cursor.lastrowid

    def get_todos(self, user_id, session_id=None):
        if self.is_shared_environment:
            session_key = self._get_session_key(user_id, session_id)
            self._ensure_session_storage(session_key)
            
            todos = self._in_memory_storage[session_key]['todos']
            # Return only uncompleted TODOs, sorted by priority
            return [t for t in todos if not t['completed']]
        else:
            with self.get_db() as conn:
                cursor = conn.execute('''
                    SELECT id, task, priority FROM todos 
                    WHERE user_id=? AND session_id=? AND completed=0 
                    ORDER BY priority DESC, created_at ASC
                ''', (user_id, session_id))
                return cursor.fetchall()
    
    def complete_todo(self, user_id, todo_id, session_id=None):
        if self.is_shared_environment:
            session_key = self._get_session_key(user_id, session_id)
            self._ensure_session_storage(session_key)
            
            todos = self._in_memory_storage[session_key]['todos']
            for todo in todos:
                if todo['id'] == todo_id and not todo['completed']:
                    todo['completed'] = True
                    return todo['task']
            return None
        else:
            with self.get_db() as conn:
                cursor = conn.execute('SELECT task FROM todos WHERE id=? AND user_id=? AND session_id=? AND completed=0', 
                                    (todo_id, user_id, session_id))
                todo = cursor.fetchone()
                
                if todo:
                    conn.execute('UPDATE todos SET completed=1 WHERE id=? AND user_id=? AND session_id=?', 
                            (todo_id, user_id, session_id))
                    conn.commit()
                    return todo['task']
                return None
        
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
        self.memory = ConversationMemory('/app/data/conversations.db')
        
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

        self.supported_cities = {
            # main italian cities
            'roma': 'Rome,IT',
            'milano': 'Milan,IT', 
            'napoli': 'Naples,IT',
            'torino': 'Turin,IT',
            'palermo': 'Palermo,IT',
            'genova': 'Genoa,IT',
            'bologna': 'Bologna,IT',
            'firenze': 'Florence,IT',
            'bari': 'Bari,IT',
            'catania': 'Catania,IT',
            'venezia': 'Venice,IT',
            'verona': 'Verona,IT',
            'messina': 'Messina,IT',
            'padova': 'Padua,IT',
            'trieste': 'Trieste,IT',
            'brescia': 'Brescia,IT',
            'taranto': 'Taranto,IT',
            'prato': 'Prato,IT',
            'modena': 'Modena,IT',
            'reggio calabria': 'Reggio Calabria,IT',
            'reggio emilia': 'Reggio Emilia,IT',
            'perugia': 'Perugia,IT',
            'livorno': 'Livorno,IT',
            'ravenna': 'Ravenna,IT',
            'cagliari': 'Cagliari,IT',
            'foggia': 'Foggia,IT',
            'rimini': 'Rimini,IT',
            'salerno': 'Salerno,IT',
            'ferrara': 'Ferrara,IT',
            'sassari': 'Sassari,IT',
            'monza': 'Monza,IT',
            'bergamo': 'Bergamo,IT',
            'pescara': 'Pescara,IT',
            'trento': 'Trento,IT',
            'forli': 'Forli,IT',
            'vicenza': 'Vicenza,IT',
            'terni': 'Terni,IT',
            'bolzano': 'Bolzano,IT',
            'novara': 'Novara,IT',
            'ancona': 'Ancona,IT',
            
            # European cities
            'londra': 'London,GB',
            'parigi': 'Paris,FR',
            'berlino': 'Berlin,DE',
            'madrid': 'Madrid,ES',
            'amsterdam': 'Amsterdam,NL',
            'bruxelles': 'Brussels,BE',
            'vienna': 'Vienna,AT',
            'zurigo': 'Zurich,CH',
            'oslo': 'Oslo,NO',
            'stoccolma': 'Stockholm,SE',
            'copenaghen': 'Copenhagen,DK',
            'helsinki': 'Helsinki,FI',
            'dublino': 'Dublin,IE',
            'lisbona': 'Lisbon,PT',
            'atene': 'Athens,GR',
            'varsavia': 'Warsaw,PL',
            'praga': 'Prague,CZ',
            'budapest': 'Budapest,HU',
            'bucarest': 'Bucharest,RO',
            'sofia': 'Sofia,BG',
            
            # Main world cities
            'new york': 'New York,US',
            'los angeles': 'Los Angeles,US',
            'chicago': 'Chicago,US',
            'miami': 'Miami,US',
            'las vegas': 'Las Vegas,US',
            'san francisco': 'San Francisco,US',
            'boston': 'Boston,US',
            'washington': 'Washington,US',
            'seattle': 'Seattle,US',
            'toronto': 'Toronto,CA',
            'vancouver': 'Vancouver,CA',
            'montreal': 'Montreal,CA',
            'tokyo': 'Tokyo,JP',
            'osaka': 'Osaka,JP',
            'seoul': 'Seoul,KR',
            'pechino': 'Beijing,CN',
            'shanghai': 'Shanghai,CN',
            'hong kong': 'Hong Kong,HK',
            'singapore': 'Singapore,SG',
            'bangkok': 'Bangkok,TH',
            'mumbai': 'Mumbai,IN',
            'delhi': 'Delhi,IN',
            'bangalore': 'Bangalore,IN',
            'sydney': 'Sydney,AU',
            'melbourne': 'Melbourne,AU',
            'auckland': 'Auckland,NZ',
            'rio de janeiro': 'Rio de Janeiro,BR',
            'san paolo': 'Sao Paulo,BR',
            'buenos aires': 'Buenos Aires,AR',
            'lima': 'Lima,PE',
            'citta del messico': 'Mexico City,MX',
            'bogota': 'Bogota,CO',
            'santiago': 'Santiago,CL',
            'caracas': 'Caracas,VE',
            'il cairo': 'Cairo,EG',
            'johannesburg': 'Johannesburg,ZA',
            'casablanca': 'Casablanca,MA',
            'lagos': 'Lagos,NG',
            'istanbul': 'Istanbul,TR',
            'mosca': 'Moscow,RU',
            'dubai': 'Dubai,AE',
            'riyadh': 'Riyadh,SA',
            'tel aviv': 'Tel Aviv,IL'
        }

        # Sorted list of the cities for the display
        self.cities_display_list = self._create_cities_display_list()

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

    #to create a sorted list of the cities for the display
    def _create_cities_display_list(self):
        italian_cities = []
        european_cities = []
        world_cities = []
        
        for city_name, api_name in self.supported_cities.items():
            if ',IT' in api_name:
                italian_cities.append(city_name.title())
            elif any(country in api_name for country in [',GB', ',FR', ',DE', ',ES', ',NL', ',BE', ',AT', ',CH', ',NO', ',SE', ',DK', ',FI', ',IE', ',PT', ',GR', ',PL', ',CZ', ',HU', ',RO', ',BG']):
                european_cities.append(city_name.title())
            else:
                world_cities.append(city_name.title())
        
        return {
            'italian': sorted(italian_cities),
            'european': sorted(european_cities),
            'world': sorted(world_cities)
        }

    #to extract the name of the city from the message
    def extract_city_from_message(self, message):
        message_lower = message.lower()
        
        # First look for exact matches (compound cities)
        for city_key in sorted(self.supported_cities.keys(), key=len, reverse=True):
            if city_key in message_lower:
                return city_key, self.supported_cities[city_key]
        
        # If it doesn't find exact matches, look for individual words that could be cities
        words = message_lower.split()
        potential_cities = []

        # List of candidates to be excluded
        excluded_words = {
            'meteo', 'tempo', 'clima', 'previsioni', 'che', 'come', 'oggi', 'domani', 'fare', 
            'bel', 'bello', 'brutto', 'cattivo', 'piove', 'sole', 'nuvoloso', 'sereno',
            'caldo', 'freddo', 'umido', 'secco', 'vento', 'ventoso', 'piovoso',
            'temperatura', 'gradi', 'gradi', 'centigradi', 'celsius', 'fahrenheit',
            'dove', 'quando', 'perché', 'cosa', 'quale', 'quanto', 'chi',
            'il', 'la', 'lo', 'gli', 'le', 'un', 'una', 'del', 'della', 'dello',
            'di', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 'a', 'ad',
            'ma', 'però', 'tuttavia', 'quindi', 'allora', 'poi', 'dopo', 'prima',
            'molto', 'tanto', 'poco', 'abbastanza', 'troppo', 'più', 'meno',
            'fa', 'sarà', 'era', 'è', 'sono', 'hanno', 'hai', 'ho', 'ha'
        }
        
        for word in words:
            if (len(word) > 2 and 
                word not in excluded_words and 
                not word.endswith('?') and 
                not word.endswith('!') and
                word.isalpha()):
                potential_cities.append(word)
        
        # Returns only if there is ONLY ONE candidate word
        if len(potential_cities) == 1:
            return None, potential_cities
        else:
            return None, None

    # to obtain help cities's message
    def get_cities_help_message(self):
        cities = self.cities_display_list
        
        result = "🌍 **Città Disponibili per il Meteo:**\n\n"
        
        result += "🇮🇹 **Italia** (selezione):\n"
        italia_sample = cities['italian'][:15]
        result += f"   {' • '.join(italia_sample)}"
        if len(cities['italian']) > 15:
            result += f" *+{len(cities['italian'])-15} altre*"
        result += "\n\n"
        
        result += "🇪🇺 **Europa** (capitali principali):\n"
        europa_sample = cities['european'][:12]
        result += f"   {' • '.join(europa_sample)}"
        if len(cities['european']) > 12:
            result += f" *+{len(cities['european'])-12} altre*"
        result += "\n\n"
        
        result += "🌎 **Mondo** (selezione):\n"
        world_sample = cities['world'][:10]
        result += f"   {' • '.join(world_sample)}"
        if len(cities['world']) > 10:
            result += f" *+{len(cities['world'])-10} altre*"
        result += "\n\n"
        
        result += "💡 **Esempi di utilizzo:**\n"
        result += "• 'Che tempo fa a Roma?'\n"
        result += "• 'Meteo Milano oggi'\n"
        result += "• 'Previsioni Londra'\n\n"
        result += f"📊 *Totale: {len(self.supported_cities)} città disponibili*"
        
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
    def get_weather_real(self, city_input="Roma"):
        user_session = session.get('session_id', 'anonymous')
        can_request, message, use_fallback = rate_limiter.can_make_request('weather', user_session)
        
        if use_fallback:
            fallback_response = self.get_weather_fallback(city_input)
            return f"{fallback_response}\n\n⚠️ {message}"
        
        if not OPENWEATHER_API_KEY:
            return f"{self.get_weather_fallback(city_input)}\n\n⚠️ API key non configurata - usando dati simulati"
        
        # Determine the city to use for the API
        if city_input.lower() in self.supported_cities:
            city_key = city_input.lower()
            api_city = self.supported_cities[city_key]
            display_city = city_key.title()
        else:
            # Unsupported city
            return f"❌ **Città '{city_input}' non disponibile.**\n\nLe città disponibili sono:\n\n{self.get_cities_help_message()}"
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {'q': api_city, 'appid': OPENWEATHER_API_KEY, 'units': 'metric', 'lang': 'it'}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if response.status_code == 200:
                # increment counter only for a success call
                rate_limiter.increment_counter('weather', user_session)
                
                temp = data['main']['temp']
                feels_like = data['main']['feels_like']
                humidity = data['main']['humidity']
                description = data['weather'][0]['description']
                
                stats = rate_limiter.get_stats()['weather']
                
                return  f"🌤️ **Meteo {display_city}:**\n" \
                        f"🌡️ Temperatura: {temp}°C (percepita {feels_like}°C)\n" \
                        f"☁️ Condizioni: {description.title()}\n" \
                        f"💧 Umidità: {humidity}%\n\n" \
                        f"📊 *API calls rimanenti oggi: {stats['remaining']}/{stats['limit']}*"
            elif response.status_code == 404:
                return f"❌ **L'API OpenWeather non supporta la città '{display_city}'.**\n\nProva con una città simile o scegli dalla lista:\n\n{self.get_cities_help_message()}"
            else:
                return f"{self.get_weather_fallback(display_city)}\n\n⚠️ Errore API (codice {response.status_code}) - usando dati simulati"
                
        except Exception as e:
            print(f"Weather API error: {e}")
            return f"{self.get_weather_fallback(display_city)}\n\n⚠️ Errore connessione API - usando dati simulati"

    #return simulated weather (only if API failed)
    def get_weather_fallback(self, city):
        temps = [18, 19, 20, 21, 22, 23, 24, 25]
        conditions = ["soleggiato", "parzialmente nuvoloso", "nuvoloso"]
        
        return f"🌤️ **Meteo {city}** (simulato):\n" \
               f"🌡️ Temperatura: {random.choice(temps)}°C\n" \
               f"☁️ Condizioni: {random.choice(conditions)}\n" \
               f"💧 Umidità: {random.randint(45, 75)}%"

    # to return real stock prices (API: Alpha Vantage)
    def get_stock_prices_real(self, symbol=None):
        user_session = session.get('session_id', 'anonymous')
        can_request, message, use_fallback = rate_limiter.can_make_request('stocks', user_session)
        
        if use_fallback:
            fallback_response = self.get_stock_prices_fallback(symbol)
            return f"{fallback_response}\n\n⚠️ {message}"
        
        if not ALPHA_VANTAGE_API_KEY:
            return f"{self.get_stock_prices_fallback(symbol)}\n\n⚠️ API key non configurata"
 
        if not symbol:
            return "❓ **Specifica quale azione ti interessa!**\n\n💡 **Esempio:** 'Prezzo azione Apple' o 'Prezzo AAPL'\n\n" + self.get_stocks_help_message()
        
        try:
            symbol = symbol.upper()
            
            url = f"https://www.alphavantage.co/query"
            params = {'function': 'GLOBAL_QUOTE', 'symbol': symbol, 'apikey': ALPHA_VANTAGE_API_KEY}
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'Global Quote' in data and data['Global Quote']:
                # increment counter only for success call
                rate_limiter.increment_counter('stocks', user_session)
                
                quote = data['Global Quote']
                price = float(quote.get('05. price', 0))
                change = float(quote.get('09. change', 0))
                change_pct = float(quote.get('10. change percent', '0').replace('%', ''))
                
                emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                stats = rate_limiter.get_stats()['stocks']
                
                return f"📊 **Prezzo Azione {symbol}:**\n" \
                    f"💰 **{symbol}**: ${price:.2f} {emoji} {change_pct:+.2f}%\n\n" \
                    f"📊 *API calls: {stats['remaining']}/{stats['limit']} (giorno), " \
                    f"{stats.get('minute_remaining', 'N/A')}/{stats.get('minute_limit', 'N/A')} (minuto)*"
            else:
                error_msg = data.get('Note', data.get('Information', 'Errore sconosciuto'))
                return f"{self.get_stock_prices_fallback(symbol)}\n\n⚠️ {error_msg}"
                
        except Exception as e:
            print(f"Stock API exception for {symbol}: {e}")
            return f"{self.get_stock_prices_fallback(symbol)}\n\n⚠️ Errore API - usando dati simulati"

    #to return (randomly) simulated stock prices (only if API failed or stock not available)
    def get_stock_prices_fallback(self, symbol=None):
        
        if not symbol:
            return "❓ **Specifica quale azione ti interessa!**\n\n💡 **Esempio:** 'Prezzo azione Apple' o 'Prezzo AAPL'\n\n" + self.get_stocks_help_message()
        
        symbol = symbol.upper()
        if symbol in self.stock_symbols_display:
            # Generate a simulated price based on a realistic logic: Use the symbol as a seed to always have the same price for the same stock
            random.seed(hash(symbol) % 1000)
            
            # Approximate base prices for some known stocks
            base_prices = {
            # Tech Giants
            'AAPL': 180, 'MSFT': 375, 'GOOGL': 140, 'META': 330, 'AMZN': 150,
            'TSLA': 250, 'NVDA': 500, 'NFLX': 460, 'AMD': 120, 'INTC': 45,
            
            # Finance
            'JPM': 150, 'BAC': 35, 'WFC': 45, 'GS': 380, 'MS': 90, 'V': 245, 'MA': 420, 'PYPL': 65,
            
            # Health/Pharma
            'JNJ': 165, 'PFE': 30, 'MRNA': 80, 'ABT': 110, 'MRK': 105,
            
            # Energy
            'XOM': 110, 'CVX': 160, 'COP': 120,
            
            # Consumer/Retail
            'KO': 60, 'PEP': 170, 'PG': 155, 'NKE': 105, 'WMT': 160, 'HD': 350, 'MCD': 280, 'SBUX': 95, 'DIS': 100,
            
            # Industrial
            'BA': 210, 'CAT': 340, 'MMM': 105, 'GE': 170, 'LMT': 480,
            
            # Telecom
            'VZ': 40, 'T': 20, 'TMUS': 160,
            
            # Altri
            'BRK.B': 440, 'IBM': 190, 'ORCL': 130, 'CRM': 250, 'ZM': 70, 'SHOP': 80, 'UBER': 70, 'ABNB': 140
            }
            
            if symbol in base_prices:
                #use the price in base_prices
                base_price = base_prices[symbol]
            else:
                #generate a simulate (deterministic, thanks to the seed link to the symbol) price
                hash_value = sum(ord(c) for c in symbol)
                base_price = 50 + (hash_value % 450)  # Range 50-500
        
            # Add a random change of ±5%
            variation = random.uniform(-0.05, 0.05)
            price = base_price * (1 + variation)
            
            change_pct = random.uniform(-5, 5)
            change_emoji = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
            
            # Random seed reset
            random.seed()
            return f"📊 **Prezzo Azione {symbol}** (simulato - API non disponibile):\n💰 **{symbol}**: ${price:.2f} {change_emoji} {change_pct:+.2f}%"
        else:
            return f"❌ **Azione '{symbol}' non supportata.**\n\n{self.get_stocks_help_message()}"
           
    
    # to return real news from US (API: NewsAPI)
    def get_news_real(self):
        user_session = session.get('session_id', 'anonymous')
        can_request, message, use_fallback = rate_limiter.can_make_request('news', user_session)
        
        if use_fallback:
            fallback_response = self.get_news_fallback()
            return f"{fallback_response}\n\n⚠️ {message}"
        
        if not NEWS_API_KEY:
            return f"{self.get_news_fallback()}\n\n⚠️ API key non configurata"
        
        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {'country': 'us', 'category': 'general', 'pageSize': 5, 'apiKey': NEWS_API_KEY}
            
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if response.status_code == 200 and data['articles']:
                # Increment onfly for success call
                rate_limiter.increment_counter('news', user_session)
                
                news_text = "📰 **Ultime Notizie dagli Stati Uniti:**\n\n"
                
                for i, article in enumerate(data['articles'][:5], 1):
                    title = article['title']
                    source = article['source']['name']
                    
                    if len(title) > 80:
                        title = title[:77] + "..."
                    
                    news_text += f"**{i}.** {title}\n"
                    news_text += f"   *Fonte: {source}*\n\n"
                
                stats = rate_limiter.get_stats()['news']
                news_text += f"📊 *API calls rimanenti oggi: {stats['remaining']}/{stats['limit']}*"
                
                return news_text
            else:
                return f"{self.get_news_fallback()}\n\n⚠️ Errore nel recupero notizie"
                
        except Exception as e:
            print(f"News API error: {e}")
            return f"{self.get_news_fallback()}\n\n⚠️ Errore API - usando notizie simulate"

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
        """Ora attuale (Italia)"""
        now = datetime.now(ZoneInfo("Europe/Rome"))
        return f"🕐 **Ora attuale:** {now.strftime('%H:%M:%S')}\n" \
               f"📅 **Data:** {now.strftime('%d/%m/%Y')}\n" \
               f"📆 **Giorno:** {now.strftime('%A')}"


    #to do management
    def manage_todo(self, user_id, action, task=None, priority=1):
        session_id = session.get('session_id')
        
        if action == 'add' and task:
            # Extract priority from text if present and set it
            if any(word in task.lower() for word in ['importante', 'urgente', 'priorità']):
                priority = 3
            elif any(word in task.lower() for word in ['bassa', 'quando possibile']):
                priority = 1
            else:
                priority = 2
            
            # Use new ConversationMemory method
            todo_id = self.memory.add_todo(user_id, task, priority, session_id)
            priority_text = {1: "bassa", 2: "media", 3: "alta"}[priority]
            return f"✅ Aggiunto '{task}' con priorità {priority_text}."

        elif action == 'list':
            # Use new ConversationMemory method
            todos = self.memory.get_todos(user_id, session_id)
            
            if not todos:
                return "📝 La tua lista è vuota! 🎉"
            
            result = "📝 **Le tue cose da fare:**\n\n"
            priority_emojis = {1: "🔵", 2: "🟡", 3: "🔴"}
            
            for todo in todos:
                emoji = priority_emojis.get(todo['priority'], "⚪")
                result += f"{emoji} **{todo['id']}:** {todo['task']}\n"
            
            result += "\n💡 *Scrivi 'completa [numero]' per completare un task*"
            return result

        elif action == 'complete' and task:
            try:
                task_id = int(task)
                # Use new ConversationMemory method
                completed_task = self.memory.complete_todo(user_id, task_id, session_id)
                
                if completed_task:
                    return f"🎉 Task completato: '{completed_task}'!"
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
        if confidence < 0.8:
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
                            # The user is likely to have specified an unrecognized action
                            # Extract possible words that could be action symbols
                            words = [word.upper() for word in message.split() if len(word) >= 2 and word.isalpha()]
                            potential_symbols = [word for word in words if word not in ['PREZZO', 'AZIONE', 'AZIONI', 'QUANTO', 'VALE', 'COSTA', 'STOCK']]
                    
                            if potential_symbols:
                                # The user likely specified an unsupported action
                                response = f"❌ **Azione non riconosciuta o non supportata:** {', '.join(potential_symbols)}\n\n💡 **Esempio corretto:** 'Prezzo azione Apple' o 'Prezzo AAPL'\n\n{self.get_stocks_help_message()}"
                            else:
                                # Price request without specific action
                                response = "❓ **Specifica quale azione ti interessa!**\n\n💡 **Esempio:** 'Prezzo azione Apple' o 'Prezzo AAPL'\n\n" + self.get_stocks_help_message()
                        else:
                            # Generic request for stocks - show help
                            response = f"📊 **Servizio Azioni Disponibile!**\n\n{self.get_stocks_help_message()}"
                        
        # CASE 2: weather
        elif intent == 'weather':
            #check if user ask the list of supported cities
            if any(phrase in message.lower() for phrase in ['disponibili', 'lista', 'elenco', 'quali città', 'che città', 'città supportate', 'città']):
                response = f"🌍 **Servizio Meteo Disponibile!**\n\n{self.get_cities_help_message()}"
            else:
                # Extract city from message if present
                city_key, potential_cities = self.extract_city_from_message(message)
                
                if city_key:
                    # City recognised
                    response = self.get_weather_real(city_key)
                elif potential_cities:
                    # Words found but not recognized as cities
                    potential_str = ', '.join(potential_cities)
                    response = f"❓ **Non ho riconosciuto la città: '{potential_str}'**\n\nSpecifica meglio la città o scegli dalla lista disponibile.\n\n💡 **Esempio:** 'Che tempo fa a Roma?'\n\n{self.get_cities_help_message()}"
                else:
                    # No cities specified - use Rome as default
                    default_weather = self.get_weather_real("roma")
                    if '🌤️ **Meteo Roma:**' in default_weather:
                        weather_content = default_weather.split('🌤️ **Meteo Roma:**\n')[1]
                        response = f"🏠 **Da me che sto a Roma, il meteo è questo:**\n\n{weather_content}"
                    else:
                        response = f"🏠 **Da me che sto a Roma, il meteo è questo:**\n\n{default_weather}"
                    
        # CASE 3: news
        elif intent == 'news':
            response = self.get_news_real()
        
        # CASE 4: time
        elif intent == 'time':
            response = self.get_current_time()

        # CASE 5: joke
        elif intent == 'joke':
            responses = self.intents_data.get(intent, {}).get('responses', [])
            if responses:
                response = random.choice(responses)
            else:
                response = "Scusami ma non ne ricordo nessuna! " + self.get_help_message()
        
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
            "🕐 **Orario**: 'Che ore sono?', 'Dimmi l'orario'"
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
    #create a unique ID session
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

# endpoint /chat: used to send messages (only POST) 
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    #return the message
    message = data.get('message', '').strip()
    #return user id
    user_id = session.get('user_id', 'anonymous')
    #return session id
    session_id = session.get('session_id', str(uuid.uuid4()))
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    #process message using AI Chatbot
    response, intent, confidence = assistant.process_message(message, user_id)
    
    assistant.memory.add_conversation(user_id, message, response, intent, session_id)
    
    #return a json response with (response, intent, confidence, timestamp and user id)
    return jsonify({
        'response': response,
        'intent': intent,
        'confidence': round(confidence * 100, 1),
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'session_id': session_id
    })

#endpoint /clear-chat: used to clear the session's chat
@app.route('/clear-chat', methods=['POST'])
def clear_chat_endpoint():
    user_id = session.get('user_id', 'anonymous')
    session_id = session.get('session_id')
    
    # clean only session's chat
    assistant.memory.clear_user_conversations(user_id, session_id)
    return jsonify({
        'status': 'success',
        'message': 'Chat cancellata con successo',
        'timestamp': datetime.now().isoformat()
    })

#endpoint /new-session: used to generate new ID session for a new session (Start a new sesssion)
@app.route('/new-session', methods=['POST'])
def new_session():
    session['user_id'] = str(uuid.uuid4())
    session['session_id'] = str(uuid.uuid4())
    
    return jsonify({
        'status': 'success',
        'user_id': session['user_id'],
        'session_id': session['session_id'],
        'message': 'Nuova sessione iniziata',
        'timestamp': datetime.now().isoformat()
    })


# endpoint /history: used to return last 10 user and session's conversation
@app.route('/history', methods=['GET'])
def history():
    user_id = session.get('user_id', 'anonymous')
    session_id = session.get('session_id')

    #read from DB 
    history = assistant.memory.get_recent_context(user_id, limit=10, session_id=session_id)
    #convert to DICT for the JSON serialization
    return jsonify({'history': [dict(h) for h in history]})


# endpoint /api-stats: used to visualize API stats
@app.route('/api-stats', methods=['GET'])
def api_stats():
    try:
        stats = rate_limiter.get_stats()
        return jsonify({
            'api_usage': stats,
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'status': 'error'
        }), 500


# endpoint /health: used to check service state (check uploaded model, setted API Keys, usage stats)
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
    port = int(os.environ.get('PORT', 7860))
    
    # Detect deployment environment
    deployment_env = os.environ.get('DEPLOYMENT_ENV', 'local')
    
    print(f"🚀 AI Assistant started on port {port}")
    print(f"🔑 Weather API: {'✅' if OPENWEATHER_API_KEY else '❌'}")
    print(f"🔑 News API: {'✅' if NEWS_API_KEY else '❌'}")
    print(f"🔑 Stock API: {'✅' if ALPHA_VANTAGE_API_KEY else '❌'}")
    print(f"🛡️ Rate limiting attivo: Weather (1000/giorno), News (100/giorno), Stocks (25/giorno, 5/minuto)")

    if deployment_env == 'local':
        print(f"📁 Database: File persistente")
        print(f"🧹 Cleanup automatico: Attivo (7 giorni)")
    else:
        print(f"📁 Database: Memoria temporanea")
        print(f"🧹 Cleanup automatico: Disattivo")
    
    app.run(host='0.0.0.0', port=port, debug=False)