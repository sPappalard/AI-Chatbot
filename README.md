---
title: AI Chatbot Assistant
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# AI Chatbot Assistant

An intelligent AI assistant based on **DistilBERT** with a custom web interface.

## ✨ Features

- 🌤️ **Weather**: Real-time weather forecasts with OpenWeatherMap API
- 📰 **News**: Latest news through NewsAPI  
- 📊 **Stocks**: Real-time stock prices with Alpha Vantage API
- ₿ **Crypto**: Cryptocurrency prices via CoinGecko
- 📝 **Todo List**: Personal task management with SQLite database
- 😄 **Entertainment**: Jokes and natural conversation
- 🕒 **Utilities**: Current time and general information

## 🧠 Architecture

- **Backend**: Flask + fine-tuned DistilBERT
- **Frontend**: Custom HTML/CSS/JavaScript
- **ML Model**: DistilBERT with 3-layer classifier
- **Database**: SQLite for conversations and todo list
- **APIs**: Smart rate limiting for external services

## 🎯 Supported Intents

The model is trained on 12 main intents:
- `greeting`, `goodbye`, `thanks`, `help`
- `weather`, `news`, `stocks`, `time`, `joke`
- `todo`, `name`, `mood`

## 🚀 Technologies

- **PyTorch** + **Transformers** for ML
- **Flask** for the REST API
- **NLTK** for pre-processing
- **SQLite** for data persistence
- **Docker** for deployment
