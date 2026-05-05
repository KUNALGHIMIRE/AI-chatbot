# 🤖 Hostel Support AI Assistant

The Hostel Support AI Assistant is a real-time AI-powered chatbot designed to help users interact with an intelligent support system for hostel-related queries.

Built using Python and Flask, the project demonstrates practical implementation of:

- Natural Language Processing (NLP) for understanding and processing user queries  
- REST API development for seamless communication between frontend and backend  
- Frontend–backend integration for a smooth interactive user experience  

The chatbot processes user input and generates intelligent, human-like responses through a simple web-based interface.

## 🔗 Live Demo

👉 Try the application here: https://ai-chatbot-8-vnjd.onrender.com  

### 🔐 Demo Credentials
- **Username:** user1  
- **Password:** password123  
 ## 📌 Project Overview

The Hostel Support AI Assistant was developed as a hands-on AI and web development project focused on real-world application of machine learning and backend systems.

- Designed and implemented a Flask-based backend to handle chat requests efficiently  
- Built a responsive web interface for real-time user interaction  
- Integrated a custom-trained NLP model to generate intelligent responses  
- Followed a modular, scalable, and production-friendly project structure  

This project demonstrates practical experience in AI, NLP, and full-stack web development and is well-suited for AI/ML or Software Engineering internships.

## 🏗️ System Design

### 📌 Architecture Overview

The AI Chatbot is built using a simple yet efficient **client-server architecture** integrated with an NLP processing layer.

The system follows a **lightweight monolithic design**, where all components are tightly integrated to ensure fast and real-time response generation.

Key characteristics:
- Client-server communication model for handling user requests  
- Integrated NLP pipeline for processing and understanding user input  
- Real-time response generation for smooth conversational experience  
- Monolithic structure for simplicity and faster development cycles  

## 🔄 System Flow

The chatbot follows a real-time request–response pipeline:

1. User enters a message in the chatbot interface  
2. Frontend sends the input to Flask backend via an HTTP request  
3. Flask receives the request and forwards it to the chatbot logic layer  
4. NLP engine processes the input and generates an appropriate response  
5. Generated response is returned to the Flask server  
6. Flask sends the response back to the frontend in JSON format  
7. The UI updates dynamically to display the chatbot reply in real-time  

**API DESIGN**

POST /chat

Request:
{
  "message": "Hello"
}

Response:
{
  "reply": "Hi! How can I help you?"
}

## 🧠 NLP Pipeline

The chatbot uses a structured NLP pipeline to process and respond to user input:

1. **Text Preprocessing**
   - Tokenization of input text  
   - Cleaning (removal of noise, punctuation, stopwords)

2. **Feature Extraction**
   - TF-IDF vectorization or embedding-based representation  

3. **Intent Detection / Matching**
   - Identifies user intent using similarity matching or classification  

4. **Response Generation**
   - Produces the most relevant response based on detected intent  

## ⭐ Key Features

The system is designed to provide a smooth and intelligent user experience:

- Real-time chatbot interaction for instant responses  
- NLP-based response generation for human-like conversations  
- RESTful API backend for efficient communication  
- Simple and responsive UI for better user experience across devices  

## 🧰 Technologies Used

The project is built using modern tools for backend development and natural language processing:

- **Programming Language:** Python 3.x  
- **Backend Framework:** Flask for API development and request handling  
- **Frontend:** HTML, CSS for user interface design  
- **AI / NLP Libraries:**  
  - NLTK for text processing  
  - SpaCy for advanced NLP tasks  
  - Transformers for modern AI-based language understanding
  - 
**📁 Project Structure**

```AI-Chatbot/
├── static/
│   └── style.css          # Styling for chatbot interface
├── templates/
│   └── index.html         # Frontend UI
├── chatbot.py             # Flask app and chatbot logic
├── Screenshot.png         # Application output screenshot
└── README.md              # Project documentation
```


## 🚀 Key Design Highlights

The system is built with simplicity and real-time performance in mind:

- Monolithic Flask architecture for fast development and deployment  
- REST API communication for structured and scalable data exchange  
- Integrated NLP pipeline for intelligent response generation  
- Real-time interaction for smooth conversational user experience

## 📸 Output

<p align="center">
  <img src="https://github.com/user-attachments/assets/2b339d61-9a5b-4e0f-9a07-39fbdac1ce75" width="450" alt="Chatbot Output Screenshot" />
</p>
