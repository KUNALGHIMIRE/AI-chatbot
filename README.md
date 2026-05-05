**AI Chatbot**

A real-time AI-powered chatbot developed using Python and Flask.
This project demonstrates practical skills in Natural Language Processing (NLP), REST API development, and frontend–backend integration.
The chatbot processes user input and generates intelligent, human-like responses through a simple web interface.

**Live Demo:[username:user1 and password:password123]**
 https://ai-chatbot-8-vnjd.onrender.com
 
**Project Overview**

Designed and implemented a Flask-based backend for handling chat requests

Built a responsive web interface for real-time user interaction

Integrated a custom-trained NLP model for generating responses

Followed a modular and clean project structure suitable for real-world deployment

This project was developed as part of hands-on learning in AI and web-based systems and is suitable for AI / ML / Software Engineering internships.


**System Design**

Architecture Overview

The AI Chatbot follows a simple client-server architecture with an integrated NLP processing layer.
This system is designed as a lightweight monolithic AI application with an integrated NLP pipeline for real-time response generation.


**System Flow**

1. User types a message in the chatbot UI
2. Frontend sends the input to Flask backend via HTTP request
3. Flask processes the request and forwards it to chatbot logic
4. NLP engine analyzes input and generates a response
5. Response is returned to Flask server
6. Flask sends JSON response back to frontend
7. UI dynamically updates with chatbot reply in real-time

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

**NLP PIPELINE**

1. Text preprocessing (tokenization, cleaning)
2. Feature extraction (TF-IDF / embeddings)
3. Intent detection / similarity matching
4. Response generation

**Key Features**

- Real-time chatbot interaction  
- NLP-based response generation  
- RESTful API backend  
- Simple and responsive UI

**Technologies Used**

- Programming Language: Python 3.x
- Backend Framework: Flask
- Frontend: HTML, CSS
- AI / NLP: NLTK, SpaCy, Transformers

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


**Key Design Highlights**

- Monolithic Flask architecture  
- REST API communication  
- Integrated NLP pipeline  
- Real-time interaction

**OUTPUT**

<img width="351" height="487" alt="image" src="https://github.com/user-attachments/assets/2b339d61-9a5b-4e0f-9a07-39fbdac1ce75" />

