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

**📁 Project Structure**
```
AI-Chatbot/
├── static/
│   └── style.css          # Styling for chatbot interface
├── templates/
│   └── index.html         # Frontend UI
├── chatbot.py             # Flask app and chatbot logic
├── Screenshot.png         # Application output screenshot
└── README.md              # Project documentation
**Technologies Used**
- Programming Language: Python 3.x
- Backend Framework: Flask
- Frontend: HTML, CSS
- AI / NLP: NLTK, SpaCy, Transformers

**System Design**

Architecture Overview

The AI Chatbot follows a simple client-server architecture with an integrated NLP processing layer.

<img width="951" height="331" alt="image" src="https://github.com/user-attachments/assets/37ba5fee-0442-46f1-99d3-ca37b5d6e469" />


**System Flow**

1. User types a message in the chatbot UI
2. Frontend sends the input to Flask backend via HTTP request
3. Flask processes the request and forwards it to chatbot logic
4. NLP engine analyzes input and generates a response
5. Response is returned to Flask server
6. Flask sends JSON response back to frontend
7. UI dynamically updates with chatbot reply in real-time

**Key Design Highlights**

Lightweight monolithic Flask architecture (easy to deploy)

RESTful communication between frontend and backend

Modular chatbot logic for easy AI model upgrades

Real-time request-response interaction

Designed for scalability (can integrate advanced LLMs later)\

**OUTPUT**

<img width="351" height="487" alt="image" src="https://github.com/user-attachments/assets/2b339d61-9a5b-4e0f-9a07-39fbdac1ce75" />

