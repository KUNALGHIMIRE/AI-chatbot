# app.py
from flask import Flask, render_template, request, jsonify
from chatbot import get_response  # this should now work


app = Flask(__name__)
app.secret_key = "secret"

# Simple login credentials
USER_CREDENTIALS = {"user1": "password123"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    if USER_CREDENTIALS.get(username) == password:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Invalid credentials"})

@app.route("/get_response", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    response = get_response(user_input)  # get response from chatbot.py
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
