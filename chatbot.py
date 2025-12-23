# chatbot.py
from sentence_transformers import SentenceTransformer, util
import torch

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# FAQ data as dictionary
faq_data = {
    # Fees & Payment
    "What is the hostel fee?": "The hostel fees are 250,000 NPR per month. Fees must be paid at the beginning of each semester.",
    "Are there any additional charges?": "Yes, electricity, water, and maintenance charges may apply depending on usage.",
    "Can I pay the hostel fee in installments?": "No, the hostel fee must be paid in full at the start of the semester.",
    "Is there a late fee for delayed payment?": "Yes, a late fee of 5% per week is charged for overdue payments.",
    "Can I get a refund if I leave the hostel mid-semester?": "Partial refunds are not allowed. Fees are non-refundable.",
    "How do I pay the hostel fee?": "You can pay via bank transfer, online payment portal, or at the hostel office.",

    # Rules & Regulations
    "What are the hostel rules?": "No loud music, maintain cleanliness, follow curfew timings, respect others, and no smoking or alcohol.",
    "What is the curfew time?": "Curfew is at 10 PM on weekdays and 11 PM on weekends.",
    "Are guests allowed in the hostel?": "No, external guests are not allowed in the rooms.",
    "Can I cook in my room?": "No, cooking is not allowed in rooms. Use the common mess area.",
    "Are students allowed to leave campus at night?": "Yes, but you must inform the warden and follow curfew rules.",
    "What if I break hostel rules?": "Violations may lead to warnings, fines, or expulsion depending on the severity.",
    "Is smoking allowed?": "No, smoking is strictly prohibited inside the hostel.",
    "Can I throw parties in the hostel?": "No parties or loud gatherings are allowed without prior permission.",

    # Facilities
    "What facilities are available?": "The hostel offers Wi-Fi, gym, study rooms, laundry, common kitchen, and recreation area.",
    "Is Wi-Fi available in the rooms?": "Yes, Wi-Fi is available 24/7 in all rooms and common areas.",
    "Are laundry facilities available?": "Yes, washing machines and dryers are available in the laundry area.",
    "Is there a common kitchen?": "Yes, a common kitchen is available for residents to prepare snacks or tea.",
    "Are there study rooms?": "Yes, quiet study rooms are available for students.",
    "Are there sports facilities?": "Yes, sports facilities like table tennis and a small gym are available.",
    "Are water coolers available?": "Yes, drinking water coolers are installed on each floor.",
    "Is there a recreation room?": "Yes, board games and TV are available in the recreation room.",

    # Room & Accommodation
    "How many students share a room?": "Typically 2–4 students share a room depending on the room type.",
    "Can I choose my roommate?": "Yes, you can request a roommate, subject to availability.",
    "What types of rooms are available?": "Single, double, and triple occupancy rooms are available.",
    "Are beds and furniture provided?": "Yes, each room comes with a bed, study table, chair, and wardrobe.",
    "Can I change my room?": "Room changes are allowed only if there is availability and approval from the warden.",
    "Do rooms have attached bathrooms?": "Some rooms have attached bathrooms, others share common bathrooms.",
    "Is hot water available?": "Yes, hot water is available in the bathrooms from 6 AM to 10 PM.",
    "Can I decorate my room?": "Yes, minor decorations are allowed but no permanent changes.",

    # Safety & Security
    "Is the hostel secure?": "Yes, 24/7 security personnel and CCTV surveillance are in place.",
    "What if there is an emergency?": "Contact the hostel warden or security immediately. Emergency numbers are displayed in each block.",
    "Are fire safety measures available?": "Yes, fire extinguishers, alarms, and evacuation plans are provided.",
    "Can I lock my room?": "Yes, each room has a lockable door. Residents are responsible for their keys.",
    "Is there a security check at entry?": "Yes, all visitors must register at the security desk.",

    # Contact & Administration
    "How can I contact the admin?": "Email admin@college.edu.np or call 01-1234567 during office hours.",
    "Who is the hostel warden?": "The current hostel warden is Mr. Ram Sharma, available for assistance.",
    "How can I report maintenance issues?": "Submit a request to the warden or use the maintenance logbook in the hostel office.",
    "What are the office hours of the hostel admin?": "The hostel office is open from 9 AM to 5 PM, Monday to Friday.",
    "Is there a WhatsApp group for notices?": "Yes, all residents are added to the official hostel WhatsApp group.",

    # Mess & Food
    "Is there a hostel mess?": "Yes, a hostel mess provides breakfast, lunch, and dinner.",
    "Can I opt out of the mess?": "No, all residents are required to take meals from the mess.",
    "Are vegetarian meals available?": "Yes, both vegetarian and non-vegetarian options are available.",
    "What if I have dietary restrictions?": "Inform the warden or mess manager for special arrangements.",
    "What are the mess timings?": "Breakfast: 7–9 AM, Lunch: 12–2 PM, Dinner: 7–9 PM.",
    "Can I bring my own food?": "Residents may store snacks in their rooms but not cook outside the mess.",

    # Leave & Absence
    "Can I leave the hostel for the weekend?": "Yes, prior approval from the warden is required.",
    "Do I need to inform if I’m leaving for a trip?": "Yes, you must inform the warden with leave dates.",
    "Are leave forms required?": "Yes, a leave form must be submitted for any overnight absence.",
    "Can I stay outside during semester breaks?": "Yes, but rooms may be temporarily reallocated during maintenance.",

    # Hygiene & Cleanliness
    "Who cleans the hostel rooms?": "Residents are responsible for keeping their rooms clean. Common areas are cleaned by staff.",
    "Are cleaning supplies provided?": "Basic cleaning supplies are available in common areas.",
    "How often are bathrooms cleaned?": "Bathrooms are cleaned daily by the hostel staff.",
    "What if I find a hygiene issue?": "Report the issue to the warden immediately for action.",

    # Miscellaneous
    "Are there recreational activities in the hostel?": "Yes, movie nights, sports events, and cultural programs are organized occasionally.",
    "Can I bring my pet?": "No pets are allowed in the hostel.",
    "Is there parking available?": "Yes, limited parking is available for residents with prior approval.",
    "Are there medical facilities?": "Yes, a first-aid kit is available and nearby hospitals are accessible.",
    "Can I use the gym?": "Yes, the gym is open for all residents with proper ID.",
    "Can I request a repair in my room?": "Yes, report the issue to the warden who will schedule maintenance.",
    "Is there a library in the hostel?": "Yes, a small hostel library is available for residents.",
    "Can I host a study group?": "Yes, you may use the common room with permission from the warden.",
    "Is there an internet issue helpline?": "Yes, contact the hostel IT support via email or phone for connectivity problems.",
    "Can I extend my stay during vacations?": "Extensions are possible if space is available, approval from the warden is required."
}

# Preprocess text
def preprocess(text):
    return text.lower().strip()

# Prepare question list and embeddings
faq_questions = [preprocess(q) for q in faq_data.keys()]
faq_answers = list(faq_data.values())
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

# Function to get response
def get_response(user_input):
    user_input = preprocess(user_input)
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, faq_embeddings)
    best_idx = torch.argmax(cosine_scores)
    
    # Threshold for similarity
    if cosine_scores[0][best_idx] < 0.3:
        return "Sorry, I don't understand your question."
    
    return faq_answers[best_idx]
