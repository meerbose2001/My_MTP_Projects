import joblib
import json

# Load artifacts
vectorizer = joblib.load("artifacts/vectorizer.joblib")
model = joblib.load("artifacts/model.joblib")
with open("artifacts/label_map.json") as f:
    label_map = json.load(f)

# Invert the label map so we can map 0/1 back to names
inv_label_map = {v: k for k, v in label_map.items()}

# 20 test samples
samples = [
    # Realistic
    "Government announces new tax reforms to boost small businesses.",
    "NASA confirms successful launch of new satellite for climate monitoring.",
    "Local schools to reopen next Monday following health department guidelines.",
    "Stock markets rally as tech companies report strong quarterly earnings.",
    "WHO warns of rising cases of seasonal flu across multiple countries.",
    "India and Japan sign trade agreement to strengthen regional cooperation.",
    "Doctors recommend regular exercise to reduce risk of heart disease.",
    "UN to hold emergency meeting on global food security crisis.",
    "Apple unveils latest iPhone model with upgraded camera and battery life.",
    "Scientists discover new species of frog in the Amazon rainforest.",
    # Clearly fake
    "Aliens land in New York City demanding pizza before peace talks.",
    "Time traveler from 3024 warns humanity about robot uprising.",
    "Scientists confirm water on Mars tastes like Coca-Cola.",
    "Hollywood actor secretly crowned king of a European country.",
    "Ancient pyramid in Egypt found to contain working WiFi router.",
    "Breaking: Cats declared official rulers of the United Nations.",
    "Man claims to survive only on sunlight and air for 10 years.",
    "New law allows teleportation for all citizens starting next week.",
    "Secret government base discovered under the Eiffel Tower.",
    "Researchers announce successful cloning of dinosaurs for new theme park.",
]

# Transform and predict
X = vectorizer.transform(samples)
preds = model.predict(X)

# Print results
for text, pred in zip(samples, preds):
    print(f"Text: {text}")
    print(f"Prediction: {inv_label_map[pred]} ({pred})")
    print("-" * 80)

