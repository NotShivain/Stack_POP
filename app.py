from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import preprocessor as p
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

model_path = "notshivain1/distilbert-base-uncased-stackoverflow-prediction-V2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

id2label = {0: 'LQ_CLOSE', 1: 'HQ', 2: 'LQ_EDIT'}

def cleanse(concat_text):
    concat_text = concat_text.lower()
    concat_text = re.sub(r'[^(a-zA-Z)\s]', '', concat_text)
    concat_text = [word for word in concat_text.split() if word not in stopwords.words('english')]
    sentence = []
    lemmatizer = WordNetLemmatizer()
    for word in concat_text:
        word = p.clean(word)
        sentence.append(lemmatizer.lemmatize(word))
    return ' '.join(sentence)

def predict_quality(title, body):
    concat_text = title + " " + body
    cleaned_text = cleanse(concat_text)
    input_tokens = tokenizer(cleaned_text, truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"])
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probs, dim=-1).item()
    return id2label[predicted_label], probs.tolist()[0]


app = Flask(__name__)

@app.route("/predict", methods=["GET","POST"])
def predict():
    data = request.json
    title = data.get("title")
    body = data.get("body")

    if not title or not body:
        return jsonify({"error": "Both 'title' and 'body' are required."}), 400

    label, probabilities = predict_quality(title, body)

    return jsonify({
        "predicted_label": label,
        "probabilities": {
            "LQ_CLOSE": probabilities[0],
            "HQ": probabilities[1],
            "LQ_EDIT": probabilities[2]
        }
    })

if __name__ == "__main__":
    app.run(debug=True)