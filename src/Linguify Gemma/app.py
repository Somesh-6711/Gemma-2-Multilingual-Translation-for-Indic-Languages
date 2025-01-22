from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize the Flask app
app = Flask(__name__)

# Load the fine-tuned model and tokenizer
MODEL_DIR = "models/fine_tuned/"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

@app.route("/translate", methods=["POST"])
def translate():
    """
    Translate text from English to Hindi, Gujarati, or Sanskrit.
    Input:
      - text: Source text to translate.
      - source_lang: Optional source language (e.g., 'en').
    Output:
      - Translated text.
    """
    data = request.get_json()
    source_text = data.get("text", "")
    if not source_text:
        return jsonify({"error": "No text provided for translation"}), 400

    # Tokenize and translate
    inputs = tokenizer(source_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"translated_text": translated_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
