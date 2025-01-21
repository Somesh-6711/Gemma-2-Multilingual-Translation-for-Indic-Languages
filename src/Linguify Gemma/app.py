from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
MODEL_DIR = "models/fine_tuned/"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

@app.route("/translate", methods=["POST"])
def translate():
    """
    Real-time translation API endpoint.

    Expects JSON input:
        - text: Text to translate
        - source_lang (optional): Source language (e.g., 'hi', 'gu', 'sa').

    Returns:
        - translated_text: Translated sentence.
    """
    data = request.get_json()
    source_text = data.get("text", "")

    if not source_text:
        return jsonify({"error": "No text provided for translation"}), 400

    # Translate text
    inputs = tokenizer(source_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"translated_text": translated_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
