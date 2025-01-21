from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_metric
import pandas as pd

def evaluate_model(model_dir, test_data_path):
    """
    Evaluate the fine-tuned model using BLEU score.

    Args:
        model_dir (str): Path to the fine-tuned model directory.
        test_data_path (str): Path to the test dataset CSV file.

    Returns:
        float: BLEU score.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    # Load the test dataset
    test_data = pd.read_csv(test_data_path)
    source_sentences = test_data["source_sentence"].tolist()
    target_sentences = test_data["target_sentence"].tolist()

    # Load BLEU metric
    bleu = load_metric("bleu")
    predictions = []
    references = []

    # Generate translations and collect references
    for source, target in zip(source_sentences, target_sentences):
        inputs = tokenizer(source, return_tensors="pt", truncation=True, padding=True)
        outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(translated_text.split())
        references.append([target.split()])

    # Compute BLEU score
    bleu_score = bleu.compute(predictions=predictions, references=references)
    print(f"BLEU Score: {bleu_score['bleu']:.4f}")
    return bleu_score["bleu"]
