# Gemma-2-Multilingual-Translation-for-Indic-Languages
# Linguify Gemma

## Overview
**Linguify Gemma** is a multilingual translation project aimed at fine-tuning the Gemma 2 model to translate Hindi, Gujarati, and Sanskrit texts into English. By leveraging a custom dataset with parallel sentence pairs, this project enhances model accuracy and demonstrates its application in real-time translation through an API.

---

## Features

1. **Fine-Tuned Translation**:
   - Translates Hindi, Gujarati, and Sanskrit texts into English with high accuracy.

2. **Custom Dataset**:
   - Utilizes datasets of parallel sentence pairs to improve translation quality.
   - Datasets include:
     - Hindi-English
     - Gujarati-English
     - Sanskrit-English

3. **Real-Time API**:
   - A Flask-based API for real-time translations.
   - Simple input-output design for ease of use.

4. **Deployed on Heroku**:
   - The API is publicly accessible online.

---

## Project Structure
```plaintext
Linguify-Gemma/
├── data/
│   ├── raw/                     # Original datasets (Hindi, Gujarati, Sanskrit)
│   ├── processed/               # Preprocessed datasets
│   ├── tokenized/               # Tokenized datasets
├── src/
│   ├── linguify_gemma/
│   │   ├── components/          # Core project components
│   │   │   ├── data_preparation.py
│   │   │   ├── model_fine_tuning.py
│   │   ├── utils/               # Utility scripts
│   │   │   ├── data_loader.py
│   │   ├── config/              # Configuration files
│   │   │   ├── config.yaml
│   ├── api/                     # API for real-time translation
│   │   ├── app.py
├── research/                    # Jupyter notebooks for experimentation
│   ├── trials.ipynb
├── requirements.txt             # Python dependencies
├── Procfile                     # For Heroku deployment
├── runtime.txt                  # Python runtime version for Heroku
├── README.md                    # Project documentation
```

---

## Dataset Details
The project uses custom parallel datasets stored in `data/raw/`:

1. **Hindi-English Dataset**:
   - Columns: `source_sentence` (English), `target_sentence` (Hindi)
2. **Gujarati-English Dataset**:
   - Columns: `source_sentence` (English), `target_sentence` (Gujarati)
3. **Sanskrit-English Dataset**:
   - Columns: `source_sentence` (English), `target_sentence` (Sanskrit)

---

## Setup and Installation

### Prerequisites
- Python 3.9 or higher
- pip
- Heroku CLI (for deployment)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/linguify-gemma.git
   cd linguify-gemma
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Preprocess the datasets:
   ```bash
   python src/linguify_gemma/components/data_preparation.py
   ```

4. Tokenize the datasets:
   ```bash
   python src/linguify_gemma/utils/data_loader.py
   ```

5. Fine-tune the model:
   ```bash
   python src/linguify_gemma/components/model_fine_tuning.py
   ```

6. Run the API locally:
   ```bash
   python src/linguify_gemma/api/app.py
   ```
   Access the API at `http://127.0.0.1:5000/translate`.

---

## Deployment
The API is deployed on Heroku.

### Deployment Steps
1. Login to Heroku:
   ```bash
   heroku login
   ```

2. Create a new Heroku app:
   ```bash
   heroku create linguify-gemma
   ```

3. Deploy the app:
   ```bash
   git push heroku master
   ```

4. Access the API:
   ```
   https://linguify-gemma.herokuapp.com/translate
   ```

---

## Usage
### Example API Call
Make a POST request to the `/translate` endpoint with JSON input:

#### Request:
```json
{
  "text": "This is a test sentence."
}
```

#### Response:
```json
{
  "translated_text": "यह एक परीक्षण वाक्य है।"
}
```

---

## Future Enhancements
1. **Multilingual Back Translation**:
   - Support translation from Hindi, Gujarati, and Sanskrit to English and vice versa.

2. **Grammar Correction**:
   - Enhance output translations with grammar correction tools.

3. **Frontend Integration**:
   - Build a user-friendly web interface for easier interaction.

4. **Usage Analytics**:
   - Track API usage and performance metrics.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments
- **Hugging Face**: For the Transformers library.
- **Heroku**: For hosting the API.

---
