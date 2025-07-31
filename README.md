# Context-Aware Dynamic Interview Assistant

## Overview
This project focuses on building a **web-based system** that generates **context-aware follow-up questions** during interview processes. Using **Natural Language Processing (NLP)** techniques, the system analyzes a user's response along with their profession, emotional state, and current news context to generate meaningful and creative follow-up questions.

The system is designed to enhance the **flow of interviews**, providing **journalists and content creators** with a more interactive and human-like experience.

---

## Features
- **Context-Aware Question Generation:** Generates follow-up questions based on the interviewee's response, profession, emotional state, and related news headlines.
- **Transformer-Based Model:** Uses **FLAN-T5** fine-tuned with **LoRA** for efficient training.
- **Paraphrasing for Fluency:** Integrates a **BERT2BERT paraphrase model** to improve the naturalness of generated questions.
- **Data Augmentation:** Uses Gemini API to create additional samples and enhance dataset diversity.
- **Web Interface:** A simple **Flask-based web application** for user interaction.
- **Real-Time News Integration:** Fetches relevant headlines from **NewsAPI** to add contextual depth to the questions.

---

## System Architecture
- **Flask Framework** – Provides the web interface and API backend.
- **FLAN-T5 (LoRA Fine-tuning)** – Generates follow-up questions.
- **BERT2BERT Paraphrase Model** – Refines the fluency of generated questions.
- **Gemini 1.5 Flash API** – Used for data augmentation and creating diverse examples.
- **NewsAPI** – Fetches up-to-date news headlines relevant to the interviewee.

---

## Dataset
- **Original Dataset:** Consists of transcribed interviews with well-known Turkish public figures (e.g., Sezen Aksu, Cem Yılmaz, Tarkan).
- **Manual Labeling:** Each entry contains:
  - Interview question
  - Interviewee's response
  - Name, profession, emotional state
  - A single follow-up question
- **Data Augmentation:** Each sample is expanded using Gemini API, increasing dataset size to 4000+ follow-up questions.
- **Format:** Stored in **JSONL** format, where each line represents an input-output pair.

---

## Model Training
- **Base Model:** `google/flan-t5-base`
- **Fine-Tuning Method:** LoRA (Low-Rank Adaptation)
- **Hyperparameters:**
  - Learning Rate: 3e-4
  - Epochs: 10
  - Max Input Length: 512
  - Max Output Length: 256
  - Beam Search: 4
  - Early Stopping: Enabled
- **Evaluation Metrics:** BLEU, ROUGE-1/2/L. Human evaluation showed that most generated questions are **logical and contextually appropriate**.

---

## How It Works
1. **User Inputs:** Name, gender, profession, emotional state, and response to an interview question.
2. **Context Retrieval:** The system fetches a relevant news headline via NewsAPI.
3. **Question Generation:** FLAN-T5 generates a context-aware follow-up question.
4. **Refinement:** BERT2BERT paraphrasing ensures fluency and naturalness.
5. **Output:** The final follow-up question is displayed to the user via a simple web interface.

---

## Installation
### Prerequisites
- Python 3.9+
- `transformers`
- `peft`
- `flask`
- `newsapi-python`
- `torch`

### Steps
```bash
# Clone the repository
git clone https://github.com/your-repo/context-aware-interview-assistant.git
cd context-aware-interview-assistant

# Install dependencies
pip install -r requirements.txt

# Set your NewsAPI key in .env file
NEWSAPI_KEY=your_api_key_here

# Run the application
python app.py
```

The application will run at `http://127.0.0.1:5000/`.

---

## Usage
1. Open the web interface.
2. Enter **Name**, **Profession**, **Emotional State**, and **Response**.
3. Click **Generate Question**.
4. The system will output a **context-aware follow-up question**.

---

## Results
- The model successfully generates **contextual and meaningful follow-up questions**, especially for neutral and positive emotional states.
- Data augmentation improved diversity and generalization of the model.
- Human evaluation indicates that generated questions are generally logical and well-structured.

---

## Future Improvements
- Different strategies for short and long responses.
- A more diverse dataset with various professions and emotions.
- Integration of knowledge-based sources (e.g., Wikipedia) for deeper question generation.
- Advanced evaluation metrics (e.g., BERTScore, BLEURT).

---

## References
- Akyön, F. Ç., Gür, N., & Yıldırım, Ö. (2022). Automated question generation and answering from Turkish texts. [arXiv:2208.00896](https://arxiv.org/abs/2208.00896)
- Meng, Y., Pan, L., Chen, Z., & Yan, X. (2023). FOLLOWUPQG: Towards Information-Seeking Follow-up Question Generation. [arXiv:2305.10007](https://arxiv.org/abs/2305.10007)
- [FLAN-T5 Model Card](https://huggingface.co/google/flan-t5-base)
- [BERT2BERT Paraphrase Model](https://huggingface.co/ahmetbagci/bert2bert-turkish-paraphrase-generation)
- [NewsAPI Documentation](https://newsapi.org/docs)
