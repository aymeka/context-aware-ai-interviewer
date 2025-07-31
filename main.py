from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel, BertTokenizerFast
import torch
import requests

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# === 1. MODELLERİ YÜKLE === #
T5_MODEL_PATH = "your path"
t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_PATH, local_files_only=True)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_PATH, local_files_only=True).to(device)

pp_tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-cased")
pp_model = EncoderDecoderModel.from_pretrained("ahmetbagci/bert2bert-turkish-paraphrase-generation").to(device)

# === 2. NewsAPI Ayarı === #
NEWSAPI_KEY = "your keys"

def get_latest_news(person_name):
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "qInTitle": person_name,
            "language": "tr",
            "sortBy": "publishedAt",
            "apiKey": NEWSAPI_KEY,
            "pageSize": 1
        }
        response = requests.get(url, params=params)
        data = response.json()
        if data.get("status") == "ok" and data["articles"]:
            return data["articles"][0]["title"]
    except Exception as e:
        print("NewsAPI hatası:", e)
    return None

# === 3. Rephrase === #
def refine_turkish_sentence(text):
    input_ids = pp_tokenizer(text, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
    output_ids = pp_model.generate(input_ids, max_new_tokens=128, do_sample=False)
    return pp_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

# === 4. Follow-up Soru Üret === #
def generate_followup_question(input_text, min_len=30, max_len=256, refine=True):
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = t5_model.generate(
        input_ids=inputs,
        min_length=min_len,
        max_new_tokens=max_len,
        num_beams=4,
        early_stopping=True
    )
    raw_output = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if refine:
        try:
            return refine_turkish_sentence(raw_output)
        except Exception as e:
            print("Refine hatası:", e)
            return raw_output
    else:
        return raw_output

# === 5. Ana Sayfa === #
@app.route("/")
def home():
    return render_template("index.html")

# === 6. Sohbet API === #
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    message = data.get("message", "")
    person = data.get("person", "Misafir")
    gender = data.get("gender", "diğer").lower()
    role = data.get("role", "Röportaj Yapılan Kişi")
    mood = data.get("mood", "Nötr")
    state = data.get("state", "initial")

    # Hitap belirle
    if gender == "erkek":
        salutation = f"{person} Bey"
    elif gender == "kadın":
        salutation = f"{person} Hanım"
    else:
        salutation = person

    if state == "initial":
        return jsonify({
            "reply": f"Merhaba {salutation}, röportaj sistemine hoş geldiniz! Bugün nasılsınız?",
            "state": "asked_wellbeing"
        })

    elif state == "asked_wellbeing":
        news = get_latest_news(person)
        if news:
            return jsonify({
                "reply": f"Teşekkür ederim. Bu arada sizinle ilgili dikkat çeken bir haber gördüm:\n\"{news}\"\nBu konuda ne düşünüyorsunuz?",
                "state": "asked_about_news"
            })
        else:
            return jsonify({
                "reply": f"Güncel bir haber bulamadım ama genel bir konudan başlayalım: Yakın zamanda sizi etkileyen bir olay oldu mu?",
                "state": "asked_about_news"
            })

    elif state == "asked_about_news":
        model_input = (
            f"generate follow-up question: Röportaj sorusu: Yakın zamanda sizi etkileyen bir olay oldu mu? "
            f"| Cevap: {message} | Kişi: {person} | Meslek: {role} | Duygu: {mood}"
        )
        question = generate_followup_question(model_input)
        return jsonify({"reply": question, "state": "followup"})

    elif state == "followup":
        model_input = (
            f"generate follow-up question: Röportaj sorusu: Önceki cevabınıza istinaden "
            f"| Cevap: {message} | Kişi: {person} | Meslek: {role} | Duygu: {mood}"
        )
        question = generate_followup_question(model_input)
        return jsonify({"reply": question})

    return jsonify({"reply": "Bir hata oluştu."})

if __name__ == "__main__":
    app.run(debug=True)