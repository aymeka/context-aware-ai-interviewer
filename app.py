from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel, BertTokenizerFast
import torch
import requests
import google.generativeai as genai

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Google Gemini Ayarı === #
genai.configure(api_key="your keys")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def grammar_correct_gemini_followup(original_question, user_answer, followup_question):
    prompt = (
        f"Aşağıda bir röportaj sorusu, verilen bir cevap ve bu cevap üzerinden üretilmiş bir takip sorusu bulunmaktadır. "
        f"Lütfen sadece takip sorusunun Türkçe gramerini düzelt, anlatım bozukluklarını gider.\n\n"
        f"🔹 Röportaj Sorusu: {original_question}\n"
        f"🔹 Cevap: {user_answer}\n"
        f"🔹 Takip Sorusu (düzenlenecek): {followup_question}\n\n"
        f"✅ Düzenlenmiş Takip Sorusu:"
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini düzeltme hatası:", e)
        return followup_question

# === MODELLER === #
T5_MODEL_PATH = "your path"
t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_PATH, local_files_only=True)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_PATH, local_files_only=True).to(device)

pp_tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-cased")
pp_model = EncoderDecoderModel.from_pretrained("ahmetbagci/bert2bert-turkish-paraphrase-generation").to(device)

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

# === Paraphraser === #
def refine_turkish_sentence(text):
    input_ids = pp_tokenizer(text, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
    output_ids = pp_model.generate(input_ids, max_new_tokens=128, do_sample=False)
    return pp_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

# === Soru Üretim ve Düzeltme === #
def generate_followup_question(input_text, original_question, user_answer, min_len=30, max_len=256, refine=True):
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
            raw_output = refine_turkish_sentence(raw_output)
        except Exception as e:
            print("Refine hatası:", e)

    print("\n🤖 Model çıktısı:", raw_output)

    final_output = grammar_correct_gemini_followup(original_question, user_answer, raw_output)

    print("🎯 Düzeltilmiş (Gemini):", final_output)
    return final_output

# === Ana Sayfa === #
@app.route("/")
def home():
    return render_template("index.html")

# === Sohbet API === #
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    message = data.get("message", "")
    person = data.get("person", "Misafir")
    gender = data.get("gender", "diğer").lower()
    role = data.get("role", "Röportaj Yapılan Kişi")
    mood = data.get("mood", "Nötr")
    state = data.get("state", "initial")

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
            question = "Bu konuda ne düşünüyorsunuz?"
            return jsonify({
                "reply": f"Teşekkür ederim. Bu arada sizinle ilgili dikkat çeken bir haber gördüm:\n\"{news}\"\n{question}",
                "state": "asked_about_news",
                "last_question": question
            })
        else:
            question = "Yakın zamanda sizi etkileyen bir olay oldu mu?"
            return jsonify({
                "reply": question,
                "state": "asked_about_news",
                "last_question": question
            })

    elif state == "asked_about_news":
        original_question = data.get("last_question", "Yakın zamanda sizi etkileyen bir olay oldu mu?")
        model_input = (
            f"generate follow-up question: Röportaj sorusu: {original_question} "
            f"| Cevap: {message} | Kişi: {person} | Meslek: {role} | Duygu: {mood}"
        )
        question = generate_followup_question(model_input, original_question, message)
        return jsonify({"reply": question, "state": "followup", "last_question": question})

    elif state == "followup":
        original_question = data.get("last_question", "Önceki cevabınıza istinaden")
        model_input = (
            f"generate follow-up question: Röportaj sorusu: {original_question} "
            f"| Cevap: {message} | Kişi: {person} | Meslek: {role} | Duygu: {mood}"
        )
        question = generate_followup_question(model_input, original_question, message)
        return jsonify({"reply": question, "state": "followup", "last_question": question})

    return jsonify({"reply": "Bir hata oluştu."})

# === Sunucuyu Başlat === #
if __name__ == "__main__":
    app.run(debug=True)
