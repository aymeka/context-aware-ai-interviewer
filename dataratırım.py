import os
import json
import time
import requests
from tqdm import tqdm
# === Ayarlar === #
API_KEYS = [
 "your keys"
]
if not API_KEYS:
    print("Lütfen API_KEYS listesine Gemini API anahtarlarınızı ekleyin.")
MODEL_NAME = "gemini-1.5-flash"
ENDPOINT_TEMPLATE = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={{}}"
RETRY_LIMIT = 3
WAIT_BETWEEN_REQUESTS = 6
CHUNK_SIZE = 100
INPUT_FILE = "interviews_structured.jsonl"
OUTPUT_DIR = "augmented_data_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_AUGMENTATIONS_PER_INPUT = 3
# === API isteği fonksiyonu === #
def send_to_gemini(prompt, api_key, temperature=0.7):
    endpoint = ENDPOINT_TEMPLATE.format(api_key)
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
        }
    }
    headers = {"Content-Type": "application/json"}
    for attempt in range(RETRY_LIMIT):
        try:
            response = requests.post(endpoint, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                if "candidates" in data and data["candidates"] and \
                   len(data["candidates"]) > 0 and \
                   "content" in data["candidates"][0] and \
                   "parts" in data["candidates"][0]["content"] and \
                   len(data["candidates"][0]["content"]["parts"]) > 0:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    print(f"⚠️ Yanıt formatı beklenenden farklı veya boş candidate: {data}")
                    return ""
            elif response.status_code == 429:
                print(f"⏳ API key limiti ({api_key}) aşıldı. Daha uzun bekleniyor (65s)...")
                time.sleep(65)
            elif response.status_code == 500 or response.status_code == 503:
                wait_time = 30 + attempt * 15
                print(f"🔁 Servis Hatası ({response.status_code}). Bekleniyor ({wait_time}s)...")
                time.sleep(wait_time)
            else:
                print(f"⚠️ API Hatası: {response.status_code} - {response.text}")
                if response.status_code in [400, 401, 403]:
                    print(f"❌ API anahtarı {api_key} ile ciddi bir sorun var, bu anahtar atlanıyor.")
                    return "SKIP_KEY"
        except requests.exceptions.RequestException as e:
            print(f"🚨 Ağ Hatası: {e}")
        except Exception as e:
            print(f"🚨 Genel Hata: {e}")
        if attempt < RETRY_LIMIT - 1:
            time.sleep(WAIT_BETWEEN_REQUESTS + attempt * 5)
    return None
# === Veri yükle === #
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        original_data = [json.loads(line) for line in f]
except FileNotFoundError:
    print(f"HATA: Giriş dosyası '{INPUT_FILE}' bulunamadı.")
    original_data = []
except json.JSONDecodeError:
    print(f"HATA: Giriş dosyası '{INPUT_FILE}' geçerli bir JSONL formatında değil.")
    original_data = []

if not original_data:
    print("Giriş verisi yüklenemedi veya boş. Program sonlandırılıyor.")
    exit()
# === Chunk'lara böl ve sırayla işle === #
total_chunks = len(original_data) // CHUNK_SIZE + (1 if len(original_data) % CHUNK_SIZE else 0)
api_key_index = 0
existing_augmentations = {}
if os.path.exists(OUTPUT_DIR):
    for chunk_file_name in os.listdir(OUTPUT_DIR):
        if chunk_file_name.startswith("augmented_") and chunk_file_name.endswith(".jsonl"):
            chunk_file_path = os.path.join(OUTPUT_DIR, chunk_file_name)
            try:
                with open(chunk_file_path, "r", encoding="utf-8") as f_existing:
                    for line in f_existing:
                        try:
                            entry = json.loads(line)
                            original_input_tuple = tuple(sorted(entry["input"].items()))
                            if original_input_tuple not in existing_augmentations:
                                existing_augmentations[original_input_tuple] = set()
                            existing_augmentations[original_input_tuple].add(entry["output"])
                        except (json.JSONDecodeError, KeyError, TypeError) as e:
                            print(f"Uyarı: {chunk_file_path} dosyasında hatalı satır atlanıyor: {line.strip()} - Hata: {e}")
            except Exception as e:
                 print(f"Uyarı: {chunk_file_path} dosyası okunurken hata: {e}")
for chunk_index in range(total_chunks):
    current_chunk_data = original_data[chunk_index * CHUNK_SIZE:(chunk_index + 1) * CHUNK_SIZE]
    output_file_path = os.path.join(OUTPUT_DIR, f"augmented_{chunk_index+1}.jsonl")
    print(f"\n🚀 Chunk {chunk_index+1}/{total_chunks} işleniyor ({len(current_chunk_data)} orijinal örnek)...")
    with open(output_file_path, "a", encoding="utf-8") as out_file: # 'a' (append) modu
        for item_index, item in enumerate(tqdm(current_chunk_data, desc="→ Üretiliyor")):
            original_input_tuple = tuple(sorted(item["input"].items())) # Mevcut item için anahtar
            current_augment_count = len(existing_augmentations.get(original_input_tuple, set()))
            if current_augment_count >= MAX_AUGMENTATIONS_PER_INPUT:
                continue
            attempts_for_new_questions = MAX_AUGMENTATIONS_PER_INPUT - current_augment_count
            prompt_template = (
                f"Sen yaratıcı bir röportaj uzmanısın. Daha önce sorulmuş bir soruya verilen cevabı, "
                f"görüşülen kişiyi, mesleğini ve o anki duygu durumunu dikkate alarak, "
                f"bu bağlama uygun, {{count}} adet YENİ ve FARKLI röportaj sorusu üret. "
                f"Sorular açık uçlu, düşündürücü ve çeşitli olmalı. "
                f"Sadece üretilen soruları numaralandırarak veya her birini yeni bir satırda listele. Başka hiçbir açıklama yapma.\n\n"
                f"Referans Alınacak Bilgiler:\n"
                f"Daha Önceki Soru: {item['input']['Röportaj sorusu']}\n"
                f"Verilen Cevap: {item['input']['Cevap']}\n"
                f"Görüşülen Kişi: {item['input']['Kişi']}\n"
                f"Mesleği: {item['input']['Meslek']}\n"
                f"Duygu Durumu: {item['input']['Duygu']}\n\n"
                f"{{count}} adet yeni soru:"
            )
            num_questions_to_request = 3
            final_prompt = prompt_template.format(count=num_questions_to_request)
            if not API_KEYS:
                print("⛔ API anahtarı kalmadı. Bu ve sonraki örnekler atlanıyor.")
                break
            active_api_key = API_KEYS[api_key_index % len(API_KEYS)]
            current_temperature = 0.7
            response_text = send_to_gemini(final_prompt, active_api_key, temperature=current_temperature)
            api_key_index += 1
            if response_text == "SKIP_KEY":
                try:
                    API_KEYS.remove(active_api_key)
                    print(f"🔑 API anahtarı {active_api_key} listeden çıkarıldı.")
                    api_key_index = 0
                    if not API_KEYS:
                        print("⛔ Tüm API anahtarları tüketildi veya sorunlu.")
                        break
                except ValueError:
                    pass
                continue
            if response_text is None or not response_text.strip():
                print(f"⛔ API'den geçerli yanıt alınamadı. '{item['input']['Röportaj sorusu'][:50]}...' atlanıyor.")
                time.sleep(WAIT_BETWEEN_REQUESTS)
                continue
            generated_questions = []
            for line in response_text.strip().split('\n'):
                clean_q = line.strip()
                if clean_q.startswith(tuple(str(i) + "." for i in range(10))):
                    clean_q = clean_q.split(".", 1)[-1].strip()
                elif clean_q.startswith(tuple(str(i) + ")" for i in range(10))):
                    clean_q = clean_q.split(")", 1)[-1].strip()
                elif clean_q.startswith(("- ", "* ", "• ")):
                    clean_q = clean_q[2:].strip()
                if clean_q and len(clean_q) > 10:
                    generated_questions.append(clean_q)
            newly_added_count_for_this_input = 0
            for q_idx, new_q in enumerate(generated_questions):
                if current_augment_count + newly_added_count_for_this_input >= MAX_AUGMENTATIONS_PER_INPUT:
                    break
                if new_q not in existing_augmentations.get(original_input_tuple, set()):
                    out_item = {
                        "input": item["input"],
                        "output": new_q
                    }
                    out_file.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                    if original_input_tuple not in existing_augmentations:
                        existing_augmentations[original_input_tuple] = set()
                    existing_augmentations[original_input_tuple].add(new_q)
                    newly_added_count_for_this_input += 1
            if newly_added_count_for_this_input > 0:
                out_file.flush()
            time.sleep(WAIT_BETWEEN_REQUESTS)
    if not API_KEYS:
        print("Tüm API anahtarları tükendiği için işlem sonlandırılıyor.")
        break
print("\n✅ Tüm chunklar işlendi ve tamamlandı.")