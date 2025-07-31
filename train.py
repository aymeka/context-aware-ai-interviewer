import os
import json
import datetime
import torch
import evaluate
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Trainer, TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from tqdm import tqdm
import numpy as np
import gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' tokenizer zaten mevcut.")
except LookupError:
    print("NLTK 'punkt' tokenizer bulunamadı, indiriliyor...")
    nltk.download('punkt', quiet=False)
    try:
        nltk.data.find('tokenizers/punkt')
        print("NLTK 'punkt' tokenizer başarıyla indirildi ve bulundu.")
    except LookupError:
        print("HATA: NLTK 'punkt' tokenizer indirilemedi veya bulunamadı. BLEU metrikleri çalışmayabilir.")

# === 1. Ayarlar === #
base_model = "google/flan-t5-base"
dataset_path = "interviews_structured_t5.jsonl"
model_dir = f"output/model_{base_model.replace('/', '_')}"
merged_model_dir = f"output/merged_{base_model.replace('/', '_')}"
results_dir = f"output/results_{base_model.replace('/', '_')}"
# Log dizini, her çalıştırmada benzersiz olacak şekilde ayarlandı.
log_dir = f"output/logs_{base_model.replace('/', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(merged_model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
if not os.path.exists(dataset_path):
    print(f"'{dataset_path}' bulunamadı. Örnek bir dosya oluşturuluyor...")
    sample_data_new_format = [
                                 {
                                     "input": "generate follow-up question: Röportaj sorusu: Yurt dışında büyük ilgi gördün. Bekliyor muydun? Cevap: Hayır, beklemiyordum. Çok şaşırdım. Kişi: Hadise | Meslek: Müzisyen | Duygu: Şaşkın",
                                     "output": "Bu beklenmedik başarının ardından, müzikal yolculuğunuzda öncelikleriniz nasıl değişti?"},
                                 {
                                     "input": "generate follow-up question: Röportaj sorusu: Yeni tarzın çok konuşuldu. Bilerek mi yaptın? Cevap: Değişim iyidir ama bu kadar konuşulacağını sanmazdım. Kişi: Hadise | Meslek: Müzisyen | Duygu: Şaşkın",
                                     "output": "Bu beklenmedik tepki karşısında, yaratıcılığınızın sınırlarını zorlamak adına gelecekte nasıl riskler alacaksınız?"}
                             ] * 330  # Yaklaşık 5000 satır için örnek veri sayısını biraz daha artırdım
    with open(dataset_path, 'w', encoding='utf-8') as f:
        for item in sample_data_new_format:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Örnek '{dataset_path}' dosyası oluşturuldu.")
# === 2. Dataset yükleme ve temizleme === #
def load_jsonl_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f.readlines()]
raw_list = load_jsonl_dataset(dataset_path)
clean_list = []
for ex_idx, ex in enumerate(raw_list):
    if ex.get("input") and isinstance(ex.get("input"), str) and \
            ex.get("output") and isinstance(ex.get("output"), str):
        if ex["input"].strip() and ex["output"].strip():
            clean_list.append(ex)
        else:
            print(f"Uyarı: {ex_idx}. indeksteki örnekte 'input' veya 'output' içeriği boş, atlanıyor.")
    else:
        print(
            f"Uyarı: {ex_idx}. indeksteki örnekte 'input' (string olmalı) veya 'output' (string olmalı) eksik/hatalı, atlanıyor.")
if not clean_list:
    raise ValueError("Temizlenmiş veri listesi boş. Lütfen veri dosyanızı kontrol edin.")
train_list, temp_list = train_test_split(clean_list, test_size=0.2, random_state=42)
val_list, test_list = train_test_split(temp_list, test_size=0.5, random_state=42)
train_data = Dataset.from_list(train_list)
val_data = Dataset.from_list(val_list)
test_data = Dataset.from_list(test_list)
# === 3. Tokenizer === #
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# === 4. Prompt ve tokenize === #
TASK_PREFIX = ""
def tokenize_t5(example):
    input_str = example.get("input")
    output_str = example.get("output")
    if not isinstance(input_str, str) or not input_str.strip():
        print(f"Hata Tespit Edildi (Boş/Hatalı Input): Örnek: {example}. Atlanıyor.")
        return {"input_ids": [], "attention_mask": [], "labels": []}
    if not isinstance(output_str, str) or not output_str.strip():
        print(f"Hata Tespit Edildi (Boş/Hatalı Output): Örnek: {example}. Atlanıyor.")
        return {"input_ids": [], "attention_mask": [], "labels": []}
    input_text = TASK_PREFIX + input_str
    target_text = output_str
    model_inputs = tokenizer(input_text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    with tokenizer.as_target_tokenizer():  # T5 için etiket tokenizasyonu farklı yapılır
        labels = tokenizer(target_text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
    if model_inputs["input_ids"].numel() == 0:
        print(f"Uyarı: Tokenizasyon sonrası Input ID'leri boş. Orijinal Input: '{input_text}'. Örnek atlanıyor.")
        return {"input_ids": [], "attention_mask": [], "labels": []}
    if labels["input_ids"].numel() == 0:
        print(f"Uyarı: Tokenizasyon sonrası Labels ID'leri boş. Orijinal Output: '{target_text}'. Örnek atlanıyor.")
        return {"input_ids": [], "attention_mask": [], "labels": []}
    model_inputs["labels"] = labels["input_ids"]
    return {k: v.squeeze() for k, v in model_inputs.items()}
train_tokenized = train_data.map(tokenize_t5, remove_columns=train_data.column_names)
val_tokenized = val_data.map(tokenize_t5, remove_columns=val_data.column_names)
test_tokenized_inputs_for_generation = test_data.map(
    lambda x: {"source_text": TASK_PREFIX + x["input"], "target_text": x["output"]},
    remove_columns=test_data.column_names
)
train_tokenized = train_tokenized.filter(lambda example: len(example['input_ids']) > 0)
val_tokenized = val_tokenized.filter(lambda example: len(example['input_ids']) > 0)
if len(train_tokenized) == 0 or len(val_tokenized) == 0:
    raise ValueError(
        "Tokenizasyon sonrası eğitim veya doğrulama verisi kalmadı. Lütfen veri ve tokenizasyon adımlarını kontrol edin.")
print(f"Tokenizasyon sonrası eğitim örneği sayısı: {len(train_tokenized)}")
print(f"Tokenizasyon sonrası doğrulama örneği sayısı: {len(val_tokenized)}")
print(f"Tokenizasyon sonrası test örneği sayısı (üretim için): {len(test_tokenized_inputs_for_generation)}")

# === 5. Model + LoRA === #

model = AutoModelForSeq2SeqLM.from_pretrained(base_model, device_map="auto")
lora_target_modules = ["q", "v", "k", "o"]  # 'k' ve 'o' eklendi
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=lora_target_modules,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
from transformers import TrainerCallback
class EpochEndLogCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        print(f"\n--- Özel Log: Dönem {int(state.epoch)} Sonu ---")
        if logs:
            print(f"  Dönem sonunda gelen loglar: {logs}")
        relevant_log = None
        for log_entry in reversed(state.log_history):
            if 'eval_loss' in log_entry and 'epoch' in log_entry and abs(log_entry['epoch'] - state.epoch) < 0.1:
                relevant_log = log_entry
                break
        if relevant_log:
            print(
                f"  Log Geçmişinden - Dönem {relevant_log['epoch']:.2f}: Doğrulama Kaybı: {relevant_log.get('eval_loss', 'N/A')}, Eğitim Kaybı (yaklaşık): {relevant_log.get('loss', 'N/A')}")
        print("--- Özel Log Sonu ---")
# === 6. Eğitim argümanları === #
training_args = TrainingArguments(
    output_dir=model_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    learning_rate=3e-4,
    save_total_limit=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir=log_dir,
    report_to="none",
    fp16=False,
    bf16=True,
)
# === 7. Trainer === #
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None
)
bleu_metric_loader = evaluate.load("bleu")
rouge_metric_loader = evaluate.load("rouge")
def compute_metrics_seq2seq(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.argmax(preds, axis=-1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    rouge_result = rouge_metric_loader.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_result = bleu_metric_loader.compute(predictions=decoded_preds,
                                             references=[[label] for label in decoded_labels])
    result = {
        "bleu": bleu_result["bleu"],
        **rouge_result
    }
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[
        EpochEndLogCallback(),
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)
    ],
    compute_metrics=compute_metrics_seq2seq,
)
print("Eğitim başlıyor...")
trainer.train()
# === 8. Model Kaydet === #
print("Model kaydediliyor...")
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
# === 9. LoRA Adaptörlerini Birleştir === #
print("LoRA adaptörleri temel modele birleştiriliyor...")
base_model_for_merge = AutoModelForSeq2SeqLM.from_pretrained(
    base_model,
    device_map="auto"
)
merged_model = PeftModel.from_pretrained(base_model_for_merge, model_dir)
merged_model = merged_model.merge_and_unload()
print("Birleştirilmiş model kaydediliyor...")
merged_model.save_pretrained(merged_model_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_model_dir)
# === 10. Doğrulama kontrolü (NaN testi) === #
print("📊 Eğitilmiş adaptörlerle (birleştirmeden önce) Doğrulama Seti sonuçları:")
eval_result = trainer.evaluate(eval_dataset=val_tokenized)
print(eval_result)
# === 11. Test seti değerlendirmesi === #
print("Test seti üzerinde nihai değerlendirme yapılıyor...")
# Birleştirilmiş modeli yükleyin
# GPU'da çalışmak için device_map="auto" olarak ayarlandı
gen_model = AutoModelForSeq2SeqLM.from_pretrained(merged_model_dir, device_map="auto")
preds_text, labels_text = [], []
for sample in tqdm(test_tokenized_inputs_for_generation, desc="Test Seti Tahminleri"):
    input_text = sample["source_text"]
    label_text = sample["target_text"]
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
        gen_model.device)
    output_sequences = gen_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        min_length=30,
        max_new_tokens=256,
        num_beams=4,  #Daha yüksek değerler daha kaliteli ama yavaş üretim sağlar.
        early_stopping=True,
    )
    prediction = tokenizer.decode(output_sequences[0], skip_special_tokens=True).strip()
    preds_text.append(prediction)
    labels_text.append(label_text.strip())
    del inputs, output_sequences
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
bleu_scores = bleu_metric_loader.compute(
    predictions=preds_text,
    references=[[label] for label in labels_text]  # BLEU için referanslar liste içinde liste olmalı
)
rouge_scores = rouge_metric_loader.compute(
    predictions=preds_text,
    references=labels_text
)
print(f"Test Seti BLEU: {bleu_scores['bleu']:.4f}")
print(f"Test Seti ROUGE-1: {rouge_scores['rouge1']:.4f}")
print(f"Test Seti ROUGE-L: {rouge_scores['rougeL']:.4f}")
with open(os.path.join(results_dir, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"BLEU: {bleu_scores['bleu']:.4f}\n")
    f.write(f"ROUGE-1: {rouge_scores['rouge1']:.4f}\n")
    f.write(f"ROUGE-2: {rouge_scores['rouge2']:.4f}\n")
    f.write(f"ROUGE-L: {rouge_scores['rougeL']:.4f}\n")
    f.write(f"ROUGE-Lsum: {rouge_scores['rougeLsum']:.4f}\n")
print("📊 Test metrikleri hesaplandı ve kaydedildi.")
# === 12. Loss grafiği === #
if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
    log_history = trainer.state.log_history
    train_steps, train_loss, eval_loss, eval_steps_from_log = [], [], [], []
    for entry in log_history:
        if "loss" in entry and "step" in entry and 'epoch' in entry:
            is_eval_log = any(key.startswith("eval_") for key in entry.keys())
            if not is_eval_log:
                train_steps.append(entry["step"])
                train_loss.append(entry["loss"])
        if "eval_loss" in entry and "step" in entry:
            eval_steps_from_log.append(entry["step"])
            eval_loss.append(entry["eval_loss"])
    if train_steps and train_loss:
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_loss, label="Eğitim Kaybı", marker='o', linestyle='-')
        if eval_steps_from_log and eval_loss:
            plt.plot(eval_steps_from_log, eval_loss, label="Doğrulama Kaybı", marker='x', linestyle='--')
        plt.xlabel("Adım (Step)")
        plt.ylabel("Kayıp (Loss)")
        plt.title(f"Eğitim ve Doğrulama Kaybı ({base_model})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "train_val_loss_curve.png"))
        plt.close()
        print("📈 Eğitim ve doğrulama kayıpları grafiği kaydedildi.")
        with open(os.path.join(results_dir, "loss_values.json"), "w", encoding="utf-8") as f:
            json.dump({
                "train_loss_steps": list(zip(train_steps, train_loss)),
                "eval_loss_steps": list(zip(eval_steps_from_log, eval_loss)),
            }, f, indent=2, ensure_ascii=False)
        print("📈 Kayıp değerleri JSON olarak kaydedildi.")
    else:
        print("❌ Eğitim veya doğrulama kayıp değerleri log geçmişinde bulunamadı. Grafik çizilemedi.")
else:
    print("❌ Trainer state veya log geçmişi bulunamadı. Kayıp grafiği çizilemiyor.")
print("\n✅ Eğitim, değerlendirme ve görselleştirme başarıyla tamamlandı.")
