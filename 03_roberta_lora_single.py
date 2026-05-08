# ============================================================
# HÜCRE 1 — Google Drive bağla
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================
# HÜCRE 2 — Proje klasör yapısını oluştur
# ============================================================
import os

# Ana klasör — Drive'da
DRIVE_BASE = "/content/drive/MyDrive/imdb_peft_project"

DIRS = {
    "root":         DRIVE_BASE,
    "checkpoints":  f"{DRIVE_BASE}/checkpoints/roberta_lora",
    "checkpoints2": f"{DRIVE_BASE}/checkpoints/deberta_lora",
    "oof":          f"{DRIVE_BASE}/oof_predictions",
    "results":      f"{DRIVE_BASE}/results",
    "notebooks":    f"{DRIVE_BASE}/notebooks",
    "code":         f"{DRIVE_BASE}/code",  # .py dosyaları buraya
}

for path in DIRS.values():
    os.makedirs(path, exist_ok=True)
    print(f"✓ {path}")

print("\nKlasör yapısı hazır.")

# ============================================================
# HÜCRE 3 — GitHub repo clone veya pull
# ============================================================
# İLK KEZ KULLANIMDA: repo'yu clone'la
# SONRAKI KULLANIMLARDA: sadece pull yap

import subprocess

GITHUB_USERNAME = "DenizhanOngun"   # değiştir
GITHUB_REPO     = "lora-imdb-classifier"      # değiştir — repo adın
GITHUB_EMAIL    = "denizhan.eser@hotmail.com"            # değiştir
import getpass
GITHUB_TOKEN = getpass.getpass("GitHub Token'ını gir: ")
REPO_URL  = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
REPO_PATH = f"{DRIVE_BASE}/code/{GITHUB_REPO}"

# Git kimlik ayarları
!git config --global user.email "{GITHUB_EMAIL}"
!git config --global user.name  "{GITHUB_USERNAME}"

if not os.path.exists(REPO_PATH):
    print("Repo bulunamadı, clone'lanıyor...")
    !git clone {REPO_URL} {REPO_PATH}
    print("✓ Clone tamamlandı.")
else:
    print("Repo mevcut, güncelleniyor...")
    !cd {REPO_PATH} && git pull
    print("✓ Pull tamamlandı.")

print(f"\nRepo konumu: {REPO_PATH}")

# ============================================================
# HÜCRE 4 — GitHub'a push fonksiyonu
# (her önemli adımdan sonra çağır)
# ============================================================
def push_to_github(commit_message: str):
    """
    Repo klasöründeki tüm değişiklikleri GitHub'a push'lar.
    Kullanım: push_to_github("fold 2 tamamlandi")
    """
    import subprocess

    commands = [
        f"cd {REPO_PATH} && git add .",
        f"cd {REPO_PATH} && git commit -m '{commit_message}'",
        f"cd {REPO_PATH} && git push",
    ]

    for cmd in commands:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            # Commit edilecek değişiklik yoksa sessizce geç
            if "nothing to commit" in result.stdout:
                print("ℹ Değişiklik yok, push atlandı.")
                return
            print(f"⚠ Hata: {result.stderr}")
            return

    print(f"✓ GitHub'a push edildi: '{commit_message}'")

# ============================================================
# HÜCRE 5 — Kod dosyasını Drive'daki repo'ya kopyala
# (her yeni .py dosyası oluşturduktan sonra çağır)
# ============================================================
import shutil

def save_code_to_repo(source_path: str, filename: str = None):
    """
    Bir .py dosyasını Drive'daki repo klasörüne kopyalar.
    Kullanım: save_code_to_repo("/content/01_data_preprocessing.py")
    """
    if filename is None:
        filename = os.path.basename(source_path)

    dest_path = os.path.join(REPO_PATH, filename)
    shutil.copy2(source_path, dest_path)
    print(f"✓ {filename} → repo'ya kopyalandı.")

# ============================================================
# HÜCRE 6 — Drive kayıt yardımcı fonksiyonları
# (model ağırlıkları ve OOF için)
# ============================================================
import numpy as np
import json

def save_oof(predictions: np.ndarray, model_name: str, fold: int):
    """OOF tahminlerini Drive'a kaydeder."""
    path = f"{DIRS['oof']}/{model_name}_fold{fold}.npy"
    np.save(path, predictions)
    print(f"✓ OOF kaydedildi: {path}")

def load_oof(model_name: str, n_folds: int = 5) -> np.ndarray:
    """Kaydedilmiş OOF tahminlerini yükler ve birleştirir."""
    all_preds = []
    for fold in range(n_folds):
        path = f"{DIRS['oof']}/{model_name}_fold{fold}.npy"
        if os.path.exists(path):
            all_preds.append(np.load(path))
            print(f"✓ Yüklendi: fold {fold}")
        else:
            print(f"⚠ Bulunamadı: fold {fold} — eğitim gerekiyor.")
    return np.concatenate(all_preds) if all_preds else None

def save_results(metrics: dict, filename: str = "results.json"):
    """Metrik sonuçlarını Drive'a kaydeder."""
    path = f"{DIRS['results']}/{filename}"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Sonuçlar kaydedildi: {path}")

# ============================================================
# HÜCRE 7 — Kurulum özeti
# ============================================================
print("=" * 50)
print("KURULUM TAMAMLANDI")
print("=" * 50)
print(f"Drive klasörü : {DRIVE_BASE}")
print(f"GitHub repo   : {REPO_PATH}")
print()
print("Kullanılabilir fonksiyonlar:")
print("  push_to_github('mesaj')         → kodu GitHub'a gönder")
print("  save_code_to_repo('dosya.py')   → .py dosyasını repo'ya kopyala")
print("  save_oof(preds, 'roberta', 0)   → OOF tahminlerini kaydet")
print("  load_oof('roberta')             → OOF tahminlerini yükle")
print("  save_results(metrics)           → sonuçları kaydet")

# ============================================================
# TIPIK KULLANIM AKIŞI
# ============================================================
# 1. Bu dosyayı çalıştır (Drive bağlan, repo hazırla)
# 2. Diğer hücrelerde çalış
# 3. Önemli adımlardan sonra:
#      save_code_to_repo("/content/01_data_preprocessing.py")
#      push_to_github("veri on isleme tamamlandi")
# 4. Her fold bittikten sonra:
#      save_oof(preds, "roberta", fold_no)
# 5. Colab kapanırsa:
#      load_oof("roberta") ile kaldığın yerden devam et

!pip install torchao --upgrade -q

!pip install transformers datasets peft accelerate -q

import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
import re, time

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# GPU control
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {device}")

train_df = pd.read_parquet(f"{DIRS['root']}/train_df.parquet")
test_df  = pd.read_parquet(f"{DIRS['root']}/test_df.parquet")

print(f"Train : {len(train_df)} example")
print(f"Test  : {len(test_df)} example")

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def head_tail_truncate(text, tokenizer, max_len=512, head_len=128):
    tail_len = max_len - head_len
    tokens = tokenizer(text, add_special_tokens=False,
                       truncation=False, return_tensors=None)
    input_ids      = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    if len(input_ids) > max_len - 2:
        input_ids      = input_ids[:head_len] + input_ids[-tail_len:]
        attention_mask = attention_mask[:head_len] + attention_mask[-tail_len:]

    result = tokenizer(
        tokenizer.decode(input_ids),
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors=None
    )
    return result

class IMDBDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512, head_len=128):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.head_len  = head_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text  = self.df.loc[idx, "text_clean"]
        label = self.df.loc[idx, "label"]
        encoding = head_tail_truncate(text, self.tokenizer,
                                      self.max_len, self.head_len)
        return {
            "input_ids":      torch.tensor(encoding["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(label,                      dtype=torch.long),
        }

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
train_dataset = IMDBDataset(train_df, tokenizer)
test_dataset  = IMDBDataset(test_df,  tokenizer)

print(f"✓ Tokenizer ready.")
print(f"✓ Train dataset: {len(train_dataset)} example")
print(f"✓ Test dataset: {len(test_dataset)} example")

# Load RoBERTa model
print("Loading RoBERTa...")
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,                        # rank — düşük = az parametre
    lora_alpha=32,               # scaling faktörü
    lora_dropout=0.1,
    target_modules=["query", "value"],  # attention katmanları
    bias="none"
)

# Add LoRA to the model
model = get_peft_model(model, lora_config)

# How many parameters are being trained?
model.print_trainable_parameters()

from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds)
    }

training_args = TrainingArguments(
    output_dir=f"{DIRS['checkpoints']}/roberta_lora_single",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    seed=SEED,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("Training starts...")
start = time.time()
trainer.train()
train_time = time.time() - start
print(f"\n✓ Training completed. Duration: {train_time/60:.1f} minutes")
