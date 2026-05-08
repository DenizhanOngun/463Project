!pip install transformers datasets peft accelerate -q

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

 #============================================================
# HÜCRE 1 — Import's
# NOT: Önce 00_colab_setup.py needs run before that
# ============================================================
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import RobertaTokenizer, DebertaV2Tokenizer
from torch.utils.data import Dataset
import torch

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
print("✓ Imports are okay .")

# ============================================================
# Cell 2 — Load Dataset
# ============================================================
dataset = load_dataset("imdb")

train_df = pd.DataFrame(dataset["train"])
test_df  = pd.DataFrame(dataset["test"])

print(f"Train : {len(train_df)} example")
print(f"Test  : {len(test_df)} example")
print(f"\nDistribution Label:\n{train_df['label'].value_counts()}")
# 0 = negative, 1 = positive

# ============================================================
# HÜCRE 3 — EDA:Length of the comments
# ============================================================
train_df["word_count"] = train_df["text"].apply(lambda x: len(x.split()))

print(f"Mean word : {train_df['word_count'].mean():.0f}")
print(f"Median word   : {train_df['word_count'].median():.0f}")
print(f"Max word     : {train_df['word_count'].max()}")
print(f"512+ word     : {(train_df['word_count'] > 512).sum()} "
      f"({(train_df['word_count'] > 512).mean()*100:.1f}%)")

plt.figure(figsize=(10, 4))
plt.hist(train_df["word_count"], bins=50, color="steelblue", edgecolor="white")
plt.axvline(512, color="red", linestyle="--", label="512 token limit")
plt.xlabel("Number of Word")
plt.ylabel("Frequency")
plt.title("IMDB Review Length Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(f"{DIRS['results']}/length_distribution.png", dpi=150)
plt.show()
print("✓ The graphic has been saved to Drive.")

# ============================================================
# HÜCRE 4 — Clean Up
# ============================================================
def clean_text(text: str, lowercase: bool = False) -> str:
    """
     IMDB-specific cleanup.

    lowercase=False  → For RoBERTa / DeBERTa
                       Capital letters convey emotion.:
                       "TERRIBLE" != "terrible"
                       Transformer already knows this, don't lose.

    lowercase=True   → For TF-IDF + SVM baseline
                       It does not benefit from the classic ML font size.
    """
    text = re.sub(r"<[^>]+>", " ", text)  # HTML tag temizle
    text = re.sub(r"\s+", " ", text)       # Fazla boşluk temizle
    text = text.strip()
    if lowercase:
        text = text.lower()
    return text

# For Transformer models (default: lowercase=False)
train_df["text_clean"]       = train_df["text"].apply(clean_text)
test_df["text_clean"]        = test_df["text"].apply(clean_text)

# For TF-IDF + SVM baseline
train_df["text_clean_lower"] = train_df["text"].apply(
    lambda x: clean_text(x, lowercase=True)
)
test_df["text_clean_lower"]  = test_df["text"].apply(
    lambda x: clean_text(x, lowercase=True)
)

# Check — find an example with capital letters to see the difference.
sample_idx = train_df["text"].str.contains(r"[A-Z]{3,}").idxmax()
print("Original :", train_df["text"].iloc[sample_idx][:200])
print("\nFor Transformer  :", train_df["text_clean"].iloc[sample_idx][:200])
print("\nFor SVM           :", train_df["text_clean_lower"].iloc[sample_idx][:200])

# HÜCRE 5 —  Head+Tail truncation function
def head_tail_truncate(text: str, tokenizer, max_len: int = 512,
                       head_len: int = 128) -> dict:
    tail_len = max_len - head_len

    tokens = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_tensors=None
    )
    input_ids      = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    if len(input_ids) > max_len - 2:  # 2 = [CLS] + [SEP] için
        input_ids      = input_ids[:head_len] + input_ids[-tail_len:]
        attention_mask = attention_mask[:head_len] + attention_mask[-tail_len:]

    # Encode directly instead of prepare_for_model
    result = tokenizer(
        tokenizer.decode(input_ids),
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors=None
    )
    return result

# ===========================================================
# CELL 6 — Load Tokenizers
# ============================================================
print("Loading Tokenizers...")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
deberta_tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
print("✓ RoBERTa tokenizer ready.")
print("✓ DeBERTa tokenizer ready.")

# ============================================================
# HÜCRE 7 — PyTorch Dataset sınıfı
# ============================================================
class IMDBDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer,
                 max_len: int = 512, head_len: int = 128,
                 text_col: str = "text_clean"):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.head_len  = head_len
        self.text_col  = text_col  # for transformer "text_clean"
                                   # for SVM  "text_clean_lower"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text  = self.df.loc[idx, self.text_col]
        label = self.df.loc[idx, "label"]

        encoding = head_tail_truncate(
            text,
            self.tokenizer,
            max_len=self.max_len,
            head_len=self.head_len
        )

        return {
            "input_ids":      torch.tensor(encoding["input_ids"],
                                           dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"],
                                           dtype=torch.long),
            "labels":         torch.tensor(label, dtype=torch.long),
        }

# ============================================================
# HÜCRE 8 — Dataset'leri oluştur ve kontrol et
# ============================================================
print("Creating Datasets...")

roberta_train_dataset = IMDBDataset(train_df, roberta_tokenizer)
roberta_test_dataset  = IMDBDataset(test_df,  roberta_tokenizer)
deberta_train_dataset = IMDBDataset(train_df, deberta_tokenizer)
deberta_test_dataset  = IMDBDataset(test_df,  deberta_tokenizer)

sample = roberta_train_dataset[0]
print(f"\ninput_ids shape     : {sample['input_ids'].shape}")
print(f"attention_mask shape: {sample['attention_mask'].shape}")
print(f"label               : {sample['labels'].item()}")
print("\n✓ All datasets ready.")

# ============================================================
# HÜCRE 9 — Measure the truncation effect (for the report)
# ============================================================
sample_df = train_df.head(500).copy()
sample_df["token_count"] = sample_df["text_clean"].apply(
    lambda x: len(
        roberta_tokenizer(x, add_special_tokens=False)["input_ids"]
    )
)

truncated = (sample_df["token_count"] > 512).sum()
print(f"500 instances required truncation: {truncated} ({truncated/5:.1f}%)")
print("These instances were subjected to the Head+Tail strategy.")

# HÜCRE 10 — Save to Drive + Push to GitHub
train_df.to_parquet(f"{DIRS['root']}/train_df.parquet")
test_df.to_parquet(f"{DIRS['root']}/test_df.parquet")
print("✓ DataFrames have been saved to Drive.")

# First, go to File → Save to Drive, then run this cell.
NOTEBOOK_NAME = "01_data_preprocessing"

!jupyter nbconvert --to script \
  "/content/drive/MyDrive/Colab Notebooks/{NOTEBOOK_NAME}.ipynb" \
  --output-dir "/content/"

import os

# Convert .txt to .py
os.rename("/content/01_data_preprocessing.txt",
          "/content/01_data_preprocessing.py")

save_code_to_repo(f"/content/{NOTEBOOK_NAME}.py")
push_to_github("Data loading and preprocessing are complete.")

# Notebook'u bul
import subprocess
result = subprocess.run(
    ["find", "/content/drive/MyDrive", "-name", "01_data_preprocessing.ipynb"],
    capture_output=True, text=True
)
print(result.stdout)

!jupyter nbconvert --to script \
  "/content/drive/MyDrive/Colab Notebooks/01_data_preprocessing.ipynb" \
  --output-dir "/content/"
