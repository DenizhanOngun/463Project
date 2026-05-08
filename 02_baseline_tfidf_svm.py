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

!pip install transformers datasets peft accelerate scikit-learn -q

import pandas as pd

train_df = pd.read_parquet(f"{DIRS['root']}/train_df.parquet")
test_df  = pd.read_parquet(f"{DIRS['root']}/test_df.parquet")

print(f"Train : {len(train_df)} example")
print(f"Test  : {len(test_df)} example")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time

# TF-IDF — Use the lowercase version (this is the correct one for SVM)
print("TF-IDF vectors are being generated...")
tfidf = TfidfVectorizer(
    max_features=50000,   # most frequent 50k words
    ngram_range=(1, 2),   # unigram + bigram
    min_df=2              # words that appear at least twice
)

X_train = tfidf.fit_transform(train_df["text_clean_lower"])
X_test  = tfidf.transform(test_df["text_clean_lower"])
y_train = train_df["label"]
y_test  = test_df["label"]

print(f"Train matrix : {X_train.shape}")
print(f"Test matrix  : {X_test.shape}")

# SVM Training
print("\nSVM being trained...")
start = time.time()
svm = LinearSVC(max_iter=2000)
svm.fit(X_train, y_train)
train_time = time.time() - start

# Evaluation
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)

print(f"\n{'='*40}")
print(f"TF-IDF + SVM Results")
print(f"{'='*40}")
print(f"Accuracy  : {acc:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"Train time: {train_time:.1f} second")
print(f"\n{classification_report(y_test, y_pred)}")

import json

results = {
    "model": "TF-IDF + SVM",
    "accuracy": round(acc, 4),
    "f1_score": round(f1, 4),
    "train_time_seconds": round(train_time, 1),
    "train_time_note": "Colab CPU serves as an indicator for comparison.",
    "trainable_parameters": "N/A - no neural network",
    "tfidf_features": 50000,
    "ngram_range": "(1,2)"
}

save_results(results, "baseline_tfidf_svm.json")
print(json.dumps(results, indent=2))


