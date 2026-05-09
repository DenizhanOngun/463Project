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

!pip install scikit-learn numpy pandas matplotlib seaborn -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             roc_auc_score, classification_report)
import json, time

SEED = 42
np.random.seed(SEED)

print("✓ Imports complete.")

# Load OOF probabilities from Drive
roberta_oof = np.load(f"{DIRS['oof']}/roberta_fold4.npy")
deberta_oof = np.load(f"{DIRS['oof']}/deberta_fold4.npy")

print(f"\nRoBERTa OOF shape: {roberta_oof.shape}")
print(f"DeBERTa OOF shape: {deberta_oof.shape}")

# Train labels
train_df = pd.read_parquet(f"{DIRS['root']}/train_df.parquet")
y_train  = train_df["label"].values

# Feature matrix — input from meta-learner
X_meta = np.column_stack([roberta_oof, deberta_oof])
print(f"Meta feature matrix: {X_meta.shape}")
print(f"Example: {X_meta[0]} → label: {y_train[0]}")

# Doğru yükleme — fold başına ayrı yükle, birleştir
import os

def load_oof_correct(model_name, n_folds=5):
    all_preds = []
    for fold in range(n_folds):
        path = f"{DIRS['oof']}/{model_name}_fold{fold}.npy"
        preds = np.load(path)
        # Her fold'da sadece o fold'un validation kısmı var
        # ama save_oof tüm diziyi kaydetti, son fold'u al
        all_preds.append(preds)
    return all_preds

# Her fold'un son kaydedilen halini incele
for fold in range(5):
    path = f"{DIRS['oof']}/roberta_fold{fold}.npy"
    arr  = np.load(path)
    print(f"Fold {fold} shape: {arr.shape} — sıfır olmayanlar: {(arr != 0).sum()}")

# Load predictions for both models for the test set
# We didn't generate these in the OOF notebook, we need to generate them now
# But our models are saved in Drive — we can load them from there and make predictions
# Alternative: predict the test set with the same OOF models

# For now: train meta-learner on OOF
# We will use the latest fold models for testing — this is the standard approach

# Train/val split — for meta-learner
from sklearn.model_selection import train_test_split

X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(
    X_meta, y_train,
    test_size=0.2,
    random_state=SEED,
    stratify=y_train
)

print(f"Meta train: {X_train_meta.shape}")
print(f"Meta val  : {X_val_meta.shape}")
print(f"\nLabel distribution (train): {np.bincount(y_train_meta)}")
print(f"Label distribution (val)  : {np.bincount(y_val_meta)}")

def evaluate_meta(name, y_true, y_pred, y_prob):
    return {
        "model"    : name,
        "accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "f1"       : round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall"   : round(recall_score(y_true, y_pred), 4),
        "roc_auc"  : round(roc_auc_score(y_true, y_prob), 4),
    }

results = []

# ── 1. Weighted Average (baseline, not learned) ───────────────────
wa_probs = (roberta_oof[:len(y_val_meta)] * 0.5 +
            deberta_oof[:len(y_val_meta)] * 0.5)

# Use the correct indexes
_, X_val_r, _, y_val_check = train_test_split(
    roberta_oof, y_train, test_size=0.2, random_state=SEED, stratify=y_train
)
_, X_val_d = train_test_split(
    deberta_oof, test_size=0.2, random_state=SEED
)

wa_probs = X_val_r * 0.5 + X_val_d * 0.5
wa_preds = (wa_probs > 0.5).astype(int)
results.append(evaluate_meta("Weighted Average", y_val_meta, wa_preds, wa_probs))
print("✓ Weighted Average is completed.")

# ── 2. Logistic Regression ────────────────────────────────────────
start = time.time()
lr = LogisticRegression(random_state=SEED, max_iter=1000)
lr.fit(X_train_meta, y_train_meta)
lr_time = time.time() - start

lr_probs = lr.predict_proba(X_val_meta)[:, 1]
lr_preds = lr.predict(X_val_meta)
res = evaluate_meta("Logistic Regression", y_val_meta, lr_preds, lr_probs)
res["train_time"] = round(lr_time, 3)
res["coefficients"] = {
    "roberta_weight": round(lr.coef_[0][0], 4),
    "deberta_weight": round(lr.coef_[0][1], 4)
}
results.append(res)
print(f"✓ Logistic Regression is completed. ({lr_time:.2f}s)")
print(f"  Weight of RoBERTa: {lr.coef_[0][0]:.4f}")
print(f"  Weight of DeBERTa: {lr.coef_[0][1]:.4f}")

# ── 3. MLP ───────────────────────────────────────────────────────
start = time.time()
mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    max_iter=500,
    random_state=SEED
)
mlp.fit(X_train_meta, y_train_meta)
mlp_time = time.time() - start

mlp_probs = mlp.predict_proba(X_val_meta)[:, 1]
mlp_preds = mlp.predict(X_val_meta)
res = evaluate_meta("MLP", y_val_meta, mlp_preds, mlp_probs)
res["train_time"] = round(mlp_time, 3)
results.append(res)
print(f"✓ MLP is completed. ({mlp_time:.2f}s)")

# ── Results Table ──────────────────────────────────────────────
results_df = pd.DataFrame(results)
print(f"\n{'='*70}")
print("META-STUDENT COMPARISON")
print(f"{'='*70}")
print(results_df[["model", "accuracy", "f1", "precision",
                   "recall", "roc_auc"]].to_string(index=False))

save_results({"meta_learner_comparison": results}, "meta_learner_results.json")
print("✓ Results have been saved.")

from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

models = [
    ("Weighted Average", wa_preds, wa_probs),
    ("Logistic Regression", lr_preds, lr_probs),
    ("MLP", mlp_preds, mlp_probs),
]

for ax, (name, preds, probs) in zip(axes, models):
    cm = confusion_matrix(y_val_meta, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Negative", "Pozitive"],
                yticklabels=["Negative", "Pozitive"])
    ax.set_title(f"{name}\nAcc: {accuracy_score(y_val_meta, preds):.4f}")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(f"{DIRS['results']}/confusion_matrices.png", dpi=150)
plt.show()
print("✓ Confusion matrix saved to Drive.")

from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(8, 6))

for name, preds, probs in models:
    fpr, tpr, _ = roc_curve(y_val_meta, probs)
    auc_score   = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.4f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Meta-Learner Comparision ")
plt.legend()
plt.tight_layout()
plt.savefig(f"{DIRS['results']}/roc_curves.png", dpi=150)
plt.show()
print("✓ ROC curve saved to Drive.")

NOTEBOOK_NAME = "06_meta_learner"

!jupyter nbconvert --to script \
  "/content/drive/MyDrive/Colab Notebooks/{NOTEBOOK_NAME}.ipynb" \
  --output-dir "/content/"

import os
os.rename(f"/content/{NOTEBOOK_NAME}.txt",
          f"/content/{NOTEBOOK_NAME}.py")

save_code_to_repo(f"/content/{NOTEBOOK_NAME}.py")
push_to_github("meta learner is completed LR 0.9462 MLP 0.9476")
