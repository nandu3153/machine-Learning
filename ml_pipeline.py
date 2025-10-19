# if you want to run this code, you need:
# dataset folder -that contains--> train.csv and test.csv
# src folder -that contains--> utils.py (with download_images function)
# by the way i also added Documentation file (README.md) for you to understand the code better
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from torchvision import models, transforms
from sklearn.linear_model import Ridge
from src.utils import download_images 
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import joblib
import torch
import re
import os


warnings.filterwarnings('ignore')

# Config
DATASET_FOLDER = 'dataset/'
IMAGES_FOLDER = 'images/'
OUTPUTS_FOLDER = 'outputs/'
SAMPLE_SIZE = 10000  
USE_IMAGES = True  
os.makedirs(IMAGES_FOLDER, exist_ok=True)
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)


# SMAPE metric
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


# 1. Load Data
print("Loading data...")
train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(
    f"Price stats: mean={train['price'].mean():.2f}, median={train['price'].median():.2f}, min={train['price'].min():.2f}, max={train['price'].max():.2f}")

# Sample train for speed
if SAMPLE_SIZE is not None:
    train = train.sample(n=min(SAMPLE_SIZE, len(train)), random_state=42).reset_index(drop=True)
    print(f"Sampled train to {len(train)} rows")

# 2. EDA
print("\n--- EDA ---")
# Text
train['text_len'] = train['catalog_content'].str.len()
print(f"Text length: mean={train['text_len'].mean():.0f}, max={train['text_len'].max():.0f}")

# IPQ extraction
train['value'] = train['catalog_content'].str.extract(r'Value: ([\d.]+)', expand=False).astype(float).fillna(1.0)
train['unit'] = train['catalog_content'].str.extract(r'Unit: (\w+)', expand=False).fillna('Count')
print(f"IPQ: value mean={train['value'].mean():.1f}, top units: {train['unit'].value_counts().head()}")

# Price dist
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.histplot(train['price'], bins=50, kde=True)
plt.title('Price Distribution')
plt.subplot(1, 3, 2)
sns.boxplot(y=train['price'])
plt.title('Price Outliers')
plt.subplot(1, 3, 3)
sns.scatterplot(x='value', y='price', data=train)
plt.title('Price vs Value')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_FOLDER, 'eda_plots.png'))
plt.show()

# Word freq
all_text = ' '.join(train['catalog_content'].str.lower())
words = re.findall(r'\b\w+\b', all_text)
print("Top words:", Counter(words).most_common(5))

# 3. Download Images (if used)
if USE_IMAGES:
    print("\nDownloading images...")
    # Train sample
    train_sample_links = train['image_link'].unique()[:min(2000, len(train))]  
    download_images(train_sample_links, os.path.join(IMAGES_FOLDER, 'train_sample/'))
    print(f"Downloaded {len(os.listdir(os.path.join(IMAGES_FOLDER, 'train_sample/')))} train images")

    # Full test (do this once; time-consuming)
    if not os.path.exists(os.path.join(IMAGES_FOLDER, 'test/')) or len(
            os.listdir(os.path.join(IMAGES_FOLDER, 'test/'))) < len(test):
        download_images(test['image_link'].tolist(), os.path.join(IMAGES_FOLDER, 'test/'))
        print(f"Downloaded {len(os.listdir(os.path.join(IMAGES_FOLDER, 'test/')))} test images")

# 4. Feature Engineering
print("\n--- Feature Engineering ---")

# Text Embeddings (BERT)
print("Computing text embeddings...")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
text_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # 384-dim


train['text_emb'] = [get_text_embedding(text) for text in tqdm(train['catalog_content'], desc="Text embeds")]
test['text_emb'] = [get_text_embedding(text) for text in tqdm(test['catalog_content'], desc="Test text embeds")]

# Image Embeddings 
if USE_IMAGES:
    print("Computing image embeddings...")
    resnet = models.resnet50(pretrained=True)
    resnet.fc = torch.nn.Identity()
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    def get_image_embedding(img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0)
            with torch.no_grad():
                return resnet(img_t).cpu().numpy().flatten()  
        except:
            return np.zeros(2048)  


    # Train
    train['img_emb'] = []
    for _, row in tqdm(train.iterrows(), total=len(train), desc="Train img embeds"):
        filename = row['image_link'].split('/')[-1]
        img_path = os.path.join(IMAGES_FOLDER, 'train_sample/', filename)
        train.at[_, 'img_emb'] = get_image_embedding(img_path)

    # Test
    test['img_emb'] = []
    for _, row in tqdm(test.iterrows(), total=len(test), desc="Test img embeds"):
        filename = row['image_link'].split('/')[-1]
        img_path = os.path.join(IMAGES_FOLDER, 'test/', filename)
        test.at[_, 'img_emb'] = get_image_embedding(img_path)
else:
    # Dummy images for concat
    img_dim = 2048
    train['img_emb'] = [np.zeros(img_dim) for _ in range(len(train))]
    test['img_emb'] = [np.zeros(img_dim) for _ in range(len(test))]

# Combine features
text_dim = 384
img_dim = 2048 if USE_IMAGES else 0
num_features = ['value', 'text_len']
for df in [train, test]:
    df['text_len'] = df['catalog_content'].str.len()
    df['value'] = df['catalog_content'].str.extract(r'Value: ([\d.]+)', expand=False).astype(float).fillna(1.0)

# Stack
X_train_text = np.stack(train['text_emb'].values)
X_train_img = np.stack(train['img_emb'].values)
X_train_num = train[num_features].values

X_train = np.hstack([X_train_text, X_train_img, X_train_num])
y_train = np.log1p(train['price'].clip(lower=0.01))  

X_test_text = np.stack(test['text_emb'].values)
X_test_img = np.stack(test['img_emb'].values)
X_test_num = test[num_features].values
X_test = np.hstack([X_test_text, X_test_img, X_test_num])

# 5. Train Model
print("\n--- Training ---")
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = Ridge(alpha=1.0, random_state=42)
model.fit(X_tr_scaled, y_tr)

# Val predictions
preds_val_log = model.predict(X_val_scaled)
preds_val = np.expm1(preds_val_log)
mae_val = mean_absolute_error(np.expm1(y_val), preds_val)
smape_val = smape(np.expm1(y_val), preds_val)
print(f"Val MAE: {mae_val:.2f}, Val SMAPE: {smape_val:.2f}%")

# 6. Predict & Save
print("\n--- Predictions ---")
preds_test_log = model.predict(X_test_scaled)
test['price'] = np.maximum(np.expm1(preds_test_log), 0.01)
output_df = test[['sample_id', 'price']].round(2)

output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
output_df.to_csv(output_filename, index=False)

print(f"Predictions saved to {output_filename}")
print(f"Total predictions: {len(output_df)}")
print(f"Sample predictions:\n{output_df.head()}")


# Save model and scaler
joblib.dump(model, os.path.join(OUTPUTS_FOLDER, 'ridge_model.pkl'))
joblib.dump(scaler, os.path.join(OUTPUTS_FOLDER, 'scaler.pkl'))
print("Model and scaler saved.")