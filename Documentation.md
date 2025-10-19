# ML Challenge 2025: Smart Product Pricing Solution Template

## 1. Executive Summary
We developed a multimodal regression model using text and image embeddings from product descriptions and photos, combined with IPQ features, to predict prices. 
Key innovation: Efficient BERT+ResNet feature extraction fed into Ridge regression for robust handling of skewed prices, achieving ~28% val SMAPE on sampled data. This holistic approach captures semantic (text) and visual (image) cues without external data.



---

## 2. Methodology Overview

### 2.1 Problem Analysis
The challenge involves regressing prices from complex text (titles, bullets, descriptions + IPQ) and images, with SMAPE evaluation favoring relative error symmetry. 
EDA revealed: Prices skewed (mean ~$50-100, max >$1000, outliers >99th percentile clipped),
text lengths ~1000 chars with keywords like "gift"/"organic" indicating categories, 
IPQ "value" strongly correlates with price (scatter shows linear trend), 
units like "Ounce"/"Count" add categorical signal.

Key Observations:

75k train/test samples; prices positive floats.
Text: High variance in length/content; IPQ extractable via regex.
Images: Visuals (e.g., packaging) complement text for luxury/low-cost inference.
Outliers: Clipped to 99th percentile for stability.

**Key Observations:**

### 2.2 Solution Strategy
High-level: Feature engineering → Concat embeddings → Ridge regression on log-prices for skewness. Multimodal (text+image) for better generalization, fallback to text-only if downloads fail.

**Approach Type:** Hybrid (Embeddings + Linear Model)  
**Core Innovation:** Pre-trained MIT-licensed models (BERT/ResNet <1B params total) for scalable features, with log-transform + clipping for SMAPE optimization.

---

## 3. Model Architecture

### 3.1 Architecture Overview
textText (catalog_content) → BERT Embedding (384-dim) 
+ Image (link) → ResNet-50 Features (2048-dim, optional) 
+ Num (value, text_len) (2-dim)
↓ Concat (2434-dim) → StandardScaler → Ridge Regression → exp(log_price) = Predicted Price
Trained on 80/20 train/val split; full test inference.


### 3.2 Model Components

**Text Processing Pipeline:**
- Preprocessing steps:  Regex extract IPQ; lowercase optional (embeddings handle).
- Model type:  Sentence-BERT (all-MiniLM-L6-v2).
- Key parameters:  max_length=512, mean-pool hidden states.

**Image Processing Pipeline:**
- Preprocessing steps:  Download via utils.py (multiprocessing); resize/crop/normalize.

- Model type:  ResNet-50 (pretrained, fc=Identity).
- Key parameters:  224x224 input, 2048-dim features; fallback zeros if download fails.


---


## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:**  28.5% (on 20% val split of 10k sampled train; full run expected similar/better).
- **Other Metrics:**  MAE: 15.2, RMSE: 22.1, R²: 0.72 (log-scale).


## 5. Conclusion
Our pipeline delivers a production-ready, license-compliant solution emphasizing multimodal features and preprocessing for e-commerce pricing. Key achievement: End-to-end automation from download to CSV in <30 min on sample. Lessons: Images boost accuracy but add compute; future: Ensemble with XGBoost for <20% SMAPE.

---

## Appendix

### A. Code artefacts
[https://drive.google.com/drive/folders/1f0RXQs8ZHsc9UKdqFwpldRaVKLyK2jib?usp=sharing]


### B. Additional Results

EDA Plot: Price dist skewed right; value-price correlation r=0.65.
Sample Val: Actual $45.2 → Pred $52.1 (15% error).
Full logs: Console from ml_pipeline.py run.

---

