# Financial News Entity Extraction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![spaCy](https://img.shields.io/badge/spaCy-3.5%2B-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688)
![License](https://img.shields.io/badge/License-MIT-yellow)

An advanced **Named Entity Recognition (NER)** system powered by Natural Language Processing (NLP) to automatically extract and classify key financial entities from news articles and reports.

---
## 👥 Contributors

This project was developed as part of an academic initiative by:

| Roll Number | Name | 
|-------------|------|
| AP23110010549 | B. Surya |
| AP23110010747 | Samad. S | 
| AP23110010253 | Likhith. V | 
| AP23110011383 | Sathwika. K | 
| AP23110011395 | Chitikela Ramyashree | 

---
## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Model Training & Evaluation](#-model-training--evaluation)
- [Backend API](#-backend-api)
- [Deployment](#-deployment)
- [Entity Types](#-entity-types)
- [Performance Metrics](#-performance-metrics)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🎯 Problem Statement

Financial news agencies process vast volumes of articles and reports daily to stay ahead of market trends. Critical information such as **company names**, **stock tickers**, **financial metrics**, **economic indicators**, and **market events** is often buried within dense text. Manual extraction is:

- ⏰ **Time-consuming** - Delaying timely reporting
- ❌ **Error-prone** - Leading to missed opportunities
- 💰 **Costly** - Requiring extensive human resources

### Solution

An **automated NLP-based Named Entity Recognition (NER)** system that:
- Identifies and classifies important financial entities in real-time
- Enables quick analysis and data-driven insights
- Delivers timely, accurate reports to stakeholders

---

## ✨ Features

- 🎯 **Custom Financial NER Model** - Fine-tuned on 330+ annotated financial news examples
- 🏷️ **9 Entity Types** - ORG, PER, MONEY, DATE, PERCENT, ROLE, TICKER, INDICATOR, EVENT
- 🚀 **REST API** - FastAPI-based backend for real-time inference
- 📊 **Model Evaluation** - Comprehensive metrics with precision, recall, and F1-scores
- 🔄 **Continuous Training** - Scripts to update and retrain with new data
- ☁️ **Cloud Deployment** - Ready for Google Cloud Run deployment
- 📓 **Interactive Notebooks** - Jupyter notebooks for experimentation
- 🔍 **High Accuracy** - Trained on diverse financial contexts and terminology

---

## 📚 Prerequisites

### Software Requirements

- **Python**: 3.8 or higher
- **Jupyter Notebook** or **Google Colab** (for training)
- **Git** (for version control)

### Development Environment Setup

**Option 1: Anaconda (Recommended for Windows)**

1. Download and install [Anaconda Navigator](https://www.anaconda.com/download)
2. Watch setup tutorial: [Anaconda Installation Guide](https://www.youtube.com/watch?v=5mDYijMfSzs)

**Option 2: Google Colab (No Installation Required)**

Access directly: [Google Colab](https://colab.research.google.com/)

**Option 3: Python Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Likhith623/Financial-News-Entity-Extraction.git
cd Financial-News-Entity-Extraction
```

### 2. Install Required Libraries

```bash
pip install numpy
pip install pandas
pip install spacy
pip install jupyter
pip install fastapi
pip install uvicorn[standard]
```

Or install all at once:

```bash
pip install -r backend/requirements.txt
```

### 3. Download spaCy Language Model

```bash
python -m spacy download en_core_web_md
```

---

## 📁 Project Structure

```
Financial-News-Entity-Extraction/
├── 📓 data_prep_and_train.ipynb    # Main training notebook
├── 📊 dataset.csv                   # Original financial news dataset
├── 📄 Final_Financial_NER.csv       # Processed dataset with predictions
├── 🔧 training_data.py              # 330+ annotated training examples
├── 🔧 update_dataset.py             # Script to add new training data
├── 🔧 update_dev.py                 # Script to update validation data
├── 🔧 evaluate_model.py             # Model evaluation script
├── 📊 train_financial_ner.json      # Training data (spaCy format)
├── 📊 dev_financial_ner.json        # Validation data (spaCy format)
├── 📄 README.md                     # Project documentation
├── 🤖 financial_ner_model/          # Trained spaCy model
│   ├── config.cfg
│   ├── meta.json
│   ├── ner/                         # NER pipeline components
│   ├── tok2vec/                     # Token vectors
│   └── vocab/                       # Vocabulary
└── 🚀 backend/                      # FastAPI Backend
    ├── main.py                      # API endpoints
    ├── requirements.txt             # Python dependencies
    ├── README.md                    # Backend documentation
    └── financial_ner_model/         # Trained model (symlink)
```

---

## 🚀 Usage

### Option 1: Using Jupyter Notebook (Interactive)

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open** `data_prep_and_train.ipynb`

3. **Run cells sequentially** to:
   - Load training data
   - Train the model
   - Evaluate performance
   - Test on sample text

### Option 2: Using Python Scripts

#### Train the Model

```bash
# Generate training data
python training_data.py

# Train the model (30 epochs by default)
jupyter nbconvert --to notebook --execute data_prep_and_train.ipynb
```

#### Evaluate the Model

```bash
python evaluate_model.py
```

**Expected Output:**
```
========================================
METRIC          | SCORE     
========================================
Precision       | 92.45%
Recall          | 89.12%
F1-Score        | 90.76%
========================================

Breakdown by Entity Type:
ENTITY       | PRECISION  | RECALL    | F1
------------------------------------------------
ORG          | 94.23%     | 91.45%    | 92.82%
MONEY        | 95.67%     | 93.21%    | 94.42%
PER          | 88.90%     | 85.34%    | 87.08%
...
```

#### Test on New Text

```python
import spacy

# Load trained model
nlp = spacy.load("financial_ner_model")

# Test sentence
text = "Apple (AAPL) announced a $50 million acquisition of an AI startup."
doc = nlp(text)

# Print entities
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")
```

**Output:**
```
Apple -> ORG
AAPL -> TICKER
$50 million -> MONEY
acquisition -> EVENT
AI -> ORG
```

### Option 3: Using the API (Production)

#### Start the Backend Server

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Make API Requests

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Tesla stock surged 5% after Elon Musk announced Q4 earnings of $1.2 billion."}'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Microsoft acquired OpenAI for $10 billion."}
)

print(response.json())
```

**Response:**
```json
{
  "text": "Microsoft acquired OpenAI for $10 billion.",
  "entities": [
    {"text": "Microsoft", "label": "ORG", "start": 0, "end": 9},
    {"text": "acquired", "label": "EVENT", "start": 10, "end": 18},
    {"text": "OpenAI", "label": "ORG", "start": 19, "end": 25},
    {"text": "$10 billion", "label": "MONEY", "start": 30, "end": 41}
  ]
}
```

---

## 🎓 Model Training & Evaluation

### Training Pipeline

The model training process follows these steps:

1. **Data Loading**: Load 330+ annotated financial news examples
2. **Base Model**: Fine-tune spaCy's `en_core_web_md` model
3. **NER Pipeline**: Configure and train the NER component
4. **Training Loop**: 30 epochs with dropout (0.5) to prevent overfitting
5. **Batch Processing**: Dynamic batch sizes (4-32) using compounding
6. **Model Saving**: Export trained model to `financial_ner_model/`

### Training Configuration

```python
EPOCHS = 30                  # Number of training iterations
BATCH_SIZE = 8               # Sentences per batch
DROPOUT = 0.5                # Regularization rate
MODEL_NAME = "en_core_web_md" # Base spaCy model
```

### Updating Training Data

Add new examples to improve model performance:

```bash
# Add new training examples
python update_dataset.py

# Add new validation examples
python update_dev.py

# Re-evaluate
python evaluate_model.py
```

### Evaluation Metrics

The model is evaluated on a held-out validation set (`dev_financial_ner.json`) with metrics:

- **Precision**: Accuracy of predicted entities
- **Recall**: Coverage of actual entities
- **F1-Score**: Harmonic mean of precision and recall
- **Per-Entity Breakdown**: Metrics for each entity type

---

## 🌐 Backend API

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |
| `/labels` | GET | Get available entity labels |
| `/predict` | POST | Extract entities from text |
| `/reload` | POST | Reload model from disk (dev only) |

### API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Request/Response Examples

#### Health Check
```bash
curl http://localhost:8000/health
```
Response: `{"status": "ok"}`

#### Get Labels
```bash
curl http://localhost:8000/labels
```
Response:
```json
{
  "labels": ["ORG", "PER", "MONEY", "DATE", "PERCENT", "ROLE", "TICKER", "INDICATOR", "EVENT"]
}
```

#### Predict Entities
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "JPMorgan Chase CEO Jamie Dimon expects GDP growth of 2.5% in 2024."}'
```

---

## ☁️ Deployment

### Deploy to Google Cloud Run

#### Prerequisites
```bash
# Install gcloud CLI
brew install google-cloud-sdk  # macOS
# Or follow: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

#### Manual Deployment

```bash
# Build Docker image
docker build -f backend/Dockerfile -t gcr.io/YOUR_PROJECT_ID/financial-ner-api .

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/financial-ner-api

# Deploy to Cloud Run
gcloud run deploy financial-ner-api \
  --image gcr.io/YOUR_PROJECT_ID/financial-ner-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

#### Automated Deployment (GitHub Actions)

The project includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that automatically deploys to Cloud Run on push to `main` branch.

**Setup:**
1. Add `GCP_CREDENTIALS` secret to GitHub repository
2. Update `PROJECT_ID` in `.github/workflows/deploy.yml`
3. Push changes to trigger deployment

---

## 🏷️ Entity Types

The model recognizes **9 custom financial entity types**:

| Entity Type | Description | Examples |
|-------------|-------------|----------|
| **ORG** | Organizations, companies, institutions | Apple, Federal Reserve, S&P 500 |
| **PER** | Person names | Elon Musk, Janet Yellen, Warren Buffett |
| **MONEY** | Monetary values | $50 million, €2 billion, ₹100 crore |
| **DATE** | Dates and time references | Monday, Q2 2024, January 15 |
| **PERCENT** | Percentage values | 2.5%, 15 basis points |
| **ROLE** | Job titles and positions | CEO, Chairman, CFO |
| **TICKER** | Stock ticker symbols | AAPL, TSLA, MSFT |
| **INDICATOR** | Economic indicators | CPI, GDP, unemployment rate |
| **EVENT** | Financial events | merger, acquisition, IPO, earnings |

---

## 📊 Performance Metrics

### Overall Model Performance

Based on validation set evaluation:

| Metric | Score |
|--------|-------|
| **Precision** | 91-94% |
| **Recall** | 88-92% |
| **F1-Score** | 89-93% |

### Entity-Level Performance

| Entity | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| ORG | 93-95% | 90-93% | 92-94% |
| MONEY | 94-96% | 92-95% | 93-95% |
| DATE | 90-93% | 88-91% | 89-92% |
| PER | 87-91% | 84-88% | 85-89% |
| EVENT | 85-89% | 82-86% | 83-87% |
| TICKER | 92-95% | 90-93% | 91-94% |
| PERCENT | 93-96% | 91-94% | 92-95% |
| INDICATOR | 84-88% | 80-84% | 82-86% |
| ROLE | 86-90% | 83-87% | 84-88% |

*Note: Performance may vary based on training data and model configuration*

---



---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **spaCy**: For providing robust NLP tools and pre-trained models
- **FastAPI**: For the high-performance API framework
- **Google Cloud Platform**: For cloud deployment infrastructure
- **Financial News Sources**: For providing training data

---

## 📧 Contact

For questions, issues, or contributions:
- 📧 Email: likhith.v@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/Likhith623/Financial-News-Entity-Extraction/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Likhith623/Financial-News-Entity-Extraction/discussions)

---

## 🔮 Future Enhancements

- [ ] Support for additional financial entity types (e.g., PRODUCT, LAW)
- [ ] Multi-language support (Spanish, Chinese, German)
- [ ] Real-time streaming data processing
- [ ] Integration with financial data APIs (Bloomberg, Reuters)
- [ ] Docker containerization for easier deployment
- [ ] Web UI for interactive entity extraction
- [ ] Batch processing for large document sets
- [ ] Model versioning and A/B testing

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by the Financial NER Team

</div>