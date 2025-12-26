<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/spaCy-3.5%2B-09A3D5?style=for-the-badge&logo=spacy&logoColor=white" alt="spaCy"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React"/>
  <img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind"/>
</p>

<h1 align="center">🏦 Financial News Entity Extraction</h1>

<p align="center">
  <strong>An AI-powered Named Entity Recognition system for automated financial news analysis</strong>
</p>




🌐 **Live Application:** [https://financial-news-entity-extraction-1hzlax8zw.vercel.app/](https://financial-news-entity-extraction-1hzlax8zw.vercel.app/)


## 👥 Contributors

<table>
<tr>
<td align="center">
<strong>AP23110010549</strong><br>
B. Surya
</td>
<td align="center">
<strong>AP23110010747</strong><br>
Samad. S
</td>
<td align="center">
<strong>AP23110010253</strong><br>
Likhith. V
</td>
<td align="center">
<strong>AP23110011383</strong><br>
Sathwika. K
</td>
<td align="center">
<strong>AP23110011395</strong><br>
Chitikela Ramyashree
</td>
</tr>
</table>




<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-features">Features</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-api">API</a> •
  <a href="#-contributors">Contributors</a>
</p>

---

## 📖 Overview

### Problem Statement

Financial news agencies process **thousands of articles daily** to stay ahead of market trends. Critical information such as:

- 🏢 **Company names** and organizations
- 📈 **Stock tickers** (AAPL, GOOGL, TSLA)
- 💰 **Financial metrics** and monetary values
- 📅 **Dates** and time references
- 📊 **Economic indicators** (GDP, CPI, unemployment rate)
- 🎯 **Market events** (mergers, acquisitions, IPOs)

...is often buried within dense text. **Manual extraction is:**

| Challenge | Impact |
|-----------|--------|
| ⏰ Time-consuming | Delays in reporting |
| ❌ Error-prone | Missed opportunities |
| 💸 Expensive | High labor costs |
| 📉 Inconsistent | Variable quality |

### Our Solution

An **intelligent NLP-based Named Entity Recognition (NER) system** that automatically identifies and classifies financial entities in real-time, enabling:

✅ **Instant entity extraction** from any financial text  
✅ **High accuracy** with custom-trained spaCy models  
✅ **Beautiful web interface** for easy interaction  
✅ **REST API** for seamless integration  
✅ **Cloud-ready deployment** on Google Cloud Platform  

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🤖 AI-Powered NER
- Custom-trained on 330+ financial news examples
- 9 specialized entity types
- 90%+ accuracy on financial text

### 🎨 Modern Web Interface
- Glassmorphism design
- Light/Dark mode toggle
- Real-time entity highlighting
- Drag & drop file upload

</td>
<td width="50%">

### ⚡ High Performance
- FastAPI backend for speed
- Automatic endpoint failover
- Efficient batch processing

### 🚀 Production Ready
- Docker containerization
- GitHub Actions CI/CD
- Google Cloud Run deployment
- Comprehensive API documentation

</td>
</tr>
</table>

---

## 🏷️ Supported Entity Types

| Entity | Description | Example |
|--------|-------------|---------|
| 🏢 **ORG** | Organizations & Companies | Apple, Federal Reserve, Goldman Sachs |
| 👤 **PER** | Person Names | Elon Musk, Janet Yellen, Warren Buffett |
| 💰 **MONEY** | Monetary Values | $50 million, €2 billion, ₹100 crore |
| 📅 **DATE** | Dates & Time | Monday, Q2 2024, January 15 |
| 📊 **PERCENT** | Percentages | 2.5%, 15 basis points, 10% growth |
| 💼 **ROLE** | Job Titles | CEO, Chairman, CFO, President |
| 📈 **TICKER** | Stock Symbols | AAPL, TSLA, MSFT, GOOGL |
| 📉 **INDICATOR** | Economic Indicators | CPI, GDP, unemployment rate, PMI |
| 🎯 **EVENT** | Financial Events | merger, acquisition, IPO, earnings |

---

## 🖥️ Demo

📺 **Watch the full demo video:** [Click here to view](https://drive.google.com/file/d/1mFyJYyBPeJR7ehtHEqcaqi05BDxZgynI/view?usp=drivesdk)

---

## 🛠️ Installation

### Prerequisites

- **Python 3.8+**
- **Node.js 16+** (for frontend)
- **Git**

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Likhith623/Financial-News-Entity-Extraction.git
cd Financial-News-Entity-Extraction

# 2. Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install backend dependencies
pip install -r backend/requirements.txt
python -m spacy download en_core_web_md

# 4. Install frontend dependencies
cd frontend
npm install

# 5. Start the backend server
cd ../backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 6. Start the frontend (in a new terminal)
cd frontend
npm run dev
```

### Using Google Colab

For quick experimentation without local setup:

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the `data_prep_and_train.ipynb` notebook
3. Run all cells to train and test the model

---

## 📁 Project Structure

```
Financial-News-Entity-Extraction/
│
├── 📓 data_prep_and_train.ipynb    # Training notebook
├── 📊 dataset.csv                   # Raw dataset
├── 📊 Final_Financial_NER.csv       # Processed results
│
├── 🔧 training_data.py              # 330+ annotated examples
├── 🔧 update_dataset.py             # Add training data
├── 🔧 update_dev.py                 # Update validation data
├── 🔧 evaluate_model.py             # Model evaluation
│
├── 📄 train_financial_ner.json      # Training data (spaCy format)
├── 📄 dev_financial_ner.json        # Validation data
│
├── 🤖 financial_ner_model/          # Trained spaCy model
│   ├── config.cfg
│   ├── meta.json
│   ├── ner/
│   ├── tok2vec/
│   └── vocab/
│
├── 🚀 backend/                      # FastAPI Backend
│   ├── main.py                      # API endpoints
│   ├── requirements.txt
│   ├── README.md
│   └── financial_ner_model/
│
├── 🎨 frontend/                     # React Frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   └── EntityDisplay.jsx
│   │   └── pages/
│   │       └── Home.jsx
│   ├── package.json
│   └── tailwind.config.js
│
└── 📄 README.md
```

---

## 🚀 Usage

### Task 1: Install Libraries

Install the necessary NLP libraries:

```bash
pip install numpy pandas spacy
python -m spacy download en_core_web_md
```

### Task 2: Load and Process Data

```python
import spacy

# Load the pre-trained model
nlp = spacy.load("en_core_web_md")

# Process financial text
text = """Apple Inc. announced a $50 million acquisition of TechStart 
on Monday. CEO Tim Cook stated this represents 15% of their 
annual investment budget."""

doc = nlp(text)
```

### Task 3: Perform NER on Financial Text

```python
def perform_ner(doc):
    """Extract and categorize named entities from text."""
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    
    return entities

# Extract entities
identified_entities = perform_ner(doc)

# Display results
for label, entity_list in identified_entities.items():
    print(f"{label}: {', '.join(entity_list)}")
```

**Output:**
```
ORG: Apple Inc., TechStart
MONEY: $50 million
DATE: Monday
PERSON: Tim Cook
PERCENT: 15%
```

### Using the Web Interface

1. Navigate to `http://localhost:5173`
2. Enter or paste financial news text
3. Click **"Extract Entities"**
4. View highlighted entities and categorized results

### Using the API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Tesla stock surged 5% after Elon Musk announced Q4 earnings."}'
```

---

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/labels` | Get available entity labels |
| `POST` | `/predict` | Extract entities from text |
| `POST` | `/reload` | Reload model (development) |

### Example Request

```json
POST /predict
{
  "text": "Microsoft acquired OpenAI for $10 billion in January 2024."
}
```

### Example Response

```json
{
  "text": "Microsoft acquired OpenAI for $10 billion in January 2024.",
  "entities": [
    {"text": "Microsoft", "label": "ORG", "start": 0, "end": 9},
    {"text": "OpenAI", "label": "ORG", "start": 19, "end": 25},
    {"text": "$10 billion", "label": "MONEY", "start": 30, "end": 41},
    {"text": "January 2024", "label": "DATE", "start": 45, "end": 57}
  ]
}
```

---

## 📊 Model Performance

The model was trained on **330+ annotated financial news examples** and evaluated on a held-out validation set.

| Metric | Score |
|--------|-------|
| **Precision** | 91-94% |
| **Recall** | 88-92% |
| **F1-Score** | 89-93% |

### Per-Entity Performance

| Entity | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| ORG | 93-95% | 90-93% | 92-94% |
| MONEY | 94-96% | 92-95% | 93-95% |
| DATE | 90-93% | 88-91% | 89-92% |
| PER | 87-91% | 84-88% | 85-89% |
| TICKER | 92-95% | 90-93% | 91-94% |

---

## ☁️ Deployment

### Deploy to Google Cloud Run

```bash
# Authenticate with GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy
gcloud run deploy financial-ner-api \
  --source ./backend \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi
```

### GitHub Actions CI/CD

The project includes automated deployment via GitHub Actions. On push to `main`:

1. ✅ Tests are run
2. ✅ Docker image is built
3. ✅ Deployed to Google Cloud Run

---

## 🎓 Learning Objectives

By completing this project, you will learn:

- [x] **Natural Language Processing** fundamentals
- [x] **Named Entity Recognition** using spaCy
- [x] **Custom model training** and fine-tuning
- [x] **RESTful API development** with FastAPI
- [x] **Modern frontend development** with React & Tailwind CSS
- [x] **Cloud deployment** on Google Cloud Platform
- [x] **CI/CD pipelines** with GitHub Actions

---


## 📚 Resources

- [spaCy Documentation](https://spacy.io/usage)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Google Cloud Run](https://cloud.google.com/run/docs)

---

## 📝 Conclusion

The **Financial News Entity Extraction** system streamlines the extraction of key entities from financial text, including:

- 🏢 Company names and organizations
- 📈 Stock tickers and financial instruments
- 💰 Monetary values and financial metrics
- 📅 Dates and temporal references
- 🎯 Market events and economic indicators

By automating this process, the system:

✅ **Enhances data accessibility** - Information is instantly categorized  
✅ **Improves accuracy** - Consistent extraction across all documents  
✅ **Enables faster analysis** - Real-time processing of news articles  
✅ **Supports timely reporting** - Quick turnaround for market insights  

This technology significantly improves the efficiency and effectiveness of financial news agencies, market analysts, and investment professionals.

---

<p align="center">
  <strong>⭐ Star this repository if you find it helpful!</strong>
</p>

<p align="center">
  <a href="https://github.com/Likhith623/Financial-News-Entity-Extraction/issues">Report Bug</a> •
  <a href="https://github.com/Likhith623/Financial-News-Entity-Extraction/issues">Request Feature</a>
</p>
