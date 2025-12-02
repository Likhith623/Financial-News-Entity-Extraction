# Backedn (FastAPI) for Financial NER

This folder contains a small FastAPI backend to serve the `financial_ner_model` spaCy model included in the repository.

Files:
- `main.py` — FastAPI application exposing endpoints:
  - `GET /health` — simple health check
  - `GET /labels` — returns NER labels available in the model
  - `POST /predict` — accepts JSON `{ "text": "..." }` and returns detected entities
  - `POST /reload` — reloads the model from disk (development use)

Run locally (from project root):

```bash
# install dependencies (preferably in a venv)
pip install -r requirements.txt

# start the server
uvicorn backedn.main:app --host 0.0.0.0 --port 8000 --reload
```

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple announced a $50 million acquisition."}'
```

Notes:
- The backend expects the trained model to be accessible at the `financial_ner_model` directory in the repository root.
- In production, remove `--reload` and consider process managers like systemd, Docker, or gunicorn with uvicorn workers.

---

## Deploy to Google Cloud Platform (GCP)

### Prerequisites

1. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Authenticate: `gcloud auth login`
3. Set your project: `gcloud config set project YOUR_PROJECT_ID`
4. Enable required APIs:
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

### Option 1: Deploy to Cloud Run (Recommended)

Cloud Run is serverless, scales to zero, and is cost-effective.

#### Using Cloud Build (Automated):

From the project root:

```bash
# Submit build and deploy using cloudbuild.yaml
gcloud builds submit --config cloudbuild.yaml
```

This will:
- Build the Docker image
- Push to Google Container Registry
- Deploy to Cloud Run at `https://financial-ner-api-<hash>-uc.a.run.app`

#### Manual Docker Build & Deploy:

```bash
# Build from project root
docker build -f backend/Dockerfile -t gcr.io/YOUR_PROJECT_ID/financial-ner-api .

# Push to GCR
docker push gcr.io/YOUR_PROJECT_ID/financial-ner-api

# Deploy to Cloud Run
gcloud run deploy financial-ner-api \
  --image gcr.io/YOUR_PROJECT_ID/financial-ner-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10
```

### Option 2: Deploy to App Engine

App Engine provides a managed platform with simpler configuration.

From the `backend/` directory:

```bash
cd backend

# Copy the trained model into backend folder
cp -r ../financial_ner_model .

# Deploy using app.yaml
gcloud app deploy app.yaml

# View logs
gcloud app logs tail -s default
```

Your API will be available at: `https://YOUR_PROJECT_ID.appspot.com`

### Testing the Deployment

```bash
# Replace YOUR_API_URL with your Cloud Run or App Engine URL
export API_URL="https://financial-ner-api-xxx.a.run.app"

# Health check
curl $API_URL/health

# Get labels
curl $API_URL/labels

# Predict entities
curl -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple (AAPL) announced a $50 million acquisition of an AI startup."}'
```

### Cost Optimization

- **Cloud Run**: Free tier includes 2 million requests/month. Scales to zero when idle.
- **App Engine**: Always-on instances. Use `automatic_scaling` with `min_instances: 0` to reduce costs.
- **Model Size**: The spaCy model (~100MB) is included in the container. Consider using a smaller model for production if latency is critical.

### Monitoring

```bash
# Cloud Run logs
gcloud run services logs read financial-ner-api --region us-central1

# App Engine logs
gcloud app logs tail -s default
```
