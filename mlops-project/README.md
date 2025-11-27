# Iris Species Classification - MLOps Project

A production-ready machine learning application for classifying Iris species (setosa, versicolor, virginica) based on sepal and petal measurements. This project includes a trained model, FastAPI backend, modern web UI, and retraining pipeline.

## 📋 Project Structure

```
mlops-project/
├── api/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app with CORS and routers
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── predict.py       # /predict endpoint
│   │   │   └── retrain.py       # /retrain endpoint
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── model_service.py # Model loading and prediction
│   │   │   └── retrain_service.py # Retraining logic
│   │   └── utils.py             # Utilities and constants
│   ├── Dockerfile               # Docker configuration
│   └── requirements.txt         # Python dependencies
├── model/
│   ├── iris_model.pkl           # Trained LogisticRegression model
│   └── scaler.pkl               # StandardScaler for feature scaling
├── notebook/
│   └── Summative assignment - MLOP.ipynb  # Training notebook with evaluation
├── retrain/
│   └── new_data/                # Place CSV files here for retraining
├── ui/
│   ├── index.html               # Modern responsive web UI
│   └── assets/                  # Data visualization images
│       ├── class_distribution.png
│       ├── correlation.png
│       └── confusion_matrix.png
├── smoke_test.py                # API smoke test script
├── render.yaml                  # Render.com deployment config
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip
- (Optional) Docker for containerized deployment

### Local Development

1. **Install dependencies:**
   ```bash
   cd api
   pip install -r requirements.txt
   ```

2. **Run the API server:**
   ```bash
   # From the api/ directory
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   
   # Or from project root
   cd api && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Access the API:**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

4. **Open the UI:**
   - Open `ui/index.html` in a web browser
   - Set `API_URL` in the JavaScript code (line ~200) to `"http://localhost:8000"`

5. **Run smoke tests:**
   ```bash
   # Install requests if needed
   pip install requests
   
   # Run tests
   python smoke_test.py
   ```

## 🐳 Docker Deployment

### Build the Docker image:

```bash
# From the project root directory
docker build -f api/Dockerfile -t iris-classifier-api .
```

### Run the container:

```bash
docker run -p 8000:8000 iris-classifier-api
```

The API will be available at http://localhost:8000

## ☁️ Deploy to Render.com

1. **Push your code to GitHub**

2. **Create a new Web Service on Render:**
   - Connect your GitHub repository
   - Set the following:
     - **Build Command:** `pip install -r api/requirements.txt`
     - **Start Command:** `cd api && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
     - **Environment:** Python 3
     - **Python Version:** 3.11

3. **Alternative: Use render.yaml**
   - The project includes a `render.yaml` file
   - Render will automatically detect and use it

4. **Update UI API_URL:**
   - After deployment, update `API_URL` in `ui/index.html` to your Render URL
   - Or host the UI on a static hosting service (GitHub Pages, Netlify, etc.)

## 📡 API Endpoints

### `POST /predict`

Predict the Iris species from measurements.

**Request:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Prediction completed successfully",
  "data": {
    "label": "setosa",
    "label_id": 0,
    "probability": 0.997
  }
}
```

### `POST /retrain`

Retrain the model with new CSV data.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: CSV file with columns: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `target`

**CSV Format Example:**
```csv
sepal_length,sepal_width,petal_length,petal_width,target
5.1,3.5,1.4,0.2,0
6.0,2.2,4.0,1.0,1
6.5,3.0,5.2,2.0,2
```

**Response:**
```json
{
  "status": "success",
  "message": "Model retrained successfully",
  "data": {
    "metrics": {
      "accuracy": 0.98,
      "precision": 0.97,
      "recall": 0.98,
      "f1_score": 0.97
    },
    "best_params": {"C": 10},
    "classification_report": "...",
    "confusion_matrix": [[...], [...], [...]],
    "samples": {
      "train": 120,
      "test": 30,
      "total": 150
    }
  }
}
```

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### `GET /`

API information endpoint.

## 🧪 Testing

### Smoke Test

Run the included smoke test script:

```bash
python smoke_test.py
```

Make sure to update `API_URL` in the script if testing against a deployed endpoint.

### Manual Testing

1. **Test prediction:**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
     }'
   ```

2. **Test retraining:**
   ```bash
   curl -X POST "http://localhost:8000/retrain" \
     -F "file=@retrain/new_data/your_data.csv"
   ```

## 📊 Model Information

- **Algorithm:** Logistic Regression with GridSearchCV
- **Preprocessing:** StandardScaler
- **Features:** sepal_length, sepal_width, petal_length, petal_width
- **Classes:** 
  - 0 = setosa
  - 1 = versicolor
  - 2 = virginica
- **Evaluation Metrics:** Accuracy, Precision (macro), Recall (macro), F1-score (macro)

## 🔄 Retraining Workflow

1. **Prepare CSV file:**
   - Must include columns: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `target`
   - Target values must be 0, 1, or 2
   - All values must be numeric

2. **Upload via API:**
   - Use the `/retrain` endpoint or the web UI
   - The uploaded file is saved to `retrain/new_data/` with a timestamp

3. **Model update:**
   - New model combines base Iris dataset with uploaded data
   - Model and scaler are retrained and saved to `model/`
   - Evaluation metrics are computed on a holdout test set

## 📝 Model Persistence Notes

**Current Implementation:**
- Model files (`iris_model.pkl`, `scaler.pkl`) are stored in the `model/` directory
- This is suitable for development and assignment purposes

**Production Recommendations:**

For production deployments, consider:

1. **Cloud Storage (S3, GCS, Azure Blob):**
   ```python
   import boto3
   
   s3 = boto3.client('s3')
   s3.upload_file('model/iris_model.pkl', 'my-bucket', 'models/iris_model.pkl')
   ```

2. **Model Registry (MLflow, Weights & Biases):**
   - Track model versions
   - Store metadata and metrics
   - Enable model rollback

3. **Database:**
   - Store model metadata
   - Track retraining history
   - Version control

4. **Container Volume Mounts:**
   - For Docker deployments, use volumes for persistence
   - Example: `docker run -v $(pwd)/model:/app/model ...`

## 🎯 Demo Checklist

Before your video demo, verify:

- [ ] API starts successfully (`python -m uvicorn app.main:app`)
- [ ] `/health` endpoint returns `{"status": "healthy"}`
- [ ] `/predict` endpoint works with sample data
- [ ] UI loads and connects to API (set `API_URL`)
- [ ] Prediction form validates inputs and shows results
- [ ] Retrain form accepts CSV and shows metrics
- [ ] Data insights images display correctly
- [ ] Error handling works (test with invalid inputs)
- [ ] Smoke test passes: `python smoke_test.py`
- [ ] Docker build and run works (if using Docker)
- [ ] README instructions are clear and accurate

## 🛠️ Development

### Project Structure Rationale

- **Modular API:** Separated routes, services, and utilities for maintainability
- **Service Layer:** Business logic isolated from API endpoints
- **Type Hints:** Full type annotations for better IDE support and documentation
- **Error Handling:** Comprehensive error handling with informative messages
- **Logging:** Structured logging for debugging and monitoring

### Adding New Features

1. **New endpoint:** Add route in `api/app/routes/`
2. **New service:** Add service in `api/app/services/`
3. **Update dependencies:** Add to `api/requirements.txt`

## 📄 License

This project is for educational purposes as part of an MLOps course assignment.

## 🤝 Contributing

This is a course project. For questions or issues, please refer to your course materials.

---

**Built with:** FastAPI, scikit-learn, pandas, Tailwind CSS

