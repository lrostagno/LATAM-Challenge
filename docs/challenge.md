# Software Engineer (ML & LLMs) Challenge


## Part I: Model Implementation

Among the candidate models, **Logistic Regression was selected** due to its simplicity, interpretability, and lower computational cost, which are highly desirable properties for an API-driven production environment. 

The model was trained using the top 10 most important features, which maintained performance while improving computational efficiency. To address class imbalance, `class_weight="balanced"` was utilized natively within the model, providing a robust weighting scheme that improves the minority class recall without the need for manual distribution calculations.

## Part II: API Deployment with FastAPI

For this part, the model was deployed as an API using **FastAPI**.

- **Training and Model Storage:** A `train.py` script was added to handle the training process. It trains the selected Logistic Regression model, serializes it with `joblib`, and uploads the artifact (`model.joblib`) to a GCS bucket.
- **Model Loading:** The model is downloaded from GCS and loaded into memory during the API's startup event. This ensures the model is ready before handling any traffic.
- **API Design & Validation:** The `/predict` endpoint parses the input into a Pandas DataFrame and applies basic data validations (e.g., verifying valid `MES`, `TIPOVUELO`, and `OPERA`). In a true production environment, more comprehensive validation would be implemented to strictly enforce schemas and valid data ranges for all 10 features.
- **Testing:** The tests were modified to mock the model loading and prediction processes. This isolates the API routing logic and ensures deterministic results without relying on external infrastructure:

```python
with patch("challenge.api.startup", return_value=None):
    self.client = TestClient(app)
    app.state.model = MagicMock()
    app.state.model.predict.return_value = [0]
```


## Part III: Deployment

The application was deployed using **Google Cloud Run**, providing a fully managed, scalable, serverless environment. 

The deployment leverages containerization and was executed directly from the source using the `gcloud run deploy` command. Required environment variables, such as the `MODEL_BUCKET` name, are injected dynamically at runtime to allow the API to fetch the latest model artifact from GCS.


## Part IV: CI/CD Implementation

A continuous integration and delivery pipeline was implemented using **GitHub Actions**. To ensure a structured and safe development lifecycle, **GitFlow** development practices were adopted throughout the project.

- **Continuous Integration (CI):** Triggered on every **Pull Request** to `develop` or `main`. The workflow sets up a `Python 3.11` environment, installs all application and testing dependencies via `pip`, and executes the `pytest` suite. This ensures that code integrity is validated and no regressions are introduced before merging.
- **Continuous Delivery (CD):** Triggered on every **push** to the `main` branch. It automates the production release by performing the following:
    1. **Authentication:** Securely connects to Google Cloud using Workload Identity Federation (WIF), eliminating the need for static JSON service account keys.
    2. **Automated Deployment:** Executes the `gcloud run deploy` command to build the container and update the `flight-delay-api` service. This ensures the live production environment is always synchronized with the verified state of the `main` branch.