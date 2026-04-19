### Part I: Model Implementation

Among the candidate models, both XGBoost and Logistic Regression showed comparable performance according to the original analysis, with no significant differences in the evaluation metrics. Given this, **Logistic Regression was selected** due to its simplicity, interpretability, and lower computational cost, which are desirable properties in a production environment. A simpler model reduces maintenance overhead and makes debugging and monitoring more straightforward.

Additionally, the model was trained using only the top 10 most important features, as this reduction did not negatively impact performance while improving efficiency.

Regarding class imbalance, the original implementation used manually computed class weights. This was replaced with `class_weight="balanced`, which provides an equivalent weighting scheme in a more robust and less error-prone way. This change simplifies the code while preserving the intended behavior of improving recall for the minority class.

Overall, the final selected model is a Logistic Regression trained on the top 10 features with automatic class balancing enabled.


### Part II: API Deployment with FastAPI

For this part, the model was deployed as an API using **FastAPI**.

#### Training and Model Storage

A `train.py` script was added to handle the training process. It trains the selected Logistic Regression model, serializes it with `joblib`, and uploads the artifact (`model.joblib`) to a GCS bucket. 

#### Model Loading

The model is loaded during the API startup:
- It is downloaded from GCS.
- Stored temporarily in `/tmp`.
- Loaded into memory using the `DelayModel` class.

This ensures low latency during inference, since the model is ready before handling requests.

#### API Design

The `/predict` endpoint:
- Parses the input into a pandas DataFrame.
- Applies preprocessing and generates predictions using the model.

Basic input validations were added (e.g., valid `MES`, `TIPOVUELO`, and `OPERA`) to ensure the API passes the tests. In a production environment, more comprehensive validation would be required.

#### Testing

For testing, the model loading and prediction were mocked to avoid dependency on GCS and ensure deterministic results:

```python
app.state.model = MagicMock()
app.state.model.predict.return_value = [0]
with TestClient(app) as client:
    self.client = client
```


### Part III: Deployment

For the final part, the API was deployed using **Google Cloud Run**, a fully managed service that allows running containerized applications without managing servers. It automatically handles scaling, networking, and infrastructure, making it well-suited for lightweight ML inference services.

#### Containerization

The application was containerized using Docker. Instead of manually building and pushing the image, the deployment was performed directly from the source code using the following command:

```bash
gcloud run deploy flight-delay-api \
    --project latamchallengerostagno \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars MODEL_BUCKET=bucket-latam-challenge
```
