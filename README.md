# Loan Repayment Predictor

Predict the likelihood of a loan being repaid using a machine learning model.

## Features
- Developed with FastAPI.
- Uses LightGBM for predictions.
- Docker-ready for deployment.

## Prerequisites
- Python 3.9+
- Docker (optional, for containerization)

## Getting Started

### Local Setup
1. **Clone**:
   ```sh
   git clone https://github.com/al-strunova/loan-repayment-ml-predictor.git

2. **Environment and Dependenciese**:
   ```sh
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3. **Run FastAPIe**:
   ```sh
   cd src
   uvicorn app:app --reload

### Docker Deployment
1. **Build Imagee**:
   ```sh
   docker build -t loan-repayment-predictor .
2. **Run Containere**:
   ```sh
   docker run -p 8000:8000 loan-repayment-predictor

Open http://localhost:8000/ to access the UI.

## Endpoints
- '/' : main page to interact with ui
- '/predict' : Get prediction on loan repayment

## Contributing
PRs are welcome. For major changes, open an issue first.

## License
MIT License
