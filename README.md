# Loan Repayment Predictor

Predict the likelihood of a loan being repaid using a machine learning model.

## Features
- Developed with FastAPI.
- Uses LightGBM for predictions.
- Docker-ready for deployment.

## Project Structure
- **notebooks**: Jupyter notebooks used for analysis and model experimentation.
  - `Predicting_Loan_Repayment_EDA.ipynb`: Conducts Exploratory Data Analysis (EDA) and establishes a baseline model.
  - `Feature_Preprocessing_&_Model_Analysis.ipynb`: Manages data preprocessing, feature engineering, model training, and the selection of the optimal algorithm.
- **data**: Directory containing all data files utilized throughout the project.
  > **Note**: Storing raw training data directly in the project repository is generally not a best practice, especially for large datasets or sensitive information. In this case, the data is included for demonstration and exploration purposes only.
- **src**: Central repository for scripts pivotal to the FastAPI service, prediction mechanics, and training procedures.
- **models**: Maintains the top-performing model alongside preprocessing pipelines, ensuring both reproducibility and ease of reapplication.
- **templates**: Consists of XML files tasked with rendering the user interface.

## Prerequisites
- Python 3.9+
- Docker (optional, for containerization)

## Getting Started

### Local Setup
1. **Clone**:
   ```sh
   git clone https://github.com/al-strunova/loan-repayment-ml-predictor.git

2. **Environment and Dependencies**:
   ```sh
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3. **Run FastAPI**:
   ```sh
   cd src
   uvicorn app:app --reload

### Docker Deployment
1. **Build Image**:
   ```sh
   docker build -t loan-repayment-predictor .
2. **Run Container**:
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
