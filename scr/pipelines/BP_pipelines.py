"""
Blood Pressure Prediction - Main Script

This script demonstrates the complete pipeline for blood pressure prediction.
It includes:
    1. Synthetic data generation
    2. Data preprocessing and feature engineering
    3. Model training (Linear Regression and XGBoost)
    4. Model evaluation
    5. Prediction on new cases

Modules used:
    - data_processing.py
    - model_training.py
    - model_evaluation.py
    - predict.py
    """

import os
import logging
from data_processing import (
    generate_synthetic_data,
    feature_engineering,
    identify_feature_types,
    create_preprocessing_pipeline,
    prepare_data
)
from model_training import LinearBPModel, XGBoostBPModel
from model_evaluation import evaluate_models
from predict import predict_new_cases

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('blood_pressure.main')

def main_pipeline(n_samples=200, target='systolic_bp'):
    try:
        logger.info("Starting blood pressure prediction pipeline...")

        # Step 1: Generate synthetic data
        data = generate_synthetic_data(n_samples=n_samples)
        logger.info("Synthetic data generated.")

        # Step 2: Feature engineering
        data = feature_engineering(data)

        # Step 3: Identify feature types
        numerical_features, categorical_features, _ = identify_feature_types(
            data, target_columns=[target]
        )

        # Step 4: Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(
            numerical_features, categorical_features
        )

        # Step 5: Prepare data (split)
        X_train, X_test, y_train, y_test = prepare_data(
            data, target_columns=[target], test_size=0.25, random_state=42
        )

        # Step 6: Train models
        linear_model = LinearBPModel(target)
        linear_model.build(preprocessor)
        linear_model.fit(X_train, y_train)

        xgb_model = XGBoostBPModel(target)
        xgb_model.build(preprocessor)
        xgb_model.fit(X_train, y_train)

        # Step 7: Evaluate models
        evaluate_models(linear_model, xgb_model, X_test, y_test)

        # Step 8: Predict on new cases
        predict_new_cases(linear_model, xgb_model, preprocessor)

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main_pipeline()
