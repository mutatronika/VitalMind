#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blood Pressure Prediction - Main Script

This script demonstrates the complete workflow for predicting blood pressure
using machine learning models. It includes:
    - Synthetic data generation (since we don't have real data yet)
    - Data processing and feature engineering
    - Model training (Linear Regression and XGBoost)
    - Model evaluation and comparison
    
Usage:
    python main.py

Requirements:
    - All dependencies listed in requirements.txt
    - The following project modules:
        - data_processing.py
        - model_training.py
        - model_evaluation.py
"""

import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
import argparse
from datetime import datetime
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any

# Import our custom modules
from data_processing import (
    load_data, feature_engineering, identify_feature_types,
    create_preprocessing_pipeline, prepare_data
)
from model_training import (
    LinearBPModel, XGBoostBPModel, cross_validate_model
)
import model_evaluation as eval

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("blood_pressure_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('blood_pressure.main')

def generate_synthetic_data(
    n_samples: int = 1000,
    output_file: Optional[str] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic blood pressure data with realistic features and relationships.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate, by default 1000
    output_file : str, optional
        Path to save the generated data, by default None
    random_state : int, optional
        Random seed for reproducibility, by default 42
        
    Returns
    -------
    pd.DataFrame
        Generated synthetic data
    """
    try:
        logger.info(f"Generating synthetic blood pressure data with {n_samples} samples...")
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        # Generate demographic features
        age = np.random.normal(45, 15, n_samples).clip(18, 90)
        gender = np.random.choice(['Male', 'Female'], n_samples)
        height_cm = np.where(
            gender == 'Male',
            np.random.normal(175, 8, n_samples),  # Male heights
            np.random.normal(162, 7, n_samples)   # Female heights
        ).clip(145, 200)
        weight_kg = np.where(
            gender == 'Male',
            np.random.normal(80, 15, n_samples),  # Male weights
            np.random.normal(65, 12, n_samples)   # Female weights
        ).clip(40, 150)
        
        # Calculate BMI (used for blood pressure calculation)
        bmi = weight_kg / ((height_cm / 100) ** 2)
        
        # Generate lifestyle factors
        smoking = np.random.choice(['Never', 'Former', 'Current'], n_samples, p=[0.6, 0.2, 0.2])
        # Convert to numeric for easier calculation
        smoking_numeric = np.where(smoking == 'Never', 0, np.where(smoking == 'Former', 1, 2))
        
        exercise_hours_per_week = np.random.exponential(3, n_samples).clip(0, 15)
        stress_level = np.random.randint(1, 11, n_samples)  # 1-10 scale
        
        # Generate health indicators
        heart_rate = np.random.normal(75, 10, n_samples).clip(50, 120)
        cholesterol = np.random.normal(200, 40, n_samples).clip(120, 300)
        glucose = np.random.normal(90, 15, n_samples).clip(70, 200)
        diabetes = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # 0=No, 1=Yes
        family_history_hypertension = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 0=No, 1=Yes
        previous_cardiovascular_condition = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])  # 0=No, 1=Yes
        
        # Generate medication feature
        on_medication = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 0=No, 1=Yes
        
        # Generate blood pressure based on the features
        # These coefficients approximate realistic relationships
        
        # Base values for systolic and diastolic
        base_systolic = 120
        base_diastolic = 80
        
        # Age effect (increases with age)
        age_effect_systolic = 0.4 * (age - 30)
        age_effect_diastolic = 0.25 * (age - 30)
        
        # BMI effect (increases with higher BMI)
        bmi_effect_systolic = 1.5 * (bmi - 22.5)
        bmi_effect_diastolic = 1.0 * (bmi - 22.5)
        
        # Lifestyle effects
        lifestyle_effect_systolic = (
            3 * smoking_numeric +
            -0.5 * exercise_hours_per_week +
            1.0 * stress_level
        )
        lifestyle_effect_diastolic = (
            2 * smoking_numeric +
            -0.3 * exercise_hours_per_week +
            0.7 * stress_level
        )
        
        # Health indicator effects
        health_effect_systolic = (
            0.2 * (heart_rate - 75) +
            0.05 * (cholesterol - 200) +
            0.03 * (glucose - 90) +
            5 * diabetes +
            4 * family_history_hypertension +
            7 * previous_cardiovascular_condition
        )
        health_effect_diastolic = (
            0.1 * (heart_rate - 75) +
            0.03 * (cholesterol - 200) +
            0.02 * (glucose - 90) +
            3 * diabetes +
            2 * family_history_hypertension +
            4 * previous_cardiovascular_condition
        )
        
        # Medication effect (lowers BP if on medication)
        medication_effect_systolic = -8 * on_medication
        medication_effect_diastolic = -5 * on_medication
        
        # Gender effect (slightly higher in males)
        gender_effect_systolic = np.where(gender == 'Male', 2, 0)
        gender_effect_diastolic = np.where(gender == 'Male', 1, 0)
        
        # Calculate blood pressure
        systolic_bp = (
            base_systolic +
            age_effect_systolic +
            bmi_effect_systolic +
            lifestyle_effect_systolic +
            health_effect_systolic +
            medication_effect_systolic +
            gender_effect_systolic +
            np.random.normal(0, 8, n_samples)  # Random variation
        ).clip(90, 200)
        
        diastolic_bp = (
            base_diastolic +
            age_effect_diastolic +
            bmi_effect_diastolic +
            lifestyle_effect_diastolic +
            health_effect_diastolic +
            medication_effect_diastolic +
            gender_effect_diastolic +
            np.random.normal(0, 5, n_samples)  # Random variation
        ).clip(50, 120)
        
        # Convert to integers (as BP is typically measured)
        systolic_bp = np.round(systolic_bp).astype(int)
        diastolic_bp = np.round(diastolic_bp).astype(int)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'gender': gender,
            'height_cm': height_cm,
            'weight_kg': weight_kg,
            'smoking_status': smoking,
            'exercise_hours_per_week': exercise_hours_per_week,
            'stress_level': stress_level,
            'heart_rate': heart_rate,
            'cholesterol': cholesterol,
            'glucose': glucose,
            'diabetes': diabetes.astype(bool),
            'family_history_hypertension': family_history_hypertension.astype(bool),
            'previous_cardiovascular_condition': previous_cardiovascular_condition.astype(bool),
            'on_medication': on_medication.astype(bool),
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp
        })
        
        # Save data if output_file is provided
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            data.to_csv(output_file, index=False)
            logger.info(f"Synthetic data saved to {output_file}")
        
        return data
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        raise

def explore_data(data: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """
    Perform exploratory data analysis on the dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset to explore
    output_dir : str, optional
        Directory to save exploratory plots, by default None
    """
    try:
        logger.info("Performing exploratory data analysis...")
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 1. Summary statistics
        logger.info("\nData Summary:")
        logger.info(f"Shape: {data.shape}")
        summary = data.describe(include='all').T
        logger.info(f"\n{summary}")
        
        # 2. Blood pressure distribution
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(data['systolic_bp'], kde=True)
        plt.title('Systolic BP Distribution')
        plt.subplot(1, 2, 2)
        sns.histplot(data['diastolic_bp'], kde=True)
        plt.title('Diastolic BP Distribution')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'bp_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Correlation matrix
        numeric_cols = data.select_dtypes(include=['number']).columns
        correlation = data[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Key relationships
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Age vs BP
        sns.scatterplot(x='age', y='systolic_bp', data=data, alpha=0.5, ax=axes[0, 0])
        axes[0, 0].set_title('Age vs Systolic BP')
        
        # BMI vs BP
        data['bmi'] = data['weight_kg'] / ((data['height_cm']/100) ** 2)
        sns.scatterplot(x='bmi', y='systolic_bp', data=data, alpha=0.5, ax=axes[0, 1])
        axes[0, 1].set_title('BMI vs Systolic BP')
        
        # Heart Rate vs BP
        sns.scatterplot(x='heart_rate', y='systolic_bp', data=data, alpha=0.5, ax=axes[0, 2])
        axes[0, 2].set_title('Heart Rate vs Systolic BP')
        
        # Boxplots for categorical variables
        sns.boxplot(x='gender', y='systolic_bp', data=data, ax=axes[1, 0])
        axes[1, 0].set_title('Gender vs Systolic BP')
        
        sns.boxplot(x='diabetes', y='systolic_bp', data=data, ax=axes[1, 1])
        axes[1, 1].set_title('Diabetes vs Systolic BP')
        
        sns.boxplot(x='smoking_status', y='systolic_bp', data=data, ax=axes[1, 2])
        axes[1, 2].set_title('Smoking Status vs Systolic BP')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'key_relationships.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Exploratory data analysis completed.")
        if output_dir:
            logger.info(f"EDA plots saved to {output_dir}")
    
    except Exception as e:
        logger.error(f"Error during exploratory data analysis: {str(e)}")
        raise

def train_and_evaluate_models(
    data: pd.DataFrame,
    output_dir: str,
    target_columns: List[str] = ['systolic_bp', 'diastolic_bp'],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train and evaluate blood pressure prediction models.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset to use for training and evaluation
    output_dir : str
        Directory to save model outputs and evaluation results
    target_columns : List[str], optional
        List of target columns to predict, by default ['systolic_bp', 'diastolic_bp']
    test_size : float, optional
        Proportion of data to use for testing, by default 0.2
    random_state : int, optional
        Random seed for reproducibility, by default 42
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with trained models and evaluation results
    """
    try:
        # Create results directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Apply feature engineering
        logger.info("Applying feature engineering...")
        data_engineered = feature_engineering(data)
        
        # Identify feature types
        numerical_features, categorical_features, _ = identify_feature_types(
            data_engineered, target_columns=target_columns
        )
        
        # Create preprocessing pipeline
        logger.info("Creating preprocessing pipeline...")
        preprocessor = create_preprocessing_pipeline(
            numerical_features, categorical_features
        )
        
        # Split data into train and test sets
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = prepare_data(
            data_engineered, 
            target_columns=target_columns,
            test_size=test_size,
            random_state=random_state
        )
        
        # Initialize results dictionary
        results = {
            'models': {},
            'evaluations': {},
            'data': {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        }
        
        # Train and evaluate models for each target column
        for target in target_columns:
            logger.info(f"\n{'='*50}\nTraining models for {target}\n{'='*50}")
            
            models_dir = os.path.join(output_dir, 'models', target)
            os.makedirs(models_dir, exist_ok=True)
            
            eval_dir = os.path.join(output_dir, 'evaluations', target)
            os.makedirs(eval_dir, exist_ok=True)
            
            # Initialize models
            linear_model = LinearBPModel(target)
            xgb_model = XGBoostBPModel(target)
            
            # Build and train Linear Regression model
            logger.info("Training Linear Regression model...")
            linear_model.build(preprocessor)
            linear_model.fit(X_train, y_train)
            
            # Save model
            linear_model_path = linear_model.save(models_dir)
            
            # Build and train XGBoost model
            logger.info("Training XGBoost model...")
            xgb_model.build(preprocessor)
            xgb_model.fit(X_train, y_train)
            
            # Optional: Tune XGBoost hyperparameters
            # Commented out to save time, but can be uncommented for better performance
            # logger.info("Tuning XGBoost hyperparameters...")
            # best_params = xgb_model.tune_hyperparameters(X_train, y_train)
            # logger.info(f"Best parameters: {best_params}")
            
            # Save model
            xgb_model_path = xgb_model.save(models_dir)
            
            # Store models in results
            results['models'][target] = {
                'linear': linear_model,
                'xgboost': xgb_model
            }
            
            # Cross-validate models
            logger.info("Cross-validating models...")
            linear_cv = cross_validate_model(linear_model, X_train, y_train)
            xgb_cv = cross_validate_model(xgb_model, X_train, y_train)
            
            # Compare models
            logger.info("Comparing models...")
            models_to_compare = [linear_model, xgb_model]
            comparison_df = eval.compare_models(models_to_compare, X_test, y_test, save_dir=eval_dir)
            eval.plot_model_comparison(comparison_df, metric='RMSE', save_dir=eval_dir)
            eval.plot_model_comparison(comparison_df, metric='R²', save_dir=eval_dir)
            
            # Analyze residuals for both models
            logger.info("Analyzing residuals...")
            linear_pred, linear_residuals = eval.analyze_residuals(linear_model, X_test, y_test, save_dir=eval_dir)
            xgb_pred, xgb_residuals = eval.analyze_residuals(xgb_model, X_test, y_test, save_dir=eval_dir)
            
            # Plot feature importance
            logger.info("Plotting feature importance...")
            try:
                linear_importance = eval.plot_feature_importance(linear_model, top_n=10, save_dir=eval_dir)
            except Exception as e:
                logger.warning(f"Could not plot linear model feature importance: {str(e)}")
                linear_importance = None
                
            xgb_importance = eval.plot_feature_importance(xgb_model, top_n=10, save_dir=eval_dir)
            
            # Plot learning curves
            logger.info("Plotting learning curves...")
            linear_learning_curve = eval.plot_learning_curve(linear_model, X_train, y_train, save_dir=eval_dir)
            xgb_learning_curve = eval.plot_learning_curve(xgb_model, X_train, y_train, save_dir=eval_dir)
            
            # Plot error distributions
            logger.info("Plotting error distributions...")
            linear_error_stats = eval.plot_error_distribution(linear_model, X_test, y_test, save_dir=eval_dir)
            xgb_error_stats = eval.plot_error_distribution(xgb_model, X_test, y_test, save_dir=eval_dir)
            
            # Evaluate prediction intervals
            logger.info("Evaluating prediction intervals...")
            linear_intervals = eval.evaluate_prediction_intervals(linear_model, X_test, y_test, save_dir=eval_dir)
            xgb_intervals = eval.evaluate_prediction_intervals(xgb_model, X_test, y_test, save_dir=eval_dir)
            
            # Store evaluation results
            results['evaluations'][target] = {
                'linear': {
                    'cross_validation': linear_cv,
                    'predictions': linear_pred,
                    'residuals': linear_residuals,
                    'feature_importance': linear_importance,
                    'learning_curve': linear_learning_curve,
                    'error_stats': linear_error_stats,
                    'prediction_intervals': linear_intervals
                },
                'xgboost': {
                    'cross_validation': xgb_cv,
                    'predictions': xgb_pred,
                    'residuals': xgb_residuals,
                    'feature_importance': xgb_importance,
                    'learning_curve': xgb_learning_curve,
                    'error_stats': xgb_error_stats,
                    'prediction_intervals': xgb_intervals
                },
                'comparison': comparison_df
            }
        
        logger.info("\n" + "="*50 + "\nModel training and evaluation completed successfully!\n" + "="*50)
        logger.info(f"All results saved to {output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during model training and evaluation: {str(e)}")
        raise

def main():
    """
    Main function to run the blood pressure prediction pipeline.
    """
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Blood Pressure Prediction Pipeline')
        
        parser.add_argument('--data-file', type=str, default=None,
                            help='Path to existing data file (CSV). If not provided, synthetic data will be generated.')
        parser.add_argument('--output-dir', type=str, default='results',
                            help='Directory to save results (default: results)')
        parser.add_argument('--samples', type=int, default=1000,
                            help='Number of samples to generate if no data file is provided (default: 1000)')
        parser.add_argument('--test-size', type=float, default=0.2,
                            help='Proportion of data to use for testing (default: 0.2)')
        parser.add_argument('--random-state', type=int, default=42,
                            help='Random seed for reproducibility (default: 42)')
        parser.add_argument('--skip-eda', action='store_true',
                            help='Skip exploratory data analysis')
        
        args = parser.parse_args()
        
        # Create timestamp for results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_output_dir = args.output_dir
        output_dir = os.path.join(base_output_dir, f'run_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure file handler for this run
        file_handler = logging.FileHandler(os.path.join(output_dir, 'pipeline.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info("Starting Blood Pressure Prediction Pipeline")
        logger.info(f"Results will be saved to: {output_dir}")
        
        # Load or generate data
        if args.data_file:
            logger.info(f"Loading data from {args.data_file}...")
            data = load_data(args.data_file)
        else:
            logger.info(f"Generating synthetic data with {args.samples} samples...")
            data_file = os.path.join(output_dir, 'synthetic_data.csv')
            data = generate_synthetic_data(
                n_samples=args.samples,
                output_file=data_file,
                random_state=args.random_state
            )
        
        # Perform exploratory data analysis
        if not args.skip_eda:
            eda_dir = os.path.join(output_dir, 'eda')
            explore_data(data, output_dir=eda_dir)
        
        # Train and evaluate models
        model_results = train_and_evaluate_models(
            data=data,
            output_dir=output_dir,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        logger.info("Pipeline completed successfully!")
        
        return 0  # Success exit code
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
        return 1  # Error exit code

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
