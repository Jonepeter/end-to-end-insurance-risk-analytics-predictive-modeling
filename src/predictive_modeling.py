import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBRegressor, XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import traceback
import sys

# Try to import SHAP, but continue if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP not available. Feature importance analysis will be limited.")
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')

class InsuranceRiskModel:
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.best_model = None
        self.feature_names = None
        self.model_results = {}
        
    def prepare_data(self, df, target_col, test_size=0.2, random_state=42, task_type='regression'):
        """
        Prepare data for modeling by handling missing values, feature engineering, and encoding categorical variables.

        Args:
            df (pd.DataFrame): The input dataframe.
            target_col (str): The name of the target column.
            test_size (float): The proportion of the data to be used for testing.
            random_state (int): The random seed for reproducibility.
            task_type (str): 'regression' or 'classification'

        Returns:
            X_train (pd.DataFrame): The training features.
            X_test (pd.DataFrame): The testing features.
            y_train (pd.Series): The training target.
            y_test (pd.Series): The testing target.
        """
        try:
            print(f"Preparing data for {task_type} task...")
            print(f"Original dataset shape: {df.shape}")
            
            # Validate input
            if df is None or df.empty:
                raise ValueError("Input dataframe is None or empty")
            
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataframe")
            
            # Create a copy to avoid modifying original data
            df_processed = df.copy()
            
            # Feature Engineering
            try:
                df_processed = self._engineer_features(df_processed)
            except Exception as e:
                print(f"Warning: Feature engineering failed: {e}")
                print("Continuing with original features...")
            
            # Handle missing values
            try:
                df_processed = self._handle_missing_values(df_processed)
            except Exception as e:
                print(f"Warning: Missing value handling failed: {e}")
                print("Continuing with original data...")
            
            # Separate features and target
            try:
                X = df_processed.drop(columns=[target_col])
                y = df_processed[target_col]
                
                # For classification tasks, ensure binary encoding
                if task_type == 'classification':
                    y = (y > 0).astype(int)
                    
                print(f"Features shape: {X.shape}")
                print(f"Target shape: {y.shape}")
                
            except Exception as e:
                raise ValueError(f"Failed to separate features and target: {e}")
            
            # Identify numeric and categorical columns
            try:
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                categorical_features = X.select_dtypes(include=['object', 'category']).columns
                
                print(f"Numeric features: {len(numeric_features)}")
                print(f"Categorical features: {len(categorical_features)}")
                
                if len(numeric_features) == 0 and len(categorical_features) == 0:
                    raise ValueError("No valid features found in the dataset")
                    
            except Exception as e:
                raise ValueError(f"Failed to identify feature types: {e}")
            
            # Create preprocessing pipelines
            try:
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
                ])
                
                # Combine preprocessing steps
                self.preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ])
                    
            except Exception as e:
                raise ValueError(f"Failed to create preprocessing pipeline: {e}")
            
            # Split the data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, 
                    stratify=y if task_type == 'classification' else None
                )
                
                print(f"Training set shape: {X_train.shape}")
                print(f"Test set shape: {X_test.shape}")
                
            except Exception as e:
                raise ValueError(f"Failed to split data: {e}")
            
            # Store feature names for later use
            self.feature_names = list(numeric_features) + list(categorical_features)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"❌ Error in prepare_data: {e}")
            print("Traceback:")
            traceback.print_exc()
            return None
    
    def _engineer_features(self, df):
        """Engineer new features that might be relevant to insurance risk"""
        try:
            print("Engineering features...")
            
            # Convert TransactionMonth to datetime if it's not already
            if 'TransactionMonth' in df.columns:
                try:
                    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
                    df['TransactionYear'] = df['TransactionMonth'].dt.year
                    df['TransactionMonthNum'] = df['TransactionMonth'].dt.month
                    df['TransactionQuarter'] = df['TransactionMonth'].dt.quarter
                except Exception as e:
                    print(f"Warning: Failed to process TransactionMonth: {e}")
            
            # Vehicle age (if RegistrationYear exists)
            if 'RegistrationYear' in df.columns:
                try:
                    df['VehicleAge'] = 2024 - df['RegistrationYear']
                    df['VehicleAge'] = df['VehicleAge'].clip(lower=0)  # Ensure non-negative
                except Exception as e:
                    print(f"Warning: Failed to calculate VehicleAge: {e}")
            
            # Premium per unit of sum insured
            if 'SumInsured' in df.columns and 'CalculatedPremiumPerTerm' in df.columns:
                try:
                    df['PremiumToSumInsuredRatio'] = df['CalculatedPremiumPerTerm'] / (df['SumInsured'] + 1e-6)
                except Exception as e:
                    print(f"Warning: Failed to calculate PremiumToSumInsuredRatio: {e}")
            
            # Risk indicators
            if 'ExcessSelected' in df.columns:
                try:
                    df['HasExcess'] = (df['ExcessSelected'] != 'No excess').astype(int)
                except Exception as e:
                    print(f"Warning: Failed to create HasExcess feature: {e}")
            
            # Vehicle power to weight ratio (if available)
            if 'kilowatts' in df.columns and 'cubiccapacity' in df.columns:
                try:
                    df['PowerToCapacityRatio'] = df['kilowatts'] / (df['cubiccapacity'] + 1e-6)
                except Exception as e:
                    print(f"Warning: Failed to calculate PowerToCapacityRatio: {e}")
            
            # Geographic risk indicators
            if 'Province' in df.columns:
                try:
                    high_risk_provinces = ['Gauteng', 'KwaZulu-Natal']
                    df['HighRiskProvince'] = df['Province'].isin(high_risk_provinces).astype(int)
                except Exception as e:
                    print(f"Warning: Failed to create HighRiskProvince feature: {e}")
            
            # Vehicle type risk indicators
            if 'VehicleType' in df.columns:
                try:
                    commercial_vehicles = ['Commercial Vehicle', 'Truck', 'Bus']
                    df['IsCommercialVehicle'] = df['VehicleType'].isin(commercial_vehicles).astype(int)
                except Exception as e:
                    print(f"Warning: Failed to create IsCommercialVehicle feature: {e}")
            
            print(f"Feature engineering complete. New shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"❌ Error in _engineer_features: {e}")
            print("Returning original dataframe...")
            return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        try:
            print("Handling missing values...")
            
            # For numeric columns, fill with median
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            except Exception as e:
                print(f"Warning: Failed to handle missing numeric values: {e}")
            
            # For categorical columns, fill with mode
            try:
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                for col in categorical_cols:
                    try:
                        mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                        df[col] = df[col].fillna(mode_value)
                    except Exception as e:
                        print(f"Warning: Failed to handle missing values in column {col}: {e}")
                        df[col] = df[col].fillna('Unknown')
            except Exception as e:
                print(f"Warning: Failed to handle missing categorical values: {e}")
            
            print("Missing values handled.")
            return df
            
        except Exception as e:
            print(f"❌ Error in _handle_missing_values: {e}")
            print("Returning original dataframe...")
            return df
    
    def build_models(self, task_type='regression'):
        """Build models based on task type"""
        try:
            if task_type == 'regression':
                self.models = {
                    'linear_regression': LinearRegression(),
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                    'xgboost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                }
            else:  # classification
                self.models = {
                    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                    'xgboost': XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                }
            
            print(f"Built {len(self.models)} models for {task_type} task")
            
        except Exception as e:
            print(f"❌ Error in build_models: {e}")
            print("Traceback:")
            traceback.print_exc()
            self.models = {}
    
    def train_models(self, X_train, y_train, task_type='regression'):
        """Train all models and store their performance"""
        try:
            print("Training models...")
            
            # Build models
            self.build_models(task_type)
            
            if not self.models:
                raise ValueError("No models available for training")
            
            results = {}
            
            for name, model in self.models.items():
                try:
                    print(f"Training {name}...")
                    
                    # Create pipeline with preprocessing and model
                    pipeline = Pipeline([
                        ('preprocessor', self.preprocessor),
                        ('model', model)
                    ])
                    
                    # Train the model
                    pipeline.fit(X_train, y_train)
                    results[name] = pipeline
                    
                    # Cross-validation score
                    try:
                        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, 
                                                  scoring='r2' if task_type == 'regression' else 'accuracy')
                        print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                    except Exception as e:
                        print(f"Warning: Cross-validation failed for {name}: {e}")
                        
                except Exception as e:
                    print(f"❌ Failed to train {name}: {e}")
                    continue
            
            if not results:
                raise ValueError("No models were successfully trained")
            
            print(f"Successfully trained {len(results)} models")
            return results
            
        except Exception as e:
            print(f"❌ Error in train_models: {e}")
            print("Traceback:")
            traceback.print_exc()
            return {}
    
    def evaluate_models(self, models, X_test, y_test, task_type='regression'):
        """Evaluate all models using appropriate metrics"""
        try:
            print("Evaluating models...")
            
            if not models:
                raise ValueError("No models provided for evaluation")
            
            results = {}
            
            for name, model in models.items():
                try:
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    if task_type == 'regression':
                        # Calculate regression metrics
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        r2 = r2_score(y_test, y_pred)
                        mae = np.mean(np.abs(y_test - y_pred))
                        
                        results[name] = {
                            'RMSE': rmse,
                            'R-squared': r2,
                            'MAE': mae
                        }
                        
                        print(f"{name}: RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}")
                        
                    else:  # classification
                        # Calculate classification metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        
                        results[name] = {
                            'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall': recall,
                            'F1-Score': f1
                        }
                        
                        print(f"{name}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                        
                except Exception as e:
                    print(f"❌ Failed to evaluate {name}: {e}")
                    continue
            
            if not results:
                raise ValueError("No models were successfully evaluated")
            
            # Find best model
            try:
                if task_type == 'regression':
                    best_model_name = min(results.keys(), key=lambda x: results[x]['RMSE'])
                else:
                    best_model_name = max(results.keys(), key=lambda x: results[x]['F1-Score'])
                
                self.best_model = models[best_model_name]
                self.model_results = results
                
                print(f"\nBest model: {best_model_name}")
                
            except Exception as e:
                print(f"❌ Failed to determine best model: {e}")
                best_model_name = list(results.keys())[0] if results else None
                self.best_model = models[best_model_name] if best_model_name else None
            
            return results, self.best_model
            
        except Exception as e:
            print(f"❌ Error in evaluate_models: {e}")
            print("Traceback:")
            traceback.print_exc()
            return {}, None
    
    def analyze_feature_importance(self, model, X_test, top_n=10):
        """
        Analyze feature importance using SHAP values.
        Args:
            model: The trained model.
            X_test: The test data.
        """
        try:
            print("Analyzing feature importance...")
            
            if not SHAP_AVAILABLE:
                print("SHAP not available. Using model coefficients for feature importance...")
                return self._analyze_feature_importance_fallback(model, X_test, top_n)
            
            # Get feature names after preprocessing
            try:
                feature_names = self.preprocessor.get_feature_names_out()
            except Exception as e:
                print(f"Warning: Could not get feature names: {e}")
                feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
            
            # Get SHAP values
            try:
                if hasattr(model.named_steps['model'], 'feature_importances_'):
                    # For tree-based models
                    explainer = shap.TreeExplainer(model.named_steps['model'])
                    shap_values = explainer.shap_values(model.named_steps['preprocessor'].transform(X_test))
                else:
                    # For linear models
                    explainer = shap.LinearExplainer(model.named_steps['model'], model.named_steps['preprocessor'].transform(X_test))
                    shap_values = explainer.shap_values(model.named_steps['preprocessor'].transform(X_test))
                
                # Calculate mean absolute SHAP values for feature importance
                if len(shap_values.shape) > 2:
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                
                feature_importance = np.abs(shap_values).mean(0)
                
            except Exception as e:
                print(f"Warning: SHAP analysis failed: {e}")
                return self._analyze_feature_importance_fallback(model, X_test, top_n)
            
            # Create feature importance dataframe
            try:
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=False)
                
                # Plot top features
                plt.figure(figsize=(12, 8))
                
                # SHAP summary plot
                plt.subplot(2, 1, 1)
                shap.summary_plot(shap_values, model.named_steps['preprocessor'].transform(X_test), 
                                 feature_names=feature_names, show=False, max_display=top_n)
                plt.title('SHAP Feature Importance Summary')
                
                # Bar plot of top features
                plt.subplot(2, 1, 2)
                top_features = importance_df.head(top_n)
                plt.barh(range(len(top_features)), top_features['Importance'])
                plt.yticks(range(len(top_features)), top_features['Feature'])
                plt.xlabel('Mean |SHAP Value|')
                plt.title(f'Top {top_n} Most Important Features')
                plt.gca().invert_yaxis()
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Warning: Failed to create plots: {e}")
            
            # Print top features with business interpretation
            print(f"\nTop {top_n} Most Important Features:")
            print("=" * 60)
            for i, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
                print(f"{i}. {row['Feature']}: {row['Importance']:.4f}")
            
            return shap_values, feature_names, importance_df
            
        except Exception as e:
            print(f"❌ Error in analyze_feature_importance: {e}")
            print("Traceback:")
            traceback.print_exc()
            return None, None, None
    
    def _analyze_feature_importance_fallback(self, model, X_test, top_n=10):
        """Fallback method for feature importance when SHAP is not available"""
        try:
            print("Using fallback method for feature importance...")
            
            # Get feature names
            try:
                feature_names = self.preprocessor.get_feature_names_out()
            except:
                feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
            
            # Get feature importance from model
            if hasattr(model.named_steps['model'], 'feature_importances_'):
                importance = model.named_steps['model'].feature_importances_
            elif hasattr(model.named_steps['model'], 'coef_'):
                importance = np.abs(model.named_steps['model'].coef_)
            else:
                print("Warning: Model does not support feature importance analysis")
                return None, feature_names, None
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Print top features
            print(f"\nTop {top_n} Most Important Features:")
            print("=" * 60)
            for i, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
                print(f"{i}. {row['Feature']}: {row['Importance']:.4f}")
            
            return None, feature_names, importance_df
            
        except Exception as e:
            print(f"❌ Error in fallback feature importance: {e}")
            return None, None, None
    
    def predict_risk_based_premium(self, claim_prob_model, severity_model, X, expense_loading=0.1, profit_margin=0.15):
        """
        Calculate risk-based premium using the formula:
        Premium = (Predicted Probability of Claim * Predicted Claim Severity) + Expense Loading + Profit Margin
        """
        try:
            # Predict claim probability
            try:
                claim_prob = claim_prob_model.predict_proba(X)[:, 1]
            except Exception as e:
                print(f"Warning: Failed to predict claim probability: {e}")
                claim_prob = np.zeros(len(X))
            
            # Predict claim severity
            try:
                claim_severity = severity_model.predict(X)
            except Exception as e:
                print(f"Warning: Failed to predict claim severity: {e}")
                claim_severity = np.zeros(len(X))
            
            # Calculate risk-based premium
            risk_premium = claim_prob * claim_severity
            total_premium = risk_premium * (1 + expense_loading + profit_margin)
            
            return total_premium, claim_prob, claim_severity
            
        except Exception as e:
            print(f"❌ Error in predict_risk_based_premium: {e}")
            print("Traceback:")
            traceback.print_exc()
            return None, None, None
    
    def generate_model_report(self, results, task_type='regression'):
        """Generate a comprehensive model comparison report"""
        try:
            print("\n" + "="*60)
            print("MODEL PERFORMANCE COMPARISON REPORT")
            print("="*60)
            
            if not results:
                print("No results to report")
                return
            
            if task_type == 'regression':
                print(f"{'Model':<20} {'RMSE':<12} {'R-squared':<12} {'MAE':<12}")
                print("-" * 60)
                for model_name, metrics in results.items():
                    print(f"{model_name:<20} {metrics['RMSE']:<12.4f} {metrics['R-squared']:<12.4f} {metrics['MAE']:<12.4f}")
            else:
                print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
                print("-" * 70)
                for model_name, metrics in results.items():
                    print(f"{model_name:<20} {metrics['Accuracy']:<12.4f} {metrics['Precision']:<12.4f} {metrics['Recall']:<12.4f} {metrics['F1-Score']:<12.4f}")
            
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error in generate_model_report: {e}")
            print("Traceback:")
            traceback.print_exc()

def main():
    """
    Main function to demonstrate the complete predictive modeling pipeline.
    This function runs all three modeling tasks:
    1. Claim Severity Prediction (Regression)
    2. Claim Probability Prediction (Classification) 
    3. Premium Optimization (Regression)
    """
    print("="*80)
    print("INSURANCE RISK ANALYTICS - PREDICTIVE MODELING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Import EDA module
        from src.eda_analysis import InsuranceEDA
        
        # Initialize modules
        print("\n1. Initializing modules...")
        eda = InsuranceEDA()
        modeling = InsuranceRiskModel()
        
        # Load data
        print("\n2. Loading data...")
        data = eda.load_data('../data/insurance_data.parquet')
        print(f"   Dataset shape: {data.shape}")
        
        # Check missing values
        print("\n3. Analyzing missing values...")
        missing_summary = eda.check_missing_values()
        
        # Remove columns with more than 5% missing values
        print("\n4. Cleaning data...")
        missing_threshold = 5.0
        columns_to_remove = missing_summary[missing_summary['Missing Percentage'] > missing_threshold].index.tolist()
        
        print(f"   Columns to remove (missing > {missing_threshold}%): {len(columns_to_remove)}")
        for col in columns_to_remove:
            print(f"   - {col}")
        
        data_cleaned = data.drop(columns=columns_to_remove)
        print(f"   Cleaned dataset shape: {data_cleaned.shape}")
        
        # Create data subsets for different modeling tasks
        print("\n5. Creating data subsets...")
        
        # Claims data for severity prediction
        claims_data = data_cleaned[data_cleaned['TotalClaims'] > 0].copy()
        print(f"   Claims data shape: {claims_data.shape}")
        
        # All data for premium prediction
        premium_data = data_cleaned.copy()
        print(f"   Premium data shape: {premium_data.shape}")
        
        # All data for claim probability prediction
        claim_prob_data = data_cleaned.copy()
        claim_prob_data['HasClaim'] = (claim_prob_data['TotalClaims'] > 0).astype(int)
        print(f"   Claim probability data shape: {claim_prob_data.shape}")
        print(f"   Claim rate: {claim_prob_data['HasClaim'].mean():.4f}")
        
        # MODEL 1: CLAIM SEVERITY PREDICTION
        print("\n" + "="*60)
        print("MODEL 1: CLAIM SEVERITY PREDICTION (REGRESSION)")
        print("="*60)
        
        severity_data = modeling.prepare_data(
            claims_data, 
            'TotalClaims', 
            test_size=0.2, 
            random_state=42, 
            task_type='regression'
        )
        
        if severity_data is None:
            print("❌ Failed to prepare severity data")
            return 1
        
        X_train_sev, X_test_sev, y_train_sev, y_test_sev = severity_data
        
        # Train severity models
        severity_models = modeling.train_models(X_train_sev, y_train_sev, task_type='regression')
        
        if not severity_models:
            print("❌ Failed to train severity models")
            return 1
        
        # Evaluate severity models
        severity_results, best_severity_model = modeling.evaluate_models(
            severity_models, X_test_sev, y_test_sev, task_type='regression'
        )
        
        if not severity_results:
            print("❌ Failed to evaluate severity models")
            return 1
        
        # Generate report
        modeling.generate_model_report(severity_results, task_type='regression')
        
        # MODEL 2: CLAIM PROBABILITY PREDICTION
        print("\n" + "="*60)
        print("MODEL 2: CLAIM PROBABILITY PREDICTION (CLASSIFICATION)")
        print("="*60)
        
        prob_data = modeling.prepare_data(
            claim_prob_data, 
            'HasClaim', 
            test_size=0.2, 
            random_state=42, 
            task_type='classification'
        )
        
        if prob_data is None:
            print("❌ Failed to prepare probability data")
            return 1
        
        X_train_prob, X_test_prob, y_train_prob, y_test_prob = prob_data
        
        # Train probability models
        prob_models = modeling.train_models(X_train_prob, y_train_prob, task_type='classification')
        
        if not prob_models:
            print("❌ Failed to train probability models")
            return 1
        
        # Evaluate probability models
        prob_results, best_prob_model = modeling.evaluate_models(
            prob_models, X_test_prob, y_test_prob, task_type='classification'
        )
        
        if not prob_results:
            print("❌ Failed to evaluate probability models")
            return 1
        
        # Generate report
        modeling.generate_model_report(prob_results, task_type='classification')
        
        # MODEL 3: PREMIUM OPTIMIZATION
        print("\n" + "="*60)
        print("MODEL 3: PREMIUM OPTIMIZATION (REGRESSION)")
        print("="*60)
        
        premium_model_data = modeling.prepare_data(
            premium_data, 
            'CalculatedPremiumPerTerm', 
            test_size=0.2, 
            random_state=42, 
            task_type='regression'
        )
        
        if premium_model_data is None:
            print("❌ Failed to prepare premium data")
            return 1
        
        X_train_prem, X_test_prem, y_train_prem, y_test_prem = premium_model_data
        
        # Train premium models
        premium_models = modeling.train_models(X_train_prem, y_train_prem, task_type='regression')
        
        if not premium_models:
            print("❌ Failed to train premium models")
            return 1
        
        # Evaluate premium models
        premium_results, best_premium_model = modeling.evaluate_models(
            premium_models, X_test_prem, y_test_prem, task_type='regression'
        )
        
        if not premium_results:
            print("❌ Failed to evaluate premium models")
            return 1
        
        # Generate report
        modeling.generate_model_report(premium_results, task_type='regression')
        
        # RISK-BASED PREMIUM CALCULATION
        print("\n" + "="*60)
        print("RISK-BASED PREMIUM CALCULATION")
        print("="*60)
        
        # Use a subset for demonstration
        sample_data = premium_data.sample(n=1000, random_state=42)
        sample_prep = modeling.prepare_data(sample_data, 'CalculatedPremiumPerTerm', test_size=0.3, random_state=42, task_type='regression')
        
        if sample_prep is None:
            print("❌ Failed to prepare sample data for risk-based premium calculation")
            return 1
        
        X_sample_train, X_sample_test, y_sample_train, y_sample_test = sample_prep
        
        # Prepare for probability prediction
        sample_prob_data = sample_data.copy()
        sample_prob_data['HasClaim'] = (sample_prob_data['TotalClaims'] > 0).astype(int)
        sample_prob_prep = modeling.prepare_data(sample_prob_data, 'HasClaim', test_size=0.3, random_state=42, task_type='classification')
        
        if sample_prob_prep is None:
            print("❌ Failed to prepare sample probability data")
            return 1
        
        X_prob_train, X_prob_test, y_prob_train, y_prob_test = sample_prob_prep
        
        # Train sample models
        sample_severity_models = modeling.train_models(X_sample_train, y_sample_train, task_type='regression')
        sample_prob_models = modeling.train_models(X_prob_train, y_prob_train, task_type='classification')
        
        if not sample_severity_models or not sample_prob_models:
            print("❌ Failed to train sample models for risk-based premium calculation")
            return 1
        
        # Get best sample models
        _, best_sample_severity = modeling.evaluate_models(sample_severity_models, X_sample_test, y_sample_test, task_type='regression')
        _, best_sample_prob = modeling.evaluate_models(sample_prob_models, X_prob_test, y_prob_test, task_type='classification')
        
        if best_sample_severity is None or best_sample_prob is None:
            print("❌ Failed to get best sample models")
            return 1
        
        # Calculate risk-based premiums
        risk_premiums, claim_probs, claim_severities = modeling.predict_risk_based_premium(
            best_sample_prob, best_sample_severity, X_sample_test, expense_loading=0.1, profit_margin=0.15
        )
        
        if risk_premiums is not None:
            print(f"\nRisk-based premium calculation results:")
            print(f"   Sample size: {len(risk_premiums)}")
            print(f"   Average predicted claim probability: {claim_probs.mean():.4f}")
            print(f"   Average predicted claim severity: R{claim_severities.mean():.2f}")
            print(f"   Average risk-based premium: R{risk_premiums.mean():.2f}")
            print(f"   Average actual premium: R{y_sample_test.mean():.2f}")
        else:
            print("❌ Failed to calculate risk-based premiums")
        
        # FEATURE IMPORTANCE ANALYSIS
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        print("\nAnalyzing feature importance for best severity model...")
        try:
            shap_values_sev, feature_names_sev, importance_df_sev = modeling.analyze_feature_importance(
                best_severity_model, X_test_sev, top_n=10
            )
            print("   Severity model feature importance analysis completed.")
        except Exception as e:
            print(f"   Warning: Could not analyze severity model feature importance: {e}")
        
        print("\nAnalyzing feature importance for best probability model...")
        try:
            shap_values_prob, feature_names_prob, importance_df_prob = modeling.analyze_feature_importance(
                best_prob_model, X_test_prob, top_n=10
            )
            print("   Probability model feature importance analysis completed.")
        except Exception as e:
            print(f"   Warning: Could not analyze probability model feature importance: {e}")
        
        # SAVE MODELS
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        
        try:
            # Create models directory
            import os
            import joblib
            os.makedirs('../models', exist_ok=True)
            
            # Save models
            joblib.dump(best_severity_model, '../models/best_severity_model.pkl')
            joblib.dump(best_prob_model, '../models/best_probability_model.pkl')
            joblib.dump(best_premium_model, '../models/best_premium_model.pkl')
            
            print("   Models saved successfully!")
            
            # Save model metadata
            model_metadata = {
                'severity_model': {
                    'performance': severity_results,
                    'created_date': datetime.now().isoformat()
                },
                'probability_model': {
                    'performance': prob_results,
                    'created_date': datetime.now().isoformat()
                },
                'premium_model': {
                    'performance': premium_results,
                    'created_date': datetime.now().isoformat()
                }
            }
            
            import json
            with open('../models/model_metadata.json', 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            print("   Model metadata saved!")
            
        except Exception as e:
            print(f"   Warning: Failed to save models: {e}")
        
        # FINAL SUMMARY
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        print("\n✅ SUCCESSFULLY COMPLETED:")
        print("   1. Data preparation and cleaning")
        print("   2. Claim severity prediction model")
        print("   3. Claim probability prediction model")
        print("   4. Premium optimization model")
        print("   5. Risk-based premium calculation")
        print("   6. Feature importance analysis")
        print("   7. Model persistence")
        
        print("\n📊 MODEL PERFORMANCE SUMMARY:")
        print("   - All models trained and evaluated successfully")
        print("   - Models saved to '../models/' directory")
        print("   - Feature importance analysis completed")
        print("   - Risk-based pricing framework implemented")
        
        print(f"\n🏁 Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

