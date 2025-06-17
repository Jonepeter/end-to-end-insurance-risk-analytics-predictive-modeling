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
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
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
        print(f"Preparing data for {task_type} task...")
        print(f"Original dataset shape: {df.shape}")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Feature Engineering
        df_processed = self._engineer_features(df_processed)
        
        # Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # Separate features and target
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # For classification tasks, ensure binary encoding
        if task_type == 'classification':
            y = (y > 0).astype(int)
        
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        print(f"Numeric features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if task_type == 'classification' else None
        )
        
        # Store feature names for later use
        self.feature_names = list(numeric_features) + list(categorical_features)
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def _engineer_features(self, df):
        """Engineer new features that might be relevant to insurance risk"""
        print("Engineering features...")
        
        # Convert TransactionMonth to datetime if it's not already
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
            df['TransactionYear'] = df['TransactionMonth'].dt.year
            df['TransactionMonthNum'] = df['TransactionMonth'].dt.month
            df['TransactionQuarter'] = df['TransactionMonth'].dt.quarter
        
        # Vehicle age (if RegistrationYear exists)
        if 'RegistrationYear' in df.columns:
            df['VehicleAge'] = 2024 - df['RegistrationYear']
            df['VehicleAge'] = df['VehicleAge'].clip(lower=0)  # Ensure non-negative
        
        # Premium per unit of sum insured
        if 'SumInsured' in df.columns and 'CalculatedPremiumPerTerm' in df.columns:
            df['PremiumToSumInsuredRatio'] = df['CalculatedPremiumPerTerm'] / (df['SumInsured'] + 1e-6)
        
        # Risk indicators
        if 'ExcessSelected' in df.columns:
            df['HasExcess'] = (df['ExcessSelected'] != 'No excess').astype(int)
        
        # Vehicle power to weight ratio (if available)
        if 'kilowatts' in df.columns and 'cubiccapacity' in df.columns:
            df['PowerToCapacityRatio'] = df['kilowatts'] / (df['cubiccapacity'] + 1e-6)
        
        # Geographic risk indicators
        if 'Province' in df.columns:
            # Create risk zones based on provinces (example mapping)
            high_risk_provinces = ['Gauteng', 'KwaZulu-Natal']
            df['HighRiskProvince'] = df['Province'].isin(high_risk_provinces).astype(int)
        
        # Vehicle type risk indicators
        if 'VehicleType' in df.columns:
            commercial_vehicles = ['Commercial Vehicle', 'Truck', 'Bus']
            df['IsCommercialVehicle'] = df['VehicleType'].isin(commercial_vehicles).astype(int)
        
        print(f"Feature engineering complete. New shape: {df.shape}")
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        print("Missing values handled.")
        return df
    
    def build_models(self, task_type='regression'):
        """Build models based on task type"""
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
    
    def train_models(self, X_train, y_train, task_type='regression'):
        """Train all models and store their performance"""
        print("Training models...")
        
        # Build models
        self.build_models(task_type)
        
        results = {}
        
        for name, model in self.models.items():
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
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2' if task_type == 'regression' else 'accuracy')
            print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def evaluate_models(self, models, X_test, y_test, task_type='regression'):
        """Evaluate all models using appropriate metrics"""
        print("Evaluating models...")
        
        results = {}
        
        for name, model in models.items():
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
        
        # Find best model
        if task_type == 'regression':
            best_model_name = min(results.keys(), key=lambda x: results[x]['RMSE'])
        else:
            best_model_name = max(results.keys(), key=lambda x: results[x]['F1-Score'])
        
        self.best_model = models[best_model_name]
        self.model_results = results
        
        print(f"\nBest model: {best_model_name}")
        
        return results, self.best_model
    
    def analyze_feature_importance(self, model, X_test, top_n=10):
        """Analyze feature importance using SHAP values"""
        print("Analyzing feature importance with SHAP...")
        
        # Get feature names after preprocessing
        feature_names = self.preprocessor.get_feature_names_out()
        
        # Get SHAP values
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
        
        # Create feature importance dataframe
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
        
        # Print top features with business interpretation
        print(f"\nTop {top_n} Most Important Features:")
        print("=" * 60)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"{i}. {row['Feature']}: {row['Importance']:.4f}")
        
        return shap_values, feature_names, importance_df
    
    def predict_risk_based_premium(self, claim_prob_model, severity_model, X, expense_loading=0.1, profit_margin=0.15):
        """
        Calculate risk-based premium using the formula:
        Premium = (Predicted Probability of Claim * Predicted Claim Severity) + Expense Loading + Profit Margin
        """
        # Predict claim probability
        claim_prob = claim_prob_model.predict_proba(X)[:, 1]
        
        # Predict claim severity
        claim_severity = severity_model.predict(X)
        
        # Calculate risk-based premium
        risk_premium = claim_prob * claim_severity
        total_premium = risk_premium * (1 + expense_loading + profit_margin)
        
        return total_premium, claim_prob, claim_severity
    
    def generate_model_report(self, results, task_type='regression'):
        """Generate a comprehensive model comparison report"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON REPORT")
        print("="*60)
        
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

def main():
    """Main function to demonstrate the modeling pipeline"""
    # This would be called from the notebook
    pass

if __name__ == "__main__":
    main() 