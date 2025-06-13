#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import logging as logger


class InsuranceEDA:
    def __init__(self):
        """
        Initialize the EDA analysis class.
        """
        
        # Initialize dataframe
        self.data = None
        
        # Display settings
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        
        
    def load_data(self, data_path) -> None:
        """Load and perform initial data inspection."""
        logger.info("Loading data...")
        self.data = pd.read_csv(data_path, sep='|', encoding='utf-8', low_memory=False)
        
        return self.data

    def dataset_info(self):
        """Display dataset information."""
        try:
            #dimmensions of the dataset 
            print(f"Dataset Shape: {self.data.shape}")
            #display data types and missing values
            print("\nData Types:")
            print(self.data.info())
                   
            
        except Exception as e:
            logger.error(f"Error displaying dataset information: {str(e)}")
            raise
    def handle_missing_values(self):
        """
        Handle missing values using appropriate strategies based on data type and context.
        Uses a combination of techniques:
        1. KNN imputation for numerical variables
        2. Mode imputation for categorical variables
        3. Forward/Backward fill for time series data
        4. Domain-specific imputation for insurance-specific fields
        """
        try:
            from sklearn.impute import KNNImputer
            from sklearn.preprocessing import LabelEncoder
            
            # Display initial missing values
            print("\nInitial Missing Values:")
            missing_values = self.data.isnull().sum().sort_values(ascending=False)
            print(missing_values)
            
            # Create a copy of the data for comparison
            original_data = self.data.copy()
            
            # 1. Handle Time Series Data (if exists)
            date_columns = self.data.select_dtypes(include=['datetime64']).columns
            for col in date_columns:
                if self.data[col].isnull().sum() > 0:
                    # Forward fill for dates
                    self.data[col] = self.data[col].fillna(method='ffill')
                    # Backward fill for any remaining NaNs at the start
                    self.data[col] = self.data[col].fillna(method='bfill')
            
            # 2. Handle Insurance-Specific Fields
            insurance_specific_cols = {
                'TotalPremium': 'mean',  # Use mean for premium
                'TotalClaims': 0,        # Use 0 for claims (assuming no claim)
                'CustomValueEstimate': 'median',  # Use median for vehicle value
                'PolicyNumber': 'ffill',  # Forward fill for policy numbers
                'CoverType': 'mode',     # Use mode for cover type
                'VehicleType': 'mode',   # Use mode for vehicle type
                'Make': 'mode',          # Use mode for vehicle make
                'Model': 'mode'          # Use mode for vehicle model
            }
            
            for col, strategy in insurance_specific_cols.items():
                if col in self.data.columns and self.data[col].isnull().sum() > 0:
                    if strategy == 'mean':
                        self.data[col] = self.data[col].fillna(self.data[col].mean())
                    elif strategy == 'median':
                        self.data[col] = self.data[col].fillna(self.data[col].median())
                    elif strategy == 'mode':
                        self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
                    elif strategy == 'ffill':
                        self.data[col] = self.data[col].fillna(method='ffill')
                    else:
                        self.data[col] = self.data[col].fillna(strategy)
            
            # 3. Handle Numerical Variables using KNN
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            numerical_cols = [col for col in numerical_cols if self.data[col].isnull().sum() > 0]
            
            if len(numerical_cols) > 0:
                # Scale the numerical data before KNN imputation
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(self.data[numerical_cols])
                
                # Apply KNN imputation
                imputer = KNNImputer(n_neighbors=5, weights='distance')
                imputed_data = imputer.fit_transform(scaled_data)
                
                # Transform back to original scale
                self.data[numerical_cols] = scaler.inverse_transform(imputed_data)
            
            # 4. Handle Categorical Variables
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            categorical_cols = [col for col in categorical_cols if self.data[col].isnull().sum() > 0]
            
            for col in categorical_cols:
                # For categorical variables with high cardinality, use 'Unknown' category
                if self.data[col].nunique() > 10:
                    self.data[col] = self.data[col].fillna('Unknown')
                else:
                    # For low cardinality, use mode
                    self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
            
            # 5. Validate the imputation
            print("\nMissing Values After Imputation:")
            print(self.data.isnull().sum().sort_values(ascending=False))
            
            # 6. Compare distributions before and after imputation
            print("\nDistribution Comparison (Before vs After Imputation):")
            for col in numerical_cols:
                print(f"\n{col}:")
                print("Before imputation:")
                print(original_data[col].describe())
                print("\nAfter imputation:")
                print(self.data[col].describe())
            
            logger.info("Missing values handled successfully using multiple strategies")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise

    def analyze_loss_ratio(self) -> None:
        """Analyze loss ratio across different dimensions."""
        logger.info("Analyzing loss ratio...")
        
        # Calculate overall Loss Ratio
        self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium']
        
        # Analyze by Province
        province_loss_ratio = self.df.groupby('Province')['LossRatio'].mean().sort_values(ascending=False)
        fig = px.bar(province_loss_ratio, 
                    title='Average Loss Ratio by Province',
                    labels={'value': 'Loss Ratio', 'Province': 'Province'})
        fig.write_html(self.output_dir / 'loss_ratio_by_province.html')
        
        # Analyze by Vehicle Type
        vehicle_loss_ratio = self.df.groupby('VehicleType')['LossRatio'].mean().sort_values(ascending=False)
        fig = px.bar(vehicle_loss_ratio,
                    title='Average Loss Ratio by Vehicle Type',
                    labels={'value': 'Loss Ratio', 'VehicleType': 'Vehicle Type'})
        fig.write_html(self.output_dir / 'loss_ratio_by_vehicle.html')
        
        # Analyze by Gender
        gender_loss_ratio = self.df.groupby('Gender')['LossRatio'].mean().sort_values(ascending=False)
        fig = px.bar(gender_loss_ratio,
                    title='Average Loss Ratio by Gender',
                    labels={'value': 'Loss Ratio', 'Gender': 'Gender'})
        fig.write_html(self.output_dir / 'loss_ratio_by_gender.html')
        
    def analyze_financial_variables(self) -> None:
        """Analyze distributions of financial variables and detect outliers."""
        logger.info("Analyzing financial variables...")
        
        # Create subplots for key financial variables
        fig = make_subplots(rows=2, cols=2,
                          subplot_titles=('Total Premium Distribution', 'Total Claims Distribution',
                                        'Custom Value Estimate Distribution', 'Loss Ratio Distribution'))
        
        # Add histograms
        fig.add_trace(go.Histogram(x=self.df['TotalPremium'], name='Total Premium'), row=1, col=1)
        fig.add_trace(go.Histogram(x=self.df['TotalClaims'], name='Total Claims'), row=1, col=2)
        fig.add_trace(go.Histogram(x=self.df['CustomValueEstimate'], name='Custom Value'), row=2, col=1)
        fig.add_trace(go.Histogram(x=self.df['LossRatio'], name='Loss Ratio'), row=2, col=2)
        
        fig.update_layout(height=800, width=1200, title_text="Financial Variables Distribution")
        fig.write_html(self.output_dir / 'financial_distributions.html')
        
        # Box plots for outlier detection
        fig = make_subplots(rows=2, cols=2,
                          subplot_titles=('Total Premium', 'Total Claims',
                                        'Custom Value Estimate', 'Loss Ratio'))
        
        fig.add_trace(go.Box(y=self.df['TotalPremium'], name='Total Premium'), row=1, col=1)
        fig.add_trace(go.Box(y=self.df['TotalClaims'], name='Total Claims'), row=1, col=2)
        fig.add_trace(go.Box(y=self.df['CustomValueEstimate'], name='Custom Value'), row=2, col=1)
        fig.add_trace(go.Box(y=self.df['LossRatio'], name='Loss Ratio'), row=2, col=2)
        
        fig.update_layout(height=800, width=1200, title_text="Outlier Detection in Financial Variables")
        fig.write_html(self.output_dir / 'financial_outliers.html')
        
    def analyze_temporal_trends(self) -> None:
        """Analyze temporal trends in the data."""
        logger.info("Analyzing temporal trends...")
        
        # Convert date columns to datetime if they exist
        date_columns = self.df.select_dtypes(include=['object']).columns[
            self.df.select_dtypes(include=['object']).apply(lambda x: x.str.contains('\\d{4}-\\d{2}-\\d{2}').any())
        ]
        
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col])
        
        # Monthly trends analysis
        if 'PolicyStartDate' in self.df.columns:
            monthly_metrics = self.df.groupby(self.df['PolicyStartDate'].dt.to_period('M')).agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyNumber': 'count'
            }).reset_index()
            
            monthly_metrics['LossRatio'] = monthly_metrics['TotalClaims'] / monthly_metrics['TotalPremium']
            
            fig = make_subplots(rows=2, cols=1,
                              subplot_titles=('Monthly Claims and Premium', 'Monthly Loss Ratio'))
            
            fig.add_trace(go.Scatter(x=monthly_metrics['PolicyStartDate'], y=monthly_metrics['TotalClaims'],
                                   name='Total Claims'), row=1, col=1)
            fig.add_trace(go.Scatter(x=monthly_metrics['PolicyStartDate'], y=monthly_metrics['TotalPremium'],
                                   name='Total Premium'), row=1, col=1)
            fig.add_trace(go.Scatter(x=monthly_metrics['PolicyStartDate'], y=monthly_metrics['LossRatio'],
                                   name='Loss Ratio'), row=2, col=1)
            
            fig.update_layout(height=800, width=1200, title_text="Temporal Trends Analysis")
            fig.write_html(self.output_dir / 'temporal_trends.html')
            
    def analyze_vehicle_makes(self) -> None:
        """Analyze vehicle makes and their claim patterns."""
        logger.info("Analyzing vehicle makes...")
        
        vehicle_analysis = self.df.groupby('Make')['TotalClaims'].agg(['mean', 'sum', 'count']).sort_values('mean', ascending=False)
        
        # Save top and bottom 10 makes to CSV
        vehicle_analysis.head(10).to_csv(self.output_dir / 'top_10_vehicle_makes.csv')
        vehicle_analysis.tail(10).to_csv(self.output_dir / 'bottom_10_vehicle_makes.csv')
        
        # Visualize top 10 makes
        fig = px.bar(vehicle_analysis.head(10),
                    y='mean',
                    title='Top 10 Vehicle Makes by Average Claim Amount',
                    labels={'mean': 'Average Claim Amount', 'Make': 'Vehicle Make'})
        fig.write_html(self.output_dir / 'top_vehicle_makes.html')
        
    def analyze_geographic_patterns(self) -> None:
        """Analyze geographic patterns in the data."""
        logger.info("Analyzing geographic patterns...")
        
        # Analyze insurance cover type by province
        cover_type_by_province = pd.crosstab(self.df['Province'], self.df['CoverType'], normalize='index') * 100
        
        fig = px.bar(cover_type_by_province,
                    title='Insurance Cover Type Distribution by Province',
                    labels={'value': 'Percentage', 'Province': 'Province', 'CoverType': 'Cover Type'})
        fig.write_html(self.output_dir / 'cover_type_by_province.html')
        
        # Analyze average premium by province
        avg_premium_by_province = self.df.groupby('Province')['TotalPremium'].mean().sort_values(ascending=False)
        
        fig = px.bar(avg_premium_by_province,
                    title='Average Premium by Province',
                    labels={'value': 'Average Premium', 'Province': 'Province'})
        fig.write_html(self.output_dir / 'premium_by_province.html')
        
    def analyze_correlations(self) -> None:
        """Analyze correlations between numerical variables."""
        logger.info("Analyzing correlations...")
        
        # Calculate correlation matrix for numerical variables
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numerical_cols].corr()
        
        # Save correlation matrix to CSV
        correlation_matrix.to_csv(self.output_dir / 'correlation_matrix.csv')
        
        # Create correlation heatmap
        fig = px.imshow(correlation_matrix,
                       title='Correlation Matrix of Numerical Variables',
                       labels=dict(color='Correlation Coefficient'))
        fig.write_html(self.output_dir / 'correlation_matrix.html')
        
    
            
    