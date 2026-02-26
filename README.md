# Airlines Customer Analytics

## Overview

This repository contains a comprehensive analysis of airline customer satisfaction data using machine learning and statistical analysis. The project analyzes customer behavior, satisfaction levels, and various service factors to build predictive models and derive actionable insights for improving customer experience.

## Dataset

- **Size**: 129,880 customer records with 23 features
- **Source**: Invistico Airline dataset
- **Target Variable**: Customer satisfaction (binary: satisfied/not satisfied)

### Key Features

The dataset includes various customer and service-related attributes:

#### Demographic & Travel Information:
- **Age**: Customer age (range: 7-85 years, mean: ~39.4 years)
- **Gender**: Male/Female classification
- **Customer Type**: Loyal Customer vs. Disloyal Customer
- **Type of Travel**: Personal Travel vs. Business Travel
- **Class**: Ticket class (Eco, Business, Eco Plus)
- **Flight Distance**: Distance traveled (range: 50-6,951 miles, mean: ~1,981 miles)

#### Service Quality Ratings (0-5 scale):
- Seat comfort
- Departure/Arrival time convenience
- Food and drink
- Gate location
- Inflight wifi service
- Inflight entertainment
- Online support
- Ease of Online booking
- On-board service
- Leg room service
- Baggage handling
- Checkin service
- Cleanliness
- Online boarding

#### Operational Metrics:
- **Departure Delay in Minutes**: Departure time delays (mean: ~14.7 minutes)
- **Arrival Delay in Minutes**: Arrival time delays (mean: ~15.1 minutes)

## Project Workflow

### 1. Data Cleaning & Preprocessing
- Loaded 129,880 records with 23 features
- Identified 393 missing values in 'Arrival Delay in Minutes' column
- Imputed missing values using the column mean
- No other missing values detected after preprocessing

### 2. Feature Engineering
- **Target Variable Encoding**: Converted satisfaction from categorical ('satisfied'/'neutral or dissatisfied') to binary (1/0)
- **One-Hot Encoding**: Transformed categorical features:
  - Gender → Gender_Male
  - Customer Type → Customer Type_disloyal Customer
  - Type of Travel → Type of Travel_Personal Travel
  - Class → Class_Business, Class_Eco Plus

### 3. Data Standardization
- Applied **StandardScaler** to normalize all features
- Train-test split: 80% training (103,904 records), 20% testing (25,976 records)

### 4. Exploratory Data Analysis

#### Key Insights:
- **Satisfaction Distribution**: Majority of customers are satisfied
- **Age Distribution**: Mean age ~39.4 years with standard deviation of ~15.1 years
- **Delay Patterns**: Both departure and arrival delays show heavy right skew
  - Most flights (median) have 0 delay
  - Maximum delays observed: ~1,592 minutes (departure) and ~1,584 minutes (arrival)
- **Service Quality Variations**: Service ratings vary significantly by customer type and travel class

### 5. Machine Learning Models

#### Linear Probability Model (LPM)
- **Algorithm**: Linear Regression for binary classification
- **Predictions**: Probability outputs converted to binary class (threshold: 0.5)
- **Purpose**: Baseline model for customer satisfaction prediction

## Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
  - `LinearRegression`: For LPM modeling
  - `StandardScaler`: Feature normalization
  - `train_test_split`: Data partitioning
- **seaborn**: Data visualization
- **Jupyter Notebook**: Interactive analysis environment

## Code Structure

The analysis is organized in the following cells:

1. **Library Imports**: Essential packages for data science workflow
2. **Data Loading**: Read airline dataset from CSV
3. **Exploratory Analysis**: Statistical summary and visualization
4. **Data Cleaning**: Handle missing values
5. **Feature Encoding**: Prepare categorical variables
6. **Data Scaling**: Standardize features
7. **Model Training**: Train Linear Probability Model
8. **Model Evaluation**: Assess prediction performance

## Key Findings

- Customer satisfaction is influenced by multiple service quality factors
- Demographic characteristics (age, gender) play a role in satisfaction
- Travel delays significantly impact customer satisfaction
- Different customer types (loyal vs. disloyal) have distinct satisfaction patterns
- Business vs. personal travel creates different service expectations

## Model Performance

The Linear Probability Model provides baseline predictions for customer satisfaction classification. Performance metrics and detailed evaluation would be included in extended analysis.

## How to Use

### Prerequisites
```bash
pip install pandas scikit-learn seaborn jupyter matplotlib numpy
