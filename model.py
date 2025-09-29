"""
AWS SageMaker XGBoost Model Training Pipeline
==============================================
Complete end-to-end machine learning workflow using Amazon SageMaker
for binary classification with XGBoost algorithm.
"""

# ============================================================================
# STEP 1: IMPORT LIBRARIES AND CONFIGURE SAGEMAKER ENVIRONMENT
# ============================================================================

import boto3
import re
import sys
import math
import json
import os
import sagemaker
import urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get IAM role for SageMaker permissions
role = get_execution_role()

# Define S3 prefix for organizing data and model artifacts
prefix = 'sagemaker/DEMO-xgboost-dm'

# Get current AWS region
my_region = boto3.session.Session().region_name

# Retrieve XGBoost container image URI for the current region
xgboost_container = sagemaker.image_uris.retrieve("xgboost", my_region, "latest")

print(f"Success - SageMaker instance region: {my_region}, using container: {xgboost_container}")


# ============================================================================
# STEP 2: CREATE S3 BUCKET FOR DATA STORAGE
# ============================================================================

# IMPORTANT: Change this to a unique bucket name!
bucket_name = 'your-unique-s3-bucket-name'

# Create S3 resource object
s3 = boto3.resource('s3')

try:
    # US East 1 has different bucket creation syntax
    if my_region == 'us-east-1':
        s3.create_bucket(Bucket=bucket_name)
    else:
        # Specify location constraint for all other regions
        s3.create_bucket(
            Bucket=bucket_name, 
            CreateBucketConfiguration={'LocationConstraint': my_region}
        )
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error:', e)


# ============================================================================
# STEP 3: DOWNLOAD AND LOAD DATASET
# ============================================================================

# Download dataset from AWS public URL
try:
    urllib.request.urlretrieve(
        "https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv",
        "bank_clean.csv"
    )
    print('Success: downloaded bank_clean.csv.')
except Exception as e:
    print('Data load error:', e)

# Load CSV into pandas DataFrame
try:
    model_data = pd.read_csv('./bank_clean.csv', index_col=0)
    print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error:', e)


# ============================================================================
# STEP 4: SHUFFLE AND SPLIT DATA (70% TRAIN, 30% TEST)
# ============================================================================

# Shuffle data with random_state for reproducibility
# Split into 70% training and 30% testing
train_data, test_data = np.split(
    model_data.sample(frac=1, random_state=1729),
    [int(0.7 * len(model_data))]
)
print(train_data.shape, test_data.shape)


# ============================================================================
# STEP 5: PREPARE TRAINING DATA FOR SAGEMAKER
# ============================================================================

# IMPORTANT: SageMaker XGBoost requires target column FIRST, no headers
# Concatenate target column ('y_yes') with feature columns
pd.concat([
    train_data['y_yes'], 
    train_data.drop(['y_no', 'y_yes'], axis=1)
], axis=1).to_csv('train.csv', index=False, header=False)

# Upload training data to S3
boto3.Session().resource('s3').Bucket(bucket_name).Object(
    os.path.join(prefix, 'train/train.csv')
).upload_file('train.csv')

# Create TrainingInput object pointing to S3 location
s3_input_train = sagemaker.inputs.TrainingInput(
    s3_data=f's3://{bucket_name}/{prefix}/train', 
    content_type='csv'
)


# ============================================================================
# STEP 6: CONFIGURE SAGEMAKER SESSION AND XGBOOST ESTIMATOR
# ============================================================================

# Initialize SageMaker session
sess = sagemaker.Session()

# Create XGBoost estimator with training configuration
xgb = sagemaker.estimator.Estimator(
    xgboost_container,                    # Docker container image
    role,                                  # IAM role with permissions
    instance_count=1,                      # Number of training instances
    instance_type='ml.m4.xlarge',         # Instance type for training
    output_path=f's3://{bucket_name}/{prefix}/output',  # Model artifact location
    sagemaker_session=sess
)

# Set XGBoost hyperparameters
xgb.set_hyperparameters(
    max_depth=5,              # Maximum tree depth (prevents overfitting)
    eta=0.2,                  # Learning rate / step size
    gamma=4,                  # Minimum loss reduction for node split
    min_child_weight=6,       # Minimum sum of instance weight in child
    subsample=0.8,            # Fraction of samples used per tree
    silent=0,                 # Logging mode (0=print messages)
    objective='binary:logistic',  # Loss function for binary classification
    num_round=100             # Number of boosting rounds (trees)
)


# ============================================================================
# STEP 7: START TRAINING JOB
# ============================================================================

# Fit model to training data (this may take several minutes)
xgb.fit({'train': s3_input_train})


# ============================================================================
# STEP 8: DEPLOY MODEL AS REAL-TIME ENDPOINT
# ============================================================================

# Deploy trained model to a SageMaker endpoint
xgb_predictor = xgb.deploy(
    initial_instance_count=1, 
    instance_type='ml.m4.xlarge'
)


# ============================================================================
# STEP 9: MAKE PREDICTIONS ON TEST DATA
# ============================================================================

from sagemaker.serializers import CSVSerializer

# Prepare test data (remove target columns, convert to numpy array)
test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values

# Configure predictor to serialize input as CSV
xgb_predictor.serializer = CSVSerializer()

# Get predictions from deployed endpoint
predictions = xgb_predictor.predict(test_data_array).decode('utf-8')

# Parse predictions string into numpy array
predictions_array = np.fromstring(predictions[1:], sep=',')

print(predictions_array.shape)


# ============================================================================
# STEP 10: EVALUATE MODEL PERFORMANCE - CONFUSION MATRIX
# ============================================================================

# Create confusion matrix using crosstab
# Rows = Observed/Actual values, Columns = Predicted values
cm = pd.crosstab(
    index=test_data['y_yes'],           # Actual target values
    columns=np.round(predictions_array),  # Predicted values (rounded to 0 or 1)
    rownames=['Observed'], 
    colnames=['Predicted']
)

# Extract confusion matrix values
tn = cm.iloc[0, 0]  # True Negatives
fn = cm.iloc[1, 0]  # False Negatives
tp = cm.iloc[1, 1]  # True Positives
fp = cm.iloc[0, 1]  # False Positives

# Calculate overall accuracy percentage
p = (tp + tn) / (tp + tn + fp + fn) * 100

# Display formatted results
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format(
    "No Purchase", 
    tn/(tn+fn)*100, tn,      # True Negative rate and count
    fp/(tp+fp)*100, fp       # False Positive rate and count
))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format(
    "Purchase", 
    fn/(tn+fn)*100, fn,      # False Negative rate and count
    tp/(tp+fp)*100, tp       # True Positive rate and count
))


# ============================================================================
# STEP 11: CLEANUP RESOURCES (AVOID ONGOING CHARGES)
# ============================================================================

# Delete the deployed endpoint and its configuration
xgb_predictor.delete_endpoint(delete_endpoint_config=True)

# Delete all objects in S3 bucket
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()

# NOTE: Remember to also stop/delete your SageMaker notebook instance 
# manually in the AWS Console:
# 1. Go to SageMaker > Notebook instances
# 2. Select your instance
# 3. Actions > Stop
# 4. After stopped, Actions > Delete

print("Cleanup complete. Remember to delete notebook instance in AWS Console!")