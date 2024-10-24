import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer, StandardScaler
from pyspark.ml.stat import Correlation
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.functions import vector_to_array

# Initialize Spark session with optimized configurations
spark = SparkSession.builder \
    .appName("EnhancedCardioHealthAnalysis") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

# Set Spark log level to ERROR to reduce verbosity
spark.sparkContext.setLogLevel("ERROR")

# Suppress specific Spark/Hadoop warnings
logger = logging.getLogger("org.apache.spark")
logger.setLevel(logging.ERROR)

logger_native = logging.getLogger("org.apache.hadoop.util.NativeCodeLoader")
logger_native.setLevel(logging.ERROR)

def create_image_directory(base_dir='images'):
    """Create a directory to save images if it doesn't exist."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

def load_data(file_path):
    """Load CSV data into Spark DataFrame."""
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    print(f"Data Loaded Successfully with {df.count()} records and {len(df.columns)} columns.")
    return df

def analyze_missing_values(df):
    """Analyze and report missing values in the DataFrame."""
    missing_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
    missing_pd = missing_counts.toPandas().T.rename(columns={0: 'Missing_Count'})
    missing_pd = missing_pd[missing_pd['Missing_Count'] > 0]
    if not missing_pd.empty:
        print("Missing Values Detected:")
        print(missing_pd)
    else:
        print("No Missing Values Detected.")
    return missing_pd

def impute_missing_values(df, missing_pd):
    """Impute missing values: Mean for numerical and Mode for categorical variables."""
    # Identify numerical and categorical columns
    num_cols = [field.name for field in df.schema.fields if field.dataType.typeName() in ['integer', 'double']]
    cat_cols = [field.name for field in df.schema.fields if field.dataType.typeName() == 'string']
    
    # Mean imputation for numerical columns
    for col in num_cols:
        mean_val = df.select(F.mean(col)).first()[0]
        df = df.fillna({col: mean_val})
        print(f"Imputed missing values in numerical column '{col}' with mean: {mean_val}")

    # Mode imputation for categorical columns
    for col in cat_cols:
        mode_val = df.groupBy(col).count().orderBy(F.desc("count")).first()
        if mode_val:
            mode = mode_val[0]
            df = df.fillna({col: mode})
            print(f"Imputed missing values in categorical column '{col}' with mode: {mode}")
    return df

def remove_duplicates(df):
    """Remove duplicate rows from the DataFrame."""
    initial_count = df.count()
    df = df.dropDuplicates()
    final_count = df.count()
    duplicates_removed = initial_count - final_count
    print(f"Duplicates Removed: {duplicates_removed}")
    return df

def visualize_data(df, num_cols, cat_cols, image_dir):
    """Perform exploratory data analysis with visualizations and save them as images."""
    # Convert to Pandas DataFrame for visualization (sample if data is large)
    sample_pdf = df.sample(fraction=0.1, seed=42).toPandas()
    
    # Interactive Box Plots for numerical features
    for col in num_cols:
        fig = px.box(sample_pdf, y=col, title=f'Box Plot of {col}')
        save_path = os.path.join(image_dir, f'box_plot_{col}.png')
        fig.write_image(save_path)
        print(f"Saved Box Plot for {col} at {save_path}")
    
    # Interactive Histograms for numerical features
    for col in num_cols:
        fig = px.histogram(sample_pdf, x=col, nbins=30, title=f'Histogram of {col}', marginal="box")
        save_path = os.path.join(image_dir, f'histogram_{col}.png')
        fig.write_image(save_path)
        print(f"Saved Histogram for {col} at {save_path}")
    
    # Interactive Bar Charts for categorical features
    for col in cat_cols:
        counts = sample_pdf[col].value_counts().reset_index()
        counts.columns = [col, 'Count']
        fig = px.bar(counts, x=col, y='Count', title=f'Bar Chart of {col}')
        save_path = os.path.join(image_dir, f'bar_chart_{col}.png')
        fig.write_image(save_path)
        print(f"Saved Bar Chart for {col} at {save_path}")
    
    # Correlation Matrix Heatmap using Seaborn
    numeric_pdf = sample_pdf[num_cols]
    plt.figure(figsize=(12, 10))
    correlation_matrix = numeric_pdf.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    save_path = os.path.join(image_dir, 'correlation_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Correlation Matrix Heatmap at {save_path}")

def encode_and_prepare_features(df, cat_cols):
    """Encode categorical variables and assemble features for modeling."""
    from pyspark.ml.functions import vector_to_array
    
    # StringIndexer for categorical variables
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid='keep') for col in cat_cols]
    
    # OneHotEncoder for categorical variables with dropLast=False to retain all categories
    encoders = [OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded", dropLast=False) for col in cat_cols]
    
    # Assemble feature columns
    encoded_cols = [f"{col}_encoded" for col in cat_cols]
    num_cols = [field.name for field in df.schema.fields if field.dataType.typeName() in ['integer', 'double'] and field.name != 'cardio']
    feature_cols = encoded_cols + num_cols
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features")
    
    # Standard Scaling
    scaler = StandardScaler(inputCol="assembled_features", outputCol="features", withStd=True, withMean=False)
    
    # Define the pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
    pipeline_model = pipeline.fit(df)
    df_prepared = pipeline_model.transform(df)
    
    feature_names = []
    for col in cat_cols:
        distinct_values = df.select(col).distinct().count()
        # Convert the encoded vector to an array
        array_col = f"{col}_array"
        df_prepared = df_prepared.withColumn(array_col, vector_to_array(f"{col}_encoded"))
        # Extract each element of the array into a separate column
        for i in range(distinct_values):
            encoded_feature = f"{col}_encoded_{i}"
            df_prepared = df_prepared.withColumn(encoded_feature, F.col(array_col)[i])
            feature_names.append(encoded_feature)
        # Optionally, drop the intermediate array column to save space
        df_prepared = df_prepared.drop(array_col)
    
    # Add numerical feature names
    feature_names += num_cols
    
    print(f"Total number of features after encoding: {len(feature_names)}")
    
    return df_prepared, feature_names

def compute_correlation_matrix(df, feature_names, target, image_dir):
    """Compute and visualize the correlation matrix, then save it as an image."""
    assembler = VectorAssembler(inputCols=feature_names, outputCol="features_corr")
    vector_df = assembler.transform(df).select("features_corr")
    corr_matrix = Correlation.corr(vector_df, "features_corr").head()[0].toArray().tolist()
    corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(corr_df, annot=False, fmt=".2f", cmap='coolwarm')
    plt.title('Full Correlation Matrix')
    save_path = os.path.join(image_dir, 'full_correlation_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Full Correlation Matrix Heatmap at {save_path}")
    
    # Identify top correlated features with the target variable
    if target not in corr_df.columns:
        print(f"Target variable '{target}' not found in the correlation matrix.")
        return []
    
    target_corr = corr_df[target].abs().sort_values(ascending=False)
    top_features = target_corr.index[1:6]  # Top 5 excluding target itself
    print(f"Top 5 Correlated Features with {target}:")
    print(target_corr.head(6))
    return top_features.tolist()

def plot_scatter_plots(df, features, target, image_dir):
    """Create scatter plots for top correlated features against the target and save them as images."""
    sample_pdf = df.sample(fraction=0.1, seed=42).toPandas()
    for feature in features:
        fig = px.scatter(sample_pdf, x=feature, y=target,
                         title=f'Scatter Plot of {feature} vs {target}',
                         trendline="ols")
        save_path = os.path.join(image_dir, f'scatter_{feature}_vs_{target}.png')
        fig.write_image(save_path)
        print(f"Saved Scatter Plot for {feature} vs {target} at {save_path}")

def feature_importance(df, feature_names, target, image_dir):
    """Train a Logistic Regression model and display feature importance by saving a bar plot."""
    # Split the data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # Initialize Logistic Regression
    lr = LogisticRegression(featuresCol="features", labelCol=target, maxIter=10)
    
    # Pipeline
    pipeline = Pipeline(stages=[lr])
    model = pipeline.fit(train_df)
    
    # Get feature coefficients
    lr_model = model.stages[-1]
    coefficients = lr_model.coefficients.toArray()
    
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df['abs_coef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='abs_coef', ascending=False)
    
    # Plot Feature Importance
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df)
    plt.title('Feature Importance from Logistic Regression')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    save_path = os.path.join(image_dir, 'feature_importance_logistic_regression.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Feature Importance Plot at {save_path}")
    
    print("Feature Importance:")
    print(coef_df[['Feature', 'Coefficient']])

def model_training(df, feature_names, target, image_dir):
    """Train a Logistic Regression model with cross-validation and evaluate its performance."""
    # Split the data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # Initialize Logistic Regression
    lr = LogisticRegression(featuresCol="features", labelCol=target, maxIter=20)
    
    # Pipeline
    pipeline = Pipeline(stages=[lr])
    
    # Hyperparameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()
    
    # Evaluator
    evaluator = BinaryClassificationEvaluator(labelCol=target, metricName="areaUnderROC")
    
    # CrossValidator
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=5)
    
    # Run cross-validation
    cv_model = crossval.fit(train_df)
    
    # Best model
    best_model = cv_model.bestModel
    lr_model = best_model.stages[-1]
    print(f"Best Model Parameters: regParam={lr_model._java_obj.getRegParam()}, elasticNetParam={lr_model._java_obj.getElasticNetParam()}")
    
    # Predictions
    predictions = best_model.transform(test_df)
    
    # Evaluation Metrics
    auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
    accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test_df.count())
    print(f"Test AUC: {auc}")
    print(f"Test Accuracy: {accuracy}")
    
    # Save Evaluation Metrics as a Text File
    metrics_path = os.path.join(image_dir, 'model_evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Test AUC: {auc}\n")
        f.write(f"Test Accuracy: {accuracy}\n")
    print(f"Saved Model Evaluation Metrics at {metrics_path}")
    
    # Confusion Matrix
    confusion_matrix = predictions.groupBy('label', 'prediction').count().toPandas()
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix.pivot(index='label', columns='prediction', values='count'),
                annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    save_path = os.path.join(image_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Confusion Matrix at {save_path}")

def main():
    # Define the base directory for saving images
    image_dir = create_image_directory('images')
    
    # Path to the dataset
    file_path = "/home/sat3812/Downloads/archive/cardio_train.csv"
    
    # Load data
    df = load_data(file_path)
    
    # Analyze missing values
    missing_pd = analyze_missing_values(df)
    
    # Impute missing values if any
    if not missing_pd.empty:
        df = impute_missing_values(df, missing_pd)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Identify numerical and categorical columns
    num_cols = [field.name for field in df.schema.fields if field.dataType.typeName() in ['integer', 'double'] and field.name != 'cardio']
    
    cat_cols = ['gender', 'cholesterol', 'gluc']
    
    # Exploratory Data Analysis
    visualize_data(df, num_cols, cat_cols, image_dir)
    
    # Encode categorical variables and prepare features
    df_prepared, feature_names = encode_and_prepare_features(df, cat_cols)
    
    # Define target variable
    target_variable = 'cardio'
    
    # Ensure the target variable exists
    if target_variable not in df_prepared.columns:
        print(f"Target variable '{target_variable}' not found in the dataset.")
        spark.stop()
        return
    
    # Define feature columns excluding the target and any identifier columns (e.g., 'id' if present)
    feature_cols = [col for col in df_prepared.columns if col not in ['cardio', 'id']]
    
    # Compute and visualize correlation matrix
    top_features = compute_correlation_matrix(df_prepared, feature_names, target_variable, image_dir)
    
    # Scatter plots for top correlated features
    plot_scatter_plots(df_prepared, top_features, target_variable, image_dir)
    
    # Feature importance analysis
    feature_importance(df_prepared, feature_names, target_variable, image_dir)
    
    # Model Training and Evaluation
    model_training(df_prepared, feature_names, target_variable, image_dir)
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()
