import csv
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from datetime import datetime

# Initialize Spark session
spark = SparkSession.builder.appName("MarketDataML").getOrCreate()

# Load recent market data from the CSV file
def load_recent_data(csv_file="synthetic_market_data.csv", num_records=100):
    data = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    df = pd.DataFrame(data[-num_records:])  # Load the last 'num_records' rows
    
    # Assuming there's a 'date' or 'timestamp' column; convert it to datetime and set it as index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    return df

# Preprocess the data for ML models
def preprocess_data(df):
    # Convert relevant columns to numeric and process categorical data
    df['bid_price'] = pd.to_numeric(df['bid_price'], errors='coerce')
    df['ask_price'] = pd.to_numeric(df['ask_price'], errors='coerce')
    df['bid_volume'] = pd.to_numeric(df['bid_volume'], errors='coerce')
    df['ask_volume'] = pd.to_numeric(df['ask_volume'], errors='coerce')

    # Convert 'order_flow' and 'market_event' to categorical variables
    df['order_flow'] = df['order_flow'].map({"BUY": 1, "SELL": 2, "HOLD": 0})
    df['market_event'] = df['market_event'].map({"NORMAL": 0, "SPIKE": 1, "DROP": 2})

    # Handle missing values by filling them with the column mean
    df.fillna(df.mean(), inplace=True)
    return df

# Convert data to Spark DataFrame for MLlib
def convert_to_spark_df(df):
    # Convert the Pandas DataFrame to a Spark DataFrame
    spark_df = spark.createDataFrame(df)
    
    # Apply StringIndexer for categorical columns (Convert categorical columns into indexed values)
    order_flow_indexer = StringIndexer(inputCol="order_flow", outputCol="order_flow_index")
    market_event_indexer = StringIndexer(inputCol="market_event", outputCol="market_event_index")
    
    spark_df = order_flow_indexer.fit(spark_df).transform(spark_df)
    spark_df = market_event_indexer.fit(spark_df).transform(spark_df)
    
    # One-hot encoding for categorical columns
    encoder = OneHotEncoder(inputCols=["order_flow_index", "market_event_index"], outputCols=["order_flow_onehot", "market_event_onehot"])
    spark_df = encoder.fit(spark_df).transform(spark_df)
    
    # Assemble features into a single vector
    assembler = VectorAssembler(
        inputCols=["bid_price", "ask_price", "bid_volume", "ask_volume", "order_flow_onehot", "market_event_onehot"],
        outputCol="features"
    )
    spark_df = assembler.transform(spark_df)
    
    # Scale features for better model performance
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    spark_df = scaler.fit(spark_df).transform(spark_df)
    
    return spark_df

# Train a RandomForestClassifier to predict the direction of the next trade
def train_trade_classifier(spark_df):
    # For simplicity, classify based on 'order_flow' (BUY, SELL, HOLD)
    df = spark_df.withColumn("label", spark_df["order_flow_index"])
    
    # Split data into training and testing sets
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)
    
    # Train the RandomForestClassifier or try other models like GBTClassifier
    rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label")
    gbt = GBTClassifier(featuresCol="scaled_features", labelCol="label")
    
    # Choose RandomForest for this example (you can switch to gbt)
    model = rf.fit(train_data)
    
    # Make predictions
    predictions = model.transform(test_data)
    
    # Evaluate model accuracy
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    
    # Confusion Matrix and Classification Report
    y_true = predictions.select("label").rdd.flatMap(lambda x: x).collect()
    y_pred = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["BUY", "SELL", "HOLD"])
    
    # Plot Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["BUY", "SELL", "HOLD"], yticklabels=["BUY", "SELL", "HOLD"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    print("Classification Report:\n", report)
    
    return accuracy

# Train a KMeans model to cluster similar trade behaviors
def train_trade_clustering(spark_df):
    # Perform clustering using KMeans
    kmeans = KMeans(k=3, featuresCol="scaled_features", predictionCol="cluster")
    model = kmeans.fit(spark_df)
    predictions = model.transform(spark_df)
    
    # Visualize clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=predictions.toPandas(), x="bid_price", y="ask_price", hue="cluster", palette="viridis")
    plt.title("Clustering Similar Trade Behaviors")
    plt.xlabel("Bid Price")
    plt.ylabel("Ask Price")
    plt.legend(title="Cluster")
    plt.show()
    
    return predictions

# Anomaly detection (using Isolation Forest or similar approach)
def detect_anomalies(df):
    # Select relevant features
    features = df[['bid_price', 'ask_price', 'bid_volume', 'ask_volume']].values
    
    # Train an Isolation Forest model
    model = IsolationForest(contamination=0.1)
    anomalies = model.fit_predict(features)
    
    # Add anomaly column
    df['anomaly'] = anomalies
    anomalies_df = df[df['anomaly'] == -1]
    
    # Visualize anomalies
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=anomalies_df, x="bid_price", y="ask_price", color='red', label='Anomaly')
    plt.title("Anomaly Detection (Isolation Forest)")
    plt.xlabel("Bid Price")
    plt.ylabel("Ask Price")
    plt.legend()
    plt.show()
    
    return anomalies_df

# Main function to run all tasks
def main():
    # Load and preprocess the data
    df = load_recent_data()
    df = preprocess_data(df)
    
    # Convert to Spark DataFrame
    spark_df = convert_to_spark_df(df)
    
    # Train and evaluate the classification model
    accuracy = train_trade_classifier(spark_df)
    print(f"Trade Direction Classification Accuracy: {accuracy}")
    
    # Perform clustering
    cluster_predictions = train_trade_clustering(spark_df)
    
    # Perform anomaly detection
    anomalies = detect_anomalies(df)

if __name__ == "__main__":
    main()
