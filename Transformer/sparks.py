import requests
import dask.dataframe as dd
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum, count
from pyspark.sql.window import Window
import pickle

# Initialize Spark session
spark = SparkSession.builder.appName("FinancialDataAnalysis").getOrCreate()

# API URL
api_url = "http://127.0.0.1:5000/fetch_all_data"

# Step 1: Fetch data from the API and preprocess with Dask
def preprocess_with_dask(api_url):
    # Fetch data from the API
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")
    
    # Load data into a Dask DataFrame
    data = response.json()
    dask_df = dd.from_pandas(pd.DataFrame(data), npartitions=4)  # Use Dask to load data into partitions
    
    # Ensure proper data types
    dask_df['bid_price'] = dask_df['bid_price'].astype('float32')
    dask_df['ask_price'] = dask_df['ask_price'].astype('float32')
    dask_df['bid_volume'] = dask_df['bid_volume'].astype('int32')
    dask_df['ask_volume'] = dask_df['ask_volume'].astype('int32')
    
    # Convert 'timestamp' column using Dask's to_datetime method
    dask_df['timestamp'] = dd.to_datetime(dask_df['timestamp'], format="%Y-%m-%d %H:%M:%S")
    
    # Calculate momentum (bid_price - ask_price) and volatility (bid/ask variance)
    dask_df['momentum'] = dask_df['bid_price'] - dask_df['ask_price']
    dask_df['volatility'] = (dask_df['bid_price'] - dask_df['bid_price'].mean()) ** 2
    
    return dask_df

# Step 2: Convert Dask DataFrame to PySpark DataFrame
def load_to_spark(dask_df, spark):
    # Convert Dask DataFrame to Pandas, then to PySpark
    pandas_df = dask_df.compute()  # Trigger computation in Dask and convert to Pandas
    spark_df = spark.createDataFrame(pandas_df)
    return spark_df

# Step 3: Advanced PySpark Analysis
def analyze_with_spark(spark_df):
    # Advanced Aggregations
    print("=== Weighted Average Prices ===")
    weighted_avg_df = spark_df.withColumn(
        "weighted_bid_price", col("bid_price") * col("bid_volume")
    ).groupBy("market_event").agg(
        # Use the sum of weighted bid prices and the total volume for weighted average calculation
        (sum("weighted_bid_price") / sum("bid_volume")).alias("weighted_avg_bid_price"),
        avg("ask_price").alias("avg_ask_price")
    )
    # Save weighted average result to pickle
    weighted_avg_df_pd = weighted_avg_df.toPandas()  # Convert to Pandas DataFrame
    with open("weighted_avg_df.pkl", "wb") as f:
        pickle.dump(weighted_avg_df_pd, f)

    weighted_avg_df.show()

    # Anomaly Detection (e.g., high volatility)
    print("=== Anomalies (High Volatility) ===")
    threshold = spark_df.select(avg("volatility")).first()[0] * 2  # Example threshold
    anomalies_df = spark_df.filter(col("volatility") > threshold)
    anomalies_df.show()
    
    # Save anomaly detection result to pickle
    anomalies_df_pd = anomalies_df.toPandas()
    with open("anomalies_df.pkl", "wb") as f:
        pickle.dump(anomalies_df_pd, f)

    # Event-Based Analysis
    print("=== Event-Based Metrics ===")
    event_metrics_df = spark_df.groupBy("market_event").agg(
        avg("bid_price").alias("avg_bid_price"),
        avg("ask_price").alias("avg_ask_price"),
        count("*").alias("event_count")
    )
    event_metrics_df.show()

    # Save event metrics result to pickle
    event_metrics_df_pd = event_metrics_df.toPandas()
    with open("event_metrics_df.pkl", "wb") as f:
        pickle.dump(event_metrics_df_pd, f)

    # Rolling Averages
    print("=== Rolling Averages (Bid Price) ===")
    window_spec = Window.orderBy("timestamp").rowsBetween(-5, 5)  # 10-minute window (rows)
    rolling_avg_df = spark_df.withColumn("rolling_avg_bid", avg("bid_price").over(window_spec))
    rolling_avg_df.show()

    # Save rolling averages result to pickle
    rolling_avg_df_pd = rolling_avg_df.toPandas()
    with open("rolling_avg_df.pkl", "wb") as f:
        pickle.dump(rolling_avg_df_pd, f)

    # Correlation Analysis
    print("=== Correlation Between Bid Price and Volatility ===")
    correlation = spark_df.stat.corr("bid_price", "volatility")
    print(f"Correlation between Bid Price and Volatility: {correlation}")

    # Save correlation result to pickle
    correlation_result = {"correlation": correlation}
    with open("correlation_result.pkl", "wb") as f:
        pickle.dump(correlation_result, f)

    # Distribution of Order Flows
    print("=== Distribution of Order Flows ===")
    order_flow_distribution = spark_df.groupBy("order_flow").count()
    order_flow_distribution.show()

    # Save order flow distribution result to pickle
    order_flow_distribution_pd = order_flow_distribution.toPandas()
    with open("order_flow_distribution.pkl", "wb") as f:
        pickle.dump(order_flow_distribution_pd, f)

# Step 4: Main flow
if __name__ == "__main__":
    # Step 4.1: Preprocess with Dask
    dask_processed_df = preprocess_with_dask(api_url)
    
    # Step 4.2: Load data into PySpark
    spark_df = load_to_spark(dask_processed_df, spark)
    
    # Step 4.3: Analyze data with PySpark
    analyze_with_spark(spark_df)
