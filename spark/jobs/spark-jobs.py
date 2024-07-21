from pyspark.sql import SparkSession
import pandas as pd
import yfinance as yf
import os

# Fetch stock data from Yahoo Finance API
symbol = "AAPL"
period = "5d"
data = yf.download(symbol, period=period)

# Save raw data to input CSV for traceability
input_path = "./data/input/stock_price_input.csv"
data.to_csv(input_path)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("StockPriceETL") \
    .getOrCreate()

# Read input CSV file from the local data/input directory
stock_df = spark.read.csv(input_path, header=True, inferSchema=True)

# Calculate 3-day moving average of 'Close' price
from pyspark.sql import functions as F
stock_df = stock_df.withColumn("Close_MA_3", F.avg("Close").over(
    Window.orderBy("Date").rowsBetween(-2, 0)
))

# Write the transformed data to the local data/output directory
stock_df.coalesce(1).write.csv("./data/output/stock_price_output.csv", header=True, mode="overwrite")

# Stop the Spark session
spark.stop()
