from flask import Flask, jsonify
import random
import threading
import time
import csv
import os
from datetime import datetime

app = Flask(__name__)

# File name for synthetic market data
csv_file = "synthetic_market_data.csv"

# Function to initialize CSV if it doesn't exist
def initialize_csv():
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp", "bid_price", "ask_price", "bid_volume", "ask_volume",
                "order_flow", "market_event"
            ])

# Function to generate a single market data record
def generate_market_data():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    bid_price = round(random.uniform(100, 200), 2)
    ask_price = round(bid_price + random.uniform(0.01, 0.5), 2)
    bid_volume = random.randint(10, 1000)
    ask_volume = random.randint(10, 1000)
    order_flow = random.choice(["BUY", "SELL", "HOLD"])
    market_event = random.choice(["NORMAL", "SPIKE", "DROP"])

    # Introduce spikes or drops randomly
    if random.random() < 0.1:
        if market_event == "SPIKE":
            bid_price *= 1.1
            ask_price *= 1.1
        elif market_event == "DROP":
            bid_price *= 0.9
            ask_price *= 0.9

    return {
        "timestamp": timestamp,
        "bid_price": round(bid_price, 2),
        "ask_price": round(ask_price, 2),
        "bid_volume": bid_volume,
        "ask_volume": ask_volume,
        "order_flow": order_flow,
        "market_event": market_event
    }

# Background task to append generated data to CSV every second
def generate_data_continuously():
    while True:
        market_data = generate_market_data()
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(market_data.values())
        time.sleep(1)

# API route to get recent data from CSV
@app.route('/recent_data', methods=['GET'])
def recent_data():
    market_data = []
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                market_data.append(row)
    return jsonify(market_data[-10:])

# API route to fetch all data from CSV
@app.route('/fetch_all_data', methods=['GET'])
def fetch_all_data():
    market_data = []
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                market_data.append(row)
    return jsonify(market_data)  # Return all records as JSON

if __name__ == '__main__':
    # Initialize CSV and start background data generation
    initialize_csv()
    threading.Thread(target=generate_data_continuously, daemon=True).start()
    app.run(port=5000)
