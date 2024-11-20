import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

# Load pickle files
with open("weighted_avg_df.pkl", "rb") as f:
    weighted_avg_df = pickle.load(f)

with open("anomalies_df.pkl", "rb") as f:
    anomalies_df = pickle.load(f)

with open("event_metrics_df.pkl", "rb") as f:
    event_metrics_df = pickle.load(f)

with open("rolling_avg_df.pkl", "rb") as f:
    rolling_avg_df = pickle.load(f)

with open("correlation_result.pkl", "rb") as f:
    correlation_result = pickle.load(f)

with open("order_flow_distribution.pkl", "rb") as f:
    order_flow_distribution = pickle.load(f)

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Financial Data Analysis Visualizations", style={"text-align": "center"}),

    dcc.Tabs([
        # Tab 1: Weighted Average Prices
        dcc.Tab(label="Weighted Average Prices", children=[
            html.Div([
                dcc.Graph(
                    id="weighted_avg_graph",
                    figure=px.bar(weighted_avg_df, x="market_event", y="weighted_avg_bid_price",
                                  title="Weighted Average Bid Prices by Market Event")
                )
            ])
        ]),

        # Tab 2: Anomalies (High Volatility)
        dcc.Tab(label="Anomalies (High Volatility)", children=[
            html.Div([
                dcc.Graph(
                    id="anomalies_graph",
                    figure=px.scatter(anomalies_df, x="timestamp", y="volatility", color="market_event",
                                      title="Anomalies (High Volatility)")
                )
            ])
        ]),

        # Tab 3: Event-Based Metrics
        dcc.Tab(label="Event-Based Metrics", children=[
            html.Div([
                dcc.Graph(
                    id="event_metrics_graph",
                    figure=px.line(event_metrics_df, x="market_event", y="avg_bid_price", markers=True,
                                  title="Event-Based Average Bid Prices")
                )
            ])
        ]),

        # Tab 4: Rolling Averages (Bid Price)
        dcc.Tab(label="Rolling Averages (Bid Price)", children=[
            html.Div([
                dcc.Graph(
                    id="rolling_avg_graph",
                    figure=px.line(rolling_avg_df, x="timestamp", y="rolling_avg_bid", title="Rolling Average Bid Price")
                )
            ])
        ]),

        # Tab 5: Correlation Between Bid Price and Volatility
        dcc.Tab(label="Correlation Analysis", children=[
            html.Div([
                dcc.Graph(
                    id="correlation_graph",
                    figure=px.bar(x=["Bid Price vs Volatility"], y=[correlation_result["correlation"]],
                                  title="Correlation Between Bid Price and Volatility")
                )
            ])
        ]),

        # Tab 6: Order Flow Distribution
        dcc.Tab(label="Order Flow Distribution", children=[
            html.Div([
                dcc.Graph(
                    id="order_flow_graph",
                    figure=px.pie(order_flow_distribution, names="order_flow", values="count", title="Distribution of Order Flows")
                )
            ])
        ]),

        # Tab 7: Average Ask Price Over Time
        dcc.Tab(label="Avg Ask Price Over Time", children=[
            html.Div([
                dcc.Graph(
                    id="avg_ask_price_graph",
                    figure=px.line(weighted_avg_df, x="market_event", y="avg_ask_price",
                                  title="Average Ask Price Over Market Events")
                )
            ])
        ]),

        # Tab 8: Bid vs Ask Price Comparison
        dcc.Tab(label="Bid vs Ask Price", children=[
            html.Div([
                dcc.Graph(
                    id="bid_vs_ask_graph",
                    figure=px.scatter(weighted_avg_df, x="weighted_avg_bid_price", y="avg_ask_price",
                                      title="Bid Price vs Ask Price Comparison")
                )
            ])
        ]),

        # Tab 9: Event Frequency Over Time
        dcc.Tab(label="Event Frequency Over Time", children=[
            html.Div([
                dcc.Graph(
                    id="event_frequency_graph",
                    figure=px.histogram(event_metrics_df, x="market_event", nbins=30,
                                         title="Event Frequency Over Time")
                )
            ])
        ]),

        # Tab 10: Distribution of Volatility
        dcc.Tab(label="Volatility Distribution", children=[
            html.Div([
                dcc.Graph(
                    id="volatility_distribution_graph",
                    figure=px.histogram(anomalies_df, x="volatility", nbins=50, title="Distribution of Volatility")
                )
            ])
        ])
    ])
])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
