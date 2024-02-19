import plotly.graph_objs as go
import json
import plotly


def plot_future_predictions(dates, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=predictions, mode='lines+markers', name='Future Predictions'))
    fig.update_layout(title='Future Stock Price Predictions', xaxis_title='Date', yaxis_title='Price')

    # Return the plot as JSON
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def plot_model_performance(actual, predicted):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(actual))), y=actual, mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=list(range(len(predicted))), y=predicted, mode='lines', name='Predicted Prices'))
    fig.update_layout(title='Stock Price Prediction Performance', xaxis_title='Time', yaxis_title='Price')

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
