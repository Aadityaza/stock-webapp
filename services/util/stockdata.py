import requests
import os
import yfinance as yf

from dotenv import load_dotenv

load_dotenv()

fmp_key = os.getenv("FMP_API")
news_key = os.getenv("NEWS_API")


def get_market_data(category):
    url = f"https://financialmodelingprep.com/api/v3/stock_market/{category}?apikey={fmp_key}"
    response = requests.get(url)
    print(response.json())
    return response.json() if response.status_code == 200 else None


def get_market_news():
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': 'stock market',
        'sortBy': 'publishedAt',
        'apiKey': news_key,
        'pageSize': 5,
        'language': 'en'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()['articles']
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return []  # Return an empty list if the request fails


def get_market_summary():
    indices = {"^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ"}
    market_summary = []

    for index_symbol, index_name in indices.items():
        index = yf.Ticker(index_symbol)
        history = index.history(period="1d")
        print("history : ",history)

        if not history.empty:
            last_row = history.iloc[-1]
            day_change = ((last_row['Close'] - last_row['Open']) / last_row['Open']) * 100  # Calculating day change in percentage

            market_summary.append({
                'name': index_name,
                'current_price': last_row['Close'],
                'day_change': round(day_change, 2),  # Round to 2 decimal places
                'open': last_row['Open'],  # Include the 'open' attribute
                'high': last_row['High'],  # Include the 'high' attribute
                'low': last_row['Low'],    # Include the 'low' attribute
                'volume': last_row['Volume']  # Include the 'volume' attribute
            })
        else:
            market_summary.append({
                'name': index_name,
                'current_price': 0,
                'day_change': 0,
                'open': 0,
                'high': 0,
                'low': 0,
                'volume': 0
            })

    return market_summary


def get_stock_details(stock_symbol):
    print("get_stock_details: ", stock_symbol)
    # Initialize a dictionary to hold all the data
    stock_data = {}

    try:
        stock = yf.Ticker(stock_symbol)
        stock_data['yf_info'] = stock.info

        fmp_profile_url = f"https://financialmodelingprep.com/api/v3/profile/{stock_symbol}?apikey={fmp_key}"
        profile_response = requests.get(fmp_profile_url)
        if profile_response.status_code == 200:
            stock_data['fmp_profile'] = profile_response.json()[0]
    except Exception as e:
        stock_data['error'] = str(e)
    return stock_data
