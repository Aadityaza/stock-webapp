from flask import Flask, request, render_template, redirect, url_for, session
from markupsafe import Markup
import plotly.graph_objects as go
from services.util.stockdata import *
from datetime import datetime
from services.prediction.predict_stock import *
import bcrypt
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generates a random key for sessions

# File to store user data
USER_FILE = 'users.txt'


def save_user(username, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    with open(USER_FILE, 'a') as file:
        file.write(f'{username}:{hashed_password.decode("utf-8")}\n')

def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, 'r') as file:
        users = {}
        for line in file:
            username, hashed_password = line.strip().split(':')
            users[username] = hashed_password
        return users

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')

        users = load_users()

        if username in users and bcrypt.checkpw(password, users[username].encode('utf-8')):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return 'Invalid username or password'
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()

        if username in users:
            return 'Username already exists'
        else:
            save_user(username, password)
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('landing'))

# Define the custom filter
def number_format(value, format="%0.2f"):
    return format % value

# Register the filter with the Jinja2 environment
app.jinja_env.filters['number_format'] = number_format

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

        # Check if market data is already in the session
    if 'top_gainers' not in session:
        top_gainers = get_market_data("gainers")
        session['top_gainers'] = top_gainers
    else:
        top_gainers = session['top_gainers']

    if 'top_losers' not in session:
        top_losers = get_market_data("losers")
        session['top_losers'] = top_losers
    else:
        top_losers = session['top_losers']

    if 'most_active' not in session:
        most_active = get_market_data("actives")
        session['most_active'] = most_active
    else:
        most_active = session['most_active']

    market_summary = get_market_summary()
    print(market_summary)
    market_news = get_market_news()
    # Check if a user is logged in to determine whether to display the "Logout" button
    is_logged_in = True if 'username' in session else False

    # Pass data and the 'is_logged_in' variable to the dashboard template
    return render_template('dashboard.html', username=session.get('username', ''),
                           top_gainers=top_gainers, top_losers=top_losers,
                           most_active=most_active, market_summary=market_summary,
                           market_news=market_news, is_logged_in=is_logged_in)


@app.route('/stock', methods=['GET', 'POST'])
def stock():
    if 'username' not in session:
        return redirect(url_for('login'))

    default_end_date = datetime.today().strftime('%Y-%m-%d')
    default_start_date = (datetime.today() - timedelta(days=1000)).strftime('%Y-%m-%d')

    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        # Redirect to the results page with the form data
        return redirect(url_for('stock_results', symbol=stock_symbol, start=start_date, end=end_date))

    return render_template('stock_form.html', start_date=default_start_date, end_date=default_end_date)

@app.route('/stock_results')
def stock_results():
    stock_symbol = request.args.get('symbol')
    start_date = request.args.get('start')
    end_date = request.args.get('end')

    try:
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        if df.empty:
            error_message = f"No data found for the symbol: {stock_symbol}. Please check the symbol and try again."
            return render_template('stock_results.html', error_message=error_message)

        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                             open=df['Open'],
                                             high=df['High'],
                                             low=df['Low'],
                                             close=df['Close'])])
        fig.update_layout(xaxis_rangeslider_visible=False)
        chart = Markup(fig.to_html(full_html=False, default_height='500px', default_width='100%'))

        # Fetch additional stock details
        stock_details = get_stock_details(stock_symbol)

    except Exception as e:
        error_message = f"Error: {str(e)}. Unable to fetch data for {stock_symbol}"
        return render_template('stock_results.html', error_message=error_message)

    return render_template('stock_results.html', chart=chart, stock_details=stock_details)



@app.route('/predict_stock', methods=['GET', 'POST'])
def predict_stock():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Initialize variables
    future_dates = []
    future_predictions = []
    future_prediction_plot = None
    model_performance_plot = None
    error_message = None
    stock_details = {}  # Initialize stock_details
    stock_symbol= None

    if request.method == 'POST':
        try:
            # Extract data from the form
            stock_symbol = request.form.get('stock_symbol')
            future_days = int(request.form.get('future_days'))
            start_date = request.form.get('start_date') or None
            end_date = request.form.get('end_date') or None
            past_years = int(request.form.get('past_years')) if request.form.get('past_years') else 10

            # Use the stock prediction module
            future_dates, future_predictions, plot_of_perform = load_model_and_predict(stock_symbol, future_days, past_years, start_date, end_date)

            # Generate interactive plots using Plotly
            future_prediction_plot = plot_future_predictions(future_dates, future_predictions)
            model_performance_plot = plot_of_perform

            # Fetch additional stock details
            stock_details = get_stock_details(stock_symbol)

        except Exception as e:
            # Handle the exception
            error_message = f"Error: {str(e)}. Cannot predict. Make sure the stock symbol exists."

    return render_template('predict_stock.html',
                           predicted_prices=zip(future_dates, future_predictions),
                           future_prediction_plot=future_prediction_plot,
                           model_performance_plot=model_performance_plot,
                           stock_details=stock_details,
                           stock_symbol=stock_symbol,
                           error_message=error_message)




#------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    app.run(debug=True)
