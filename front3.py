import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
import sqlite3
import json
import numpy as np
from decimal import Decimal
import yfinance as yf  # For stock data
import requests
import calendar
from forex_python.converter import CurrencyRates
import pycountry
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import shutil
from datetime import datetime, timedelta

# Database setup enhancements
def init_db():
    conn = sqlite3.connect('financial_planner.db')
    c = conn.cursor()
    
    # Enhanced users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, 
                  password TEXT,
                  email TEXT,
                  life_stage TEXT,
                  goals TEXT,
                  income REAL,
                  expenses TEXT,
                  savings REAL,
                  risk_profile TEXT,
                  notifications TEXT,
                  currency TEXT,
                  created_date TEXT,
                  last_login TEXT,
                  settings TEXT,
                  reset_token TEXT,
                  reset_token_expiry TEXT)''')
    
    # Enhanced transactions table

    # In the transactions table creation:
    c.execute('''CREATE TABLE IF NOT EXISTS transactions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  date TEXT,
                  category TEXT,
                  subcategory TEXT,
                  amount REAL,
                  description TEXT,
                  payment_method TEXT,
                  recurring BOOLEAN,
                  tags TEXT,
                  transaction_type TEXT,
                  attachment_path TEXT,
                  FOREIGN KEY (username) REFERENCES users(username))''')
    
    # Investment portfolio table
    c.execute('''CREATE TABLE IF NOT EXISTS portfolios
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  symbol TEXT,
                  quantity REAL,
                  purchase_price REAL,
                  purchase_date TEXT,
                  portfolio_type TEXT,
                  FOREIGN KEY (username) REFERENCES users(username))''')
    
    # Bills and subscriptions table
    c.execute('''CREATE TABLE IF NOT EXISTS bills
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  name TEXT,
                  amount REAL,
                  due_date TEXT,
                  frequency TEXT,
                  category TEXT,
                  auto_pay BOOLEAN,
                  reminder_days INTEGER,
                  FOREIGN KEY (username) REFERENCES users(username))''')
    
    # Financial reports table
    c.execute('''CREATE TABLE IF NOT EXISTS reports
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  report_type TEXT,
                  report_date TEXT,
                  report_data TEXT,
                  FOREIGN KEY (username) REFERENCES users(username))''')
        
    conn.commit()
    conn.close()

# Enhanced User Settings and Preferences
class UserSettings:
    def __init__(self, username):
        self.username = username
        self.settings = self._load_settings()

    def _default_settings(self):
        return {
            'budget': {},
            'income': {},
            'goals': '[]',  # Store goals as JSON string
            'theme': 'light'
        }

    def _load_settings(self):
        try:
            conn = sqlite3.connect('financial_planner.db')
            c = conn.cursor()
            c.execute("SELECT settings FROM users WHERE username=?", (self.username,))
            result = c.fetchone()
            conn.close()
            
            if result and result[0]:
                settings = json.loads(result[0])
                # Ensure goals is stored as JSON string
                if 'goals' in settings and not isinstance(settings['goals'], str):
                    settings['goals'] = json.dumps(settings['goals'])
                return settings
            return self._default_settings()
        except Exception as e:
            st.error(f"Error loading settings: {str(e)}")
            return self._default_settings()
        
    def save_settings(username, settings):
        try:
            # Ensure goals is JSON string before saving
            if 'goals' in settings and not isinstance(settings['goals'], str):
                settings['goals'] = json.dumps(settings['goals'])
                
            conn = sqlite3.connect('financial_planner.db')
            c = conn.cursor()
            c.execute("""
                UPDATE users 
                SET settings=? 
                WHERE username=?
            """, (json.dumps(settings), username))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error saving settings: {str(e)}")
            return False
    
    def _default_settings(self):
        return {
            'currency': 'USD',
            'date_format': '%Y-%m-%d',
            'theme': 'light',
            'notifications': {
                'email': True,
                'push': False,
                'bill_reminders': True,
                'budget_alerts': True,
                'investment_alerts': True
            },
            'budget_alerts': {
                'threshold': 80,  # Alert when category reaches 80% of budget
                'frequency': 'weekly'
            },
            'report_preferences': {
                'frequency': 'monthly',
                'include_categories': ['all'],
                'export_format': 'pdf'
            },
            'dashboard_widgets': [
                'budget_overview',
                'recent_transactions',
                'upcoming_bills',
                'investment_summary',
                'savings_goals'
            ]
        }

# Enhanced Financial Analysis Tools
class FinancialAnalyzer:
    def __init__(self, username):
        self.username = username
        
    def calculate_net_worth(self):
        """Calculate total net worth including assets and liabilities"""
        conn = sqlite3.connect('financial_planner.db')
        c = conn.cursor()
        
        # Get assets
        c.execute("""
            SELECT SUM(amount) FROM transactions 
            WHERE username=? AND category='income'
        """, (self.username,))
        assets = c.fetchone()[0] or 0
        
        # Get liabilities
        c.execute("""
            SELECT SUM(amount) FROM transactions 
            WHERE username=? AND category='debt'
        """, (self.username,))
        liabilities = c.fetchone()[0] or 0
        
        conn.close()
        return assets - liabilities
    
    def analyze_spending_patterns(self):
        """Analyze spending patterns and provide insights."""
        conn = sqlite3.connect('financial_planner.db')
        current_month = datetime.now().strftime('%Y-%m')

        # Get total income directly from transaction_type
        income_df = pd.read_sql_query("""
            SELECT SUM(amount) as income
            FROM transactions 
            WHERE username=? 
            AND transaction_type='Income'
            AND strftime('%Y-%m', date)=?
        """, conn, params=(self.username, current_month))

        # Get total expenses directly from transaction_type
        expenses_df = pd.read_sql_query("""
            SELECT SUM(amount) as expenses
            FROM transactions 
            WHERE username=? 
            AND transaction_type='Expense'
            AND strftime('%Y-%m', date)=?
        """, conn, params=(self.username, current_month))

        conn.close()

        total_income = income_df['income'].iloc[0] or 0
        total_expenses = expenses_df['expenses'].iloc[0] or 0

        # Calculate monthly savings correctly
        monthly_savings = total_income - total_expenses

        return {
            'monthly_savings': monthly_savings,
            'total_income': total_income,
            'total_expenses': total_expenses
        }


        
# Enhanced Investment Analysis
class InvestmentAnalyzer:
    def __init__(self, username):
        self.username = username
        
    def get_portfolio_performance(self):
        """Calculate portfolio performance including returns and risk metrics"""
        conn = sqlite3.connect('financial_planner.db')
        portfolio = pd.read_sql_query("""
            SELECT * FROM portfolios WHERE username=?
        """, conn, params=(self.username,))
        conn.close()
        
        if portfolio.empty:
            return None
        
        total_value = 0
        returns = []
        
        for _, position in portfolio.iterrows():
            try:
                # Get current price using yfinance
                ticker = yf.Ticker(position['symbol'])
                history = ticker.history(period='1d')
                
                # Check if we got any data
                if not history.empty:
                    current_price = history['Close'].iloc[-1]
                else:
                    # Use purchase price if no current data available
                    current_price = position['purchase_price']
                    st.warning(f"Could not fetch current price for {position['symbol']}, using purchase price")
                
                # Calculate position value and return
                position_value = current_price * position['quantity']
                position_return = (current_price - position['purchase_price']) / position['purchase_price']
                
                total_value += position_value
                returns.append(position_return)
            except Exception as e:
                st.warning(f"Error processing {position['symbol']}: {str(e)}")
                # Use purchase price as fallback
                total_value += position['purchase_price'] * position['quantity']
                returns.append(0.0)
        
        if not returns:
            return {
                'total_value': total_value,
                'portfolio_return': 0.0,
                'portfolio_risk': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Calculate portfolio metrics
        portfolio_return = np.mean(returns)
        portfolio_risk = np.std(returns)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            'total_value': total_value,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio
        }
    
    def get_investment_recommendations(self, risk_profile):
        """Generate investment recommendations based on risk profile"""
        risk_profiles = {
            'conservative': {
                'bonds': 60,
                'stocks': 30,
                'cash': 10,
                'recommendations': [
                    'Treasury bonds',
                    'High-grade corporate bonds',
                    'Blue-chip dividend stocks'
                ]
            },
            'moderate': {
                'bonds': 40,
                'stocks': 50,
                'cash': 10,
                'recommendations': [
                    'Index funds',
                    'Balanced mutual funds',
                    'Mixed corporate bonds'
                ]
            },
            'aggressive': {
                'bonds': 20,
                'stocks': 70,
                'cash': 10,
                'recommendations': [
                    'Growth stocks',
                    'Small-cap funds',
                    'International equity'
                ]
            }
        }
        return risk_profiles.get(risk_profile.lower())

# Enhanced Debt Management
class DebtManager:
    def __init__(self, username):
        self.username = username
    
    def analyze_debt(self):
        """Analyze debt and provide repayment strategies"""
        conn = sqlite3.connect('financial_planner.db')
        debts = pd.read_sql_query("""
            SELECT * FROM transactions 
            WHERE username=? AND category='debt'
        """, conn, params=(self.username,))
        conn.close()
        
        if debts.empty:
            return None
        
        # Calculate key metrics
        total_debt = debts['amount'].sum()
        avg_interest = debts['interest_rate'].mean() if 'interest_rate' in debts else 0
        
        # Generate repayment strategies
        strategies = {
            'avalanche': self._avalanche_strategy(debts),
            'snowball': self._snowball_strategy(debts),
            'consolidation': self._consolidation_strategy(debts)
        }
        
        return {
            'total_debt': total_debt,
            'average_interest': avg_interest,
            'strategies': strategies
        }
    
    def _avalanche_strategy(self, debts):
        """Calculate highest interest first strategy"""
        # Implementation details...
        pass
    
    def _snowball_strategy(self, debts):
        """Calculate smallest balance first strategy"""
        # Implementation details...
        pass
    
    def _consolidation_strategy(self, debts):
        """Calculate debt consolidation strategy"""
        # Implementation details...
        pass

# Enhanced Bill Management
class BillManager:
    def __init__(self, username):
        self.username = username
    
    def get_upcoming_bills(self, days=30):
        """Get list of upcoming bills"""
        conn = sqlite3.connect('financial_planner.db')
        current_date = datetime.now().date()
        end_date = current_date + timedelta(days=days)
        
        bills = pd.read_sql_query("""
            SELECT * FROM bills 
            WHERE username=? AND due_date BETWEEN ? AND ?
        """, conn, params=(self.username, current_date, end_date))
        conn.close()
        
        return bills.sort_values('due_date')
    
    def set_bill_reminder(self, bill_id, reminder_days):
        """Set reminder for bill payment"""
        conn = sqlite3.connect('financial_planner.db')
        c = conn.cursor()
        c.execute("""
            UPDATE bills 
            SET reminder_days=? 
            WHERE id=? AND username=?
        """, (reminder_days, bill_id, self.username))
        conn.commit()
        conn.close()

# Enhanced Report Generator
class ReportGenerator:
    def __init__(self, username):
        self.username = username
    
    def generate_monthly_report(self, month=None):
        """Generate comprehensive monthly financial report"""
        if month is None:
            month = datetime.now().strftime('%Y-%m')
        
        # Get all relevant data
        analyzer = FinancialAnalyzer(self.username)
        investment_analyzer = InvestmentAnalyzer(self.username)
        
        # Generate report sections
        income_expense = self._analyze_income_expenses(month)
        budget_performance = self._analyze_budget_performance(month)
        investment_performance = investment_analyzer.get_portfolio_performance()
        savings_progress = self._analyze_savings_progress()
        
        report = {
            'month': month,
            'income_expense': income_expense,
            'budget_performance': budget_performance,
            'investment_performance': investment_performance,
            'savings_progress': savings_progress,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        self._save_report(report, 'monthly')
        return report
    
    def _analyze_income_expenses(self, month):
        """Analyze income and expenses for the month"""
        # Implementation details...
        pass
    
    def _analyze_budget_performance(self, month):
        """Analyze budget performance for the month"""
        # Implementation details...
        pass
    
    def _analyze_savings_progress(self):
        """Analyze progress towards savings goals"""
        # Implementation details...
        pass
    
    def _generate_recommendations(self):
        """Generate personalized financial recommendations"""
        # Implementation details...
        pass
    
    def _save_report(self, report_data, report_type):
        """Save report to database"""
        conn = sqlite3.connect('financial_planner.db')
        c = conn.cursor()
        c.execute("""
            INSERT INTO reports (username, report_type, report_date, report_data)
            VALUES (?, ?, ?, ?)
        """, (self.username, report_type, datetime.now().strftime('%Y-%m-%d'),
              json.dumps(report_data)))
        conn.commit()
        conn.close()
        
# Add this to handle password reset
def reset_password(token, new_password):
    conn = sqlite3.connect('financial_planner.db')
    c = conn.cursor()
    
    # Check if token is valid and not expired
    c.execute("""
        SELECT username FROM users 
        WHERE reset_token=? AND reset_token_expiry > ?
    """, (token, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    result = c.fetchone()
    
    if result:
        username = result[0]
        hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
        
        # Update password and clear reset token
        c.execute("""
            UPDATE users 
            SET password=?, reset_token=NULL, reset_token_expiry=NULL 
            WHERE username=?
        """, (hashed_password, username))
        conn.commit()
        conn.close()
        return True
    
    conn.close()
    return False

def send_reset_email(email, reset_link):
    sender_email = "your-email@domain.com"
    sender_password = "your-app-specific-password"
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = email
    message["Subject"] = "Password Reset Request"
    
    body = f"""
    Hello,
    
    You have requested to reset your password. Please click the link below to reset your password:
    
    {reset_link}
    
    This link will expire in 1 hour.
    
    If you did not request this reset, please ignore this email.
    
    Best regards,
    Financial Life Planner Team
    """
    
    message.attach(MIMEText(body, "plain"))
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())
        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False


# Add password complexity requirements
def validate_password(password):
    """
    Validates password strength and returns a tuple of (is_valid, message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search("[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search("[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search("[0-9]", password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

# Add session timeout
def check_session_timeout():
    if 'last_activity' in st.session_state:
        if (datetime.now() - st.session_state.last_activity).total_seconds() > 1800:  # 30 minutes
            st.session_state.authenticated = False
            return True
    st.session_state.last_activity = datetime.now()
    return False
    
# Add interactive charts
def create_spending_trend_chart(data):
    fig = px.line(data, x='date', y='amount', 
                  color='category',
                  title='Spending Trends Over Time')
    return fig

def create_portfolio_pie_chart(data):
    fig = px.pie(data, values='value', names='asset',
                 title='Portfolio Asset Allocation')
    return fig

def export_data(data, format='csv'):
    if format == 'csv':
        return data.to_csv(index=False)
    elif format == 'excel':
        return data.to_excel(index=False)
    elif format == 'pdf':
        # Implement PDF export
        pass
    
# In the "Budget Tracking" section
def check_budget_alerts(username):
    conn = sqlite3.connect('financial_planner.db')
    current_month = datetime.now().strftime('%Y-%m')
    
    # Get spending by category
    spending = pd.read_sql_query("""
        SELECT category, SUM(amount) as spent
        FROM transactions 
        WHERE username=? AND strftime('%Y-%m', date)=?
        GROUP BY category
    """, conn, params=(username, current_month))
    conn.close()
    
    # Get budget settings
    user_settings = UserSettings(username)
    budget = user_settings.settings.get('budget', {})
    
    alerts = []
    for _, row in spending.iterrows():
        if row['category'] in budget:
            budget_amount = budget[row['category']]
            if row['spent'] > budget_amount * 0.8:  # 80% threshold
                alerts.append(f"Warning: {row['category']} spending at {(row['spent']/budget_amount)*100:.1f}% of budget")
    return alerts

def track_goal_progress(goal):
    current_amount = calculate_current_amount(goal)
    target_amount = goal['target_amount']
    deadline = datetime.strptime(goal['deadline'], '%Y-%m-%d')
    days_remaining = (deadline - datetime.now()).days
    
    progress = {
        'percentage': (current_amount / target_amount) * 100,
        'remaining_amount': target_amount - current_amount,
        'days_remaining': days_remaining,
        'on_track': is_goal_on_track(current_amount, target_amount, days_remaining)
    }
    return progress

def convert_currency(amount, from_currency, to_currency):
    c = CurrencyRates()
    try:
        rate = c.get_rate(from_currency, to_currency)
        return amount * rate
    except Exception as e:
        st.error(f"Currency conversion error: {str(e)}")
        return amount
    
def backup_database():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f'backup_{timestamp}.db'
    try:
        shutil.copy2('financial_planner.db', backup_file)
        return True
    except Exception as e:
        st.error(f"Backup error: {str(e)}")
        return False

def manage_categories():
    conn = sqlite3.connect('financial_planner.db')
    c = conn.cursor()
    
    # Add custom categories
    c.execute('''CREATE TABLE IF NOT EXISTS custom_categories
                 (username TEXT,
                  category_name TEXT,
                  category_type TEXT,
                  PRIMARY KEY (username, category_name))''')
    conn.commit()
    conn.close()
    
def get_spending_data(username):
    try:
        conn = sqlite3.connect('financial_planner.db')
        spending_data = pd.read_sql_query("""
            SELECT date, category, amount 
            FROM transactions 
            WHERE username=? AND transaction_type='Expense'
            ORDER BY date
        """, conn, params=(username,))
        return spending_data
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals():
            conn.close()






# Enhanced Streamlit Interface
def main():
    st.set_page_config(page_title="Financial Life Planner", layout="wide")
    init_db()
    

    # Initialize session state variables
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "last_activity" not in st.session_state:
        st.session_state.last_activity = datetime.now()

    # Then continue with the session timeout check
    if st.session_state.authenticated:
        if check_session_timeout():
            st.warning("Session expired. Please login again.")
            st.rerun()
    
     # Initialize session state
    if "username" not in st.session_state:
        st.session_state.username = None
        st.session_state.authenticated = False
        
    

    # Login/Registration Section
    if not st.session_state.authenticated:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Login")
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
           
            # Add forgot password expander
            with st.expander("Forgot Password?"):
                forgot_username = st.text_input("Enter your username", key="forgot_username")
                forgot_email = st.text_input("Enter your registered email", key="forgot_email")
                if st.button("Reset Password"):
                    conn = sqlite3.connect('financial_planner.db')
                    c = conn.cursor()
                    c.execute("""
                        SELECT email FROM users 
                        WHERE username=? AND email=?
                    """, (forgot_username, forgot_email))
                    result = c.fetchone()
                    conn.close()
                    
                    if result:
                        # Generate reset token (valid for 1 hour)
                        reset_token = hashlib.sha256(
                            f"{forgot_username}{datetime.now().strftime('%Y-%m-%d-%H')}".encode()
                        ).hexdigest()
                        
                        # Store reset token in database
                        conn = sqlite3.connect('financial_planner.db')
                        c = conn.cursor()
                        c.execute("""
                            UPDATE users 
                            SET reset_token=?, reset_token_expiry=?
                            WHERE username=?
                        """, (reset_token, 
                            (datetime.now() + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
                            forgot_username))
                        conn.commit()
                        conn.close()
                        
                        # Create reset link
                        reset_link = f"http://yourdomain.com/reset_password?token={reset_token}"
                        
                        # Here you would typically send an email with the reset link
                        # For demonstration, we'll show the link in the app
                        st.success("Password reset link has been generated!")
                        st.code(reset_link)
                        st.info("In a production environment, this link would be sent to your email.")
                    else:
                        st.error("Username and email combination not found.")
            
                    
            if st.button("Login"):
                conn = sqlite3.connect('financial_planner.db')
                c = conn.cursor()
                c.execute("SELECT password FROM users WHERE username=?", (login_username,))
                result = c.fetchone()
                conn.close()
                
                if result and result[0] == hashlib.sha256(login_password.encode()).hexdigest():
                    st.session_state.username = login_username
                    st.session_state.authenticated = True
                    st.rerun()  # Changed from experimental_rerun to rerun
                else:
                    st.error("Invalid username or password")
        
        with col2:
            st.subheader("Register")
            reg_username = st.text_input("Username", key="reg_username")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_email = st.text_input("Email")
            
# In the registration section
            if st.button("Register"):
                if not reg_username or not reg_password or not reg_email:
                    st.error("Please fill in all fields")
                else:
                    is_valid, message = validate_password(reg_password)
                    if is_valid:
                        conn = sqlite3.connect('financial_planner.db')
                        c = conn.cursor()
                        try:
                            hashed_password = hashlib.sha256(reg_password.encode()).hexdigest()
                            c.execute("""
                                INSERT INTO users (username, password, email, created_date)
                                VALUES (?, ?, ?, ?)
                            """, (reg_username, hashed_password, reg_email, datetime.now().strftime('%Y-%m-%d')))
                            conn.commit()
                            st.success("Registration successful! Please login.")
                        except sqlite3.IntegrityError:
                            st.error("Username already exists")
                        finally:
                            conn.close()
                    else:
                        st.error(message)

    else:
        # Authenticated user interface
        st.sidebar.title(f"Welcome, {st.session_state.username}")
        
        if st.sidebar.button("Logout"):
            st.session_state.username = None
            st.session_state.authenticated = False
            st.rerun()
        
        # Initialize managers and analyzers
        user_settings = UserSettings(st.session_state.username)
        financial_analyzer = FinancialAnalyzer(st.session_state.username)
        investment_analyzer = InvestmentAnalyzer(st.session_state.username)
        debt_manager = DebtManager(st.session_state.username)
        bill_manager = BillManager(st.session_state.username)
        report_generator = ReportGenerator(st.session_state.username)
        
        # Navigation
        page = st.sidebar.radio("Navigation", 
                              ["Dashboard", "Budget Planner", "Transactions","Budget Tracking", 
                               "Goals", "Investments", "Debt Management",
                               "Bills & Subscriptions", "Reports & Analysis",
                               "Settings"])
        
        if page == "Dashboard":
            st.header("Financial Dashboard")
            
            # Add this to display alerts
            alerts = check_budget_alerts(st.session_state.username)
            if alerts:
                st.subheader("Budget Alerts")
                for alert in alerts:
                    st.warning(alert)
            
            # Quick metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                net_worth = financial_analyzer.calculate_net_worth()
                st.metric("Net Worth", f"${net_worth:,.2f}")
            
            with col2:
                portfolio = investment_analyzer.get_portfolio_performance()
                if portfolio:
                    st.metric("Portfolio Value", f"${portfolio['total_value']:,.2f}",
                             f"{portfolio['portfolio_return']*100:.1f}%")
            
            with col3:
                upcoming_bills = bill_manager.get_upcoming_bills(7)
                if not upcoming_bills.empty:
                    st.metric("Upcoming Bills", f"${upcoming_bills['amount'].sum():,.2f}")
                else:
                    st.metric("Upcoming Bills", "$0.00")
            
            with col4:
                savings_data = financial_analyzer.analyze_spending_patterns()
                if savings_data:
                    savings_amount = savings_data['monthly_savings']
                    st.metric("Savings This Month", 
                            f"${savings_amount:,.2f}",
                            delta="Income exceeds expenses" if savings_amount > 0 else "Overspending detected")
                else:
                    st.metric("Savings This Month", "$0.00")

            # Establish connection before using it
            conn = sqlite3.connect('financial_planner.db')
            spending_data = pd.read_sql_query("""
                SELECT date, category, amount 
                FROM transactions 
                WHERE username=? AND transaction_type='Expense'
                ORDER BY date
            """, conn, params=(st.session_state.username,))

            if not spending_data.empty:
                st.plotly_chart(create_spending_trend_chart(spending_data))

            # Make sure to close the connection after use
            conn.close()

            
            # Display recent transactions
            st.subheader("Recent Transactions")
            conn = sqlite3.connect('financial_planner.db')
            transactions = pd.read_sql_query("""
                SELECT date, category, subcategory, amount, description 
                FROM transactions 
                WHERE username=? 
                ORDER BY date DESC LIMIT 10
            """, conn, params=(st.session_state.username,))
            conn.close()
            
            if not transactions.empty:
                st.dataframe(transactions, use_container_width=True)
            else:
                st.write("No transactions to display.")
        
        elif page == "Budget Planner":
            st.header("Budget Planner")
    
            # Income Section
            st.subheader("Monthly Income")
            income_sources = ["Salary", "Investments", "Side Business", "Other"]
            income = {}
            total_income = 0
            
            for source in income_sources:
                current_income = user_settings.settings.get('income', {}).get(source, 0.0)
                income[source] = st.number_input(f"{source} Income", 
                                            min_value=0.0, 
                                            value=float(current_income),
                                            step=100.0)
                total_income += income[source]
            
            st.metric("Total Monthly Income", f"${total_income:,.2f}")
            
            # Expense Budget Section
            st.subheader("Monthly Expenses Budget")
            categories = ["Housing", "Utilities", "Food", "Transportation", 
                        "Entertainment", "Savings", "Insurance", "Others"]
            budget = {}
            total_budget = 0
            
            for category in categories:
                current_budget = user_settings.settings.get('budget', {}).get(category, 0.0)
                budget[category] = st.number_input(f"{category} Budget", 
                                                min_value=0.0, 
                                                value=float(current_budget),
                                                step=10.0)
                total_budget += budget[category]
            
            st.metric("Total Monthly Budget", f"${total_budget:,.2f}")
            
            # Budget Summary
            st.subheader("Budget Summary")
            remaining = total_income - total_budget
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Monthly Surplus/Deficit", 
                        f"${remaining:,.2f}",
                        delta=f"{(remaining/total_income)*100:.1f}% of income" if total_income > 0 else "0%")
            
            if st.button("Save Budget"):
                user_settings.settings['income'] = income
                user_settings.settings['budget'] = budget
                conn = sqlite3.connect('financial_planner.db')
                c = conn.cursor()
                c.execute("""
                    UPDATE users 
                    SET settings=? 
                    WHERE username=?
                """, (json.dumps(user_settings.settings), st.session_state.username))
                conn.commit()
                conn.close()
                st.success("Budget saved successfully!")

    # Add other pages as needed...
        elif page == "Transactions":
            st.header("Transaction Management")
            
            # Add New Transaction
            st.subheader("Add New Transaction")
            
            # Transaction Type Selection
            transaction_type = st.radio("Transaction Type", ["Expense", "Income"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                transaction_date = st.date_input("Date", value=datetime.now())
                # Modify categories based on transaction type
                if transaction_type == "Expense":
                    transaction_category = st.selectbox("Category", 
                        ["Housing", "Utilities", "Food", "Transportation", 
                        "Entertainment", "Savings", "Insurance", "Others"])
                else:
                    transaction_category = st.selectbox("Category", 
                        ["Salary", "Investments", "Business", "Freelance", 
                        "Rental", "Other Income"])
                transaction_amount = st.number_input("Amount", min_value=0.0, step=10.0)
            
            with col2:
                transaction_subcategory = st.text_input("Subcategory (optional)")
                transaction_description = st.text_input("Description")
                payment_method = st.selectbox("Payment Method", 
                    ["Cash", "Credit Card", "Debit Card", "Bank Transfer", "Other"])
            
            recurring = st.checkbox("Recurring Transaction")
            tags = st.text_input("Tags (comma-separated)")
            
            if st.button("Add Transaction"):
                try:
                    conn = sqlite3.connect('financial_planner.db')
                    c = conn.cursor()
                    c.execute("""
                        INSERT INTO transactions 
                        (username, date, category, subcategory, amount, description, 
                        payment_method, recurring, tags, transaction_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (st.session_state.username, transaction_date.strftime('%Y-%m-%d'),
                        transaction_category, transaction_subcategory, transaction_amount,
                        transaction_description, payment_method, recurring, tags, transaction_type))
                    conn.commit()
                    conn.close()
                    st.success("Transaction added successfully!")
                except Exception as e:
                    st.error(f"Error adding transaction: {str(e)}")

            
            # Transaction History
            st.subheader("Transaction History")
            
            # Filters
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                transaction_type_filter = st.multiselect("Transaction Type",
                    ["Income", "Expense"], default=["Income", "Expense"])
            with col2:
                filter_category = st.multiselect("Filter by Category",
                    ["Housing", "Utilities", "Food", "Transportation", 
                    "Entertainment", "Savings", "Insurance", "Others",
                    "Salary", "Investments", "Business", "Freelance", 
                    "Rental", "Other Income"])
            with col3:
                date_range = st.date_input("Date Range", 
                                        value=(datetime.now() - timedelta(days=30), datetime.now()))
            with col4:
                sort_order = st.selectbox("Sort by", ["Date (Latest)", "Date (Oldest)", 
                                                    "Amount (Highest)", "Amount (Lowest)"])
            
            query = """
                SELECT date, transaction_type, category, subcategory, amount, description, 
                payment_method, tags
                FROM transactions 
                WHERE username=?
            """
            params = [st.session_state.username]

            # Apply filters based on transaction_type
            if transaction_type_filter:
                query += " AND transaction_type IN ({})".format(','.join(['?'] * len(transaction_type_filter)))
                params.extend(transaction_type_filter)

            # Apply filters based on category
            if filter_category:
                query += " AND category IN ({})".format(','.join(['?'] * len(filter_category)))
                params.extend(filter_category)

            conn = sqlite3.connect('financial_planner.db')
            transactions = pd.read_sql_query(query, conn, params=params)
            conn.close()

            
            if not transactions.empty:
                st.subheader("Transaction List")
                for index, row in transactions.iterrows():
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                    with col1:
                        st.write(f"Date: {row['date']}")
                    with col2:
                        st.write(f"Type: {row['transaction_type']}")
                    with col3:
                        st.write(f"Category: {row['category']}")
                    with col4:
                        st.write(f"Amount: ${abs(row['amount']):,.2f}")
                    with col5:
                        if st.button("Delete", key=f"del_trans_{index}"):
                            try:
                                conn = sqlite3.connect('financial_planner.db')
                                c = conn.cursor()
                                c.execute("""
                                    DELETE FROM transactions 
                                    WHERE username=? AND date=? AND category=? AND amount=? AND description=?
                                """, (st.session_state.username, row['date'], row['category'], 
                                    row['amount'], row['description']))
                                conn.commit()
                                conn.close()
                                st.success("Transaction deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting transaction: {str(e)}")

            
        elif page == "Budget Tracking":
            st.header("Budget Tracking")
            
            # Get current month's transactions
            conn = sqlite3.connect('financial_planner.db')
            current_month = datetime.now().strftime('%Y-%m')
            transactions = pd.read_sql_query("""
                    SELECT category, SUM(amount) as spent
                    FROM transactions 
                    WHERE username=? AND strftime('%Y-%m', date)=?
                    GROUP BY category
                """, conn, params=(st.session_state.username, current_month))
        
            # Get budget data
            budget_data = user_settings.settings.get('budget', {})
            
            # Create comparison dataframe
            budget_comparison = pd.DataFrame(list(budget_data.items()), columns=['category', 'budget'])
            budget_comparison = budget_comparison.merge(transactions, on='category', how='left')
            budget_comparison['spent'] = budget_comparison['spent'].fillna(0)
            budget_comparison['remaining'] = budget_comparison['budget'] - budget_comparison['spent']
            budget_comparison['percentage'] = (budget_comparison['spent'] / budget_comparison['budget'] * 100).round(1)
        
            # Display budget progress
            for _, row in budget_comparison.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Ensure progress value is between 0 and 100, then convert to 0-1 range
                    progress = max(0, min(100, abs(row['percentage']))) / 100
                    st.progress(progress)
                with col2:
                    st.write(f"{row['category']}: ${abs(row['spent']):,.2f} / ${row['budget']:,.2f}")

        elif page == "Goals":
            st.header("Financial Goals")
            
            try:
                # Ensure goals are properly loaded as JSON
                goals_str = user_settings.settings.get('goals', '[]')
                if not isinstance(goals_str, str):
                    goals_str = '[]'
                goals = json.loads(goals_str)
                
                # Add new goal
                st.subheader("Add New Goal")
                goal_name = st.text_input("Goal Name")
                goal_amount = st.number_input("Target Amount", min_value=0.0, step=100.0)
                goal_deadline = st.date_input("Target Date")
                
                if st.button("Add Goal"):
                    if goal_name and goal_amount > 0:  # Basic validation
                        new_goal = {
                            "name": goal_name,
                            "amount": float(goal_amount),  # Ensure amount is float
                            "deadline": goal_deadline.strftime("%Y-%m-%d"),
                            "created_date": datetime.now().strftime("%Y-%m-%d")
                        }
                        
                        # Initialize goals list if empty
                        if not isinstance(goals, list):
                            goals = []
                        
                        goals.append(new_goal)
                        
                        # Update settings with new goals
                        user_settings.settings['goals'] = json.dumps(goals)
                        
                        # Save to database using the save_settings method
                        if user_settings.save_settings(st.session_state.username, user_settings.settings):
                            st.success("Goal added successfully!")
                        else:
                            st.error("Failed to save goal")
                    else:
                        st.warning("Please enter a goal name and amount greater than 0")
                
                # Display existing goals
                if goals and isinstance(goals, list):
                    st.subheader("Your Goals")
                    for i, goal in enumerate(goals):
                        try:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**{goal.get('name', 'Unnamed Goal')}**")
                            with col2:
                                st.write(f"Target: ${float(goal.get('amount', 0)):,.2f}")
                            with col3:
                                st.write(f"Deadline: {goal.get('deadline', 'No deadline')}")
                        except (KeyError, ValueError, TypeError) as e:
                            st.error(f"Error displaying goal {i+1}: {str(e)}")
                            continue
                
            except json.JSONDecodeError as e:
                st.error(f"Error loading goals: {str(e)}")
                user_settings.settings['goals'] = '[]'
                if user_settings.save_settings(st.session_state.username, user_settings.settings):
                    st.info("Goals have been reset. Please try adding a new goal.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                
                if goals and isinstance(goals, list):
                    st.subheader("Your Goals")
                    for i, goal in enumerate(goals):
                        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                        with col1:
                            st.write(f"**{goal.get('name', 'Unnamed Goal')}**")
                        with col2:
                            st.write(f"Target: ${float(goal.get('amount', 0)):,.2f}")
                        with col3:
                            st.write(f"Deadline: {goal.get('deadline', 'No deadline')}")
                        with col4:
                            if st.button("Delete", key=f"del_goal_{i}"):
                                try:
                                    goals.pop(i)
                                    user_settings.settings['goals'] = json.dumps(goals)
                                    if user_settings.save_settings(st.session_state.username, 
                                                                user_settings.settings):
                                        st.success("Goal deleted!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete goal")
                                except Exception as e:
                                    st.error(f"Error deleting goal: {str(e)}")       
    

        elif page == "Investments":
            st.header("Investment Portfolio")

            # Portfolio Summary
            portfolio = investment_analyzer.get_portfolio_performance()
            if portfolio:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Value", f"${portfolio['total_value']:,.2f}")
                with col2:
                    st.metric("Return", f"{portfolio['portfolio_return']*100:.1f}%")
                with col3:
                    st.metric("Risk (Std Dev)", f"{portfolio['portfolio_risk']*100:.1f}%")
            
            # Investment Type Selection
            investment_type = st.selectbox(
                "Investment Type",
                ["Stocks", "Company Shares", "Treasury Bills", "Bonds"]
            )
            
            if investment_type == "Stocks":
                st.subheader("Add Stock Investment")
                symbol = st.text_input("Stock Symbol").upper()
                if symbol:
                    try:
                        ticker = yf.Ticker(symbol)
                        current_price = ticker.history(period='1d')['Close'].iloc[-1]
                        st.write(f"Current Price: ${current_price:.2f}")
                        
                        quantity = st.number_input("Quantity", min_value=0.0, step=1.0)
                        if st.button("Add Stock"):
                            conn = sqlite3.connect('financial_planner.db')
                            c = conn.cursor()
                            c.execute("""
                                INSERT INTO portfolios (username, symbol, quantity, purchase_price, 
                                purchase_date, portfolio_type)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (st.session_state.username, symbol, quantity, current_price, 
                                datetime.now().strftime('%Y-%m-%d'), 'stock'))
                            conn.commit()
                            conn.close()
                            st.success(f"Added {quantity} shares of {symbol}")
                    except Exception as e:
                        st.error(f"Error fetching stock data: {str(e)}")

            elif investment_type == "Company Shares":
                st.subheader("Add Company Shares")
                company_name = st.text_input("Company Name")
                share_price = st.number_input("Share Price", min_value=0.0, step=1.0)
                quantity = st.number_input("Number of Shares", min_value=0.0, step=1.0)
                profitability = st.number_input("Annual Profitability (%)", min_value=-100.0, max_value=1000.0, step=0.1)
                
                if st.button("Add Company Shares"):
                    try:
                        conn = sqlite3.connect('financial_planner.db')
                        c = conn.cursor()
                        c.execute("""
                            INSERT INTO portfolios (username, symbol, quantity, purchase_price, 
                            purchase_date, portfolio_type)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (st.session_state.username, company_name, quantity, share_price, 
                            datetime.now().strftime('%Y-%m-%d'), 'company_share'))
                        conn.commit()
                        conn.close()
                        st.success(f"Added {quantity} shares of {company_name}")
                    except Exception as e:
                        st.error(f"Error adding company shares: {str(e)}")

            elif investment_type == "Treasury Bills":
                st.subheader("Add Treasury Bill Investment")
                tenure = st.selectbox("Tenure", ["91 Days", "182 Days", "364 Days"])
                amount = st.number_input("Investment Amount", min_value=0.0, step=100.0)
                interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
                maturity_date = st.date_input("Maturity Date")
                
                if st.button("Add Treasury Bill"):
                    try:
                        conn = sqlite3.connect('financial_planner.db')
                        c = conn.cursor()
                        c.execute("""
                            INSERT INTO portfolios (username, symbol, quantity, purchase_price, 
                            purchase_date, portfolio_type)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (st.session_state.username, f"T-Bill {tenure}", 1, amount, 
                            datetime.now().strftime('%Y-%m-%d'), 'treasury_bill'))
                        conn.commit()
                        conn.close()
                        st.success(f"Added Treasury Bill investment of ${amount:,.2f}")
                    except Exception as e:
                        st.error(f"Error adding treasury bill: {str(e)}")

            elif investment_type == "Bonds":
                st.subheader("Add Bond Investment")
                bond_type = st.selectbox("Bond Type", ["Government", "Corporate"])
                bond_name = st.text_input("Bond Name/ID")
                principal = st.number_input("Principal Amount", min_value=0.0, step=100.0)
                coupon_rate = st.number_input("Coupon Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
                maturity_years = st.number_input("Years to Maturity", min_value=1, max_value=30, step=1)
                
                if st.button("Add Bond"):
                    try:
                        conn = sqlite3.connect('financial_planner.db')
                        c = conn.cursor()
                        c.execute("""
                            INSERT INTO portfolios (username, symbol, quantity, purchase_price, 
                            purchase_date, portfolio_type)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (st.session_state.username, f"{bond_type} Bond - {bond_name}", 1, 
                            principal, datetime.now().strftime('%Y-%m-%d'), 'bond'))
                        conn.commit()
                        conn.close()
                        st.success(f"Added {bond_type} Bond investment of ${principal:,.2f}")
                    except Exception as e:
                        st.error(f"Error adding bond: {str(e)}")

            # Display Current Portfolio
            st.subheader("Current Portfolio")
            conn = sqlite3.connect('financial_planner.db')
            portfolio_df = pd.read_sql_query("""
                SELECT symbol, quantity, purchase_price, purchase_date, portfolio_type
                FROM portfolios 
                WHERE username=?
                ORDER BY purchase_date DESC
            """, conn, params=(st.session_state.username,))
            conn.close()

            if not portfolio_df.empty:
                portfolio_df['Total Value'] = portfolio_df['quantity'] * portfolio_df['purchase_price']
                st.dataframe(portfolio_df, use_container_width=True)
            else:
                st.write("No investments in portfolio yet.")
                
            if not portfolio_df.empty:
                st.subheader("Current Portfolio")
                for index, row in portfolio_df.iterrows():
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                    with col1:
                        st.write(f"Symbol: {row['symbol']}")
                    with col2:
                        st.write(f"Quantity: {row['quantity']}")
                    with col3:
                        st.write(f"Purchase Price: ${row['purchase_price']:,.2f}")
                    with col4:
                        st.write(f"Total Value: ${row['Total Value']:,.2f}")
                    with col5:
                        if st.button("Delete", key=f"del_inv_{index}"):
                            try:
                                conn = sqlite3.connect('financial_planner.db')
                                c = conn.cursor()
                                c.execute("""
                                    DELETE FROM portfolios 
                                    WHERE username=? AND symbol=? AND quantity=? AND purchase_price=?
                                """, (st.session_state.username, row['symbol'], 
                                    row['quantity'], row['purchase_price']))
                                conn.commit()
                                conn.close()
                                st.success("Investment deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting investment: {str(e)}")              


        elif page == "Debt Management":
            st.header("Debt Management")

            # Add New Debt
            st.subheader("Add New Debt")
            debt_name = st.text_input("Debt Name")
            debt_amount = st.number_input("Amount", min_value=0.0, step=100.0)
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
            
            if st.button("Add Debt"):
                conn = sqlite3.connect('financial_planner.db')
                c = conn.cursor()
                try:
                    c.execute("""
                        INSERT INTO transactions 
                        (username, date, category, description, amount, interest_rate)
                        VALUES (?, ?, 'debt', ?, ?, ?)
                    """, (st.session_state.username, datetime.now().strftime('%Y-%m-%d'),
                        debt_name, debt_amount, interest_rate))
                    conn.commit()
                    st.success("Debt added successfully!")
                except sqlite3.Error as e:
                    st.error(f"Error adding debt: {e}")
                finally:
                    conn.close()
            
            # Debt Analysis
            debt_analysis = debt_manager.analyze_debt()
            if debt_analysis:
                st.subheader("Debt Overview")
                st.metric("Total Debt", f"${debt_analysis['total_debt']:,.2f}")
                if debt_analysis['average_interest'] is not None:
                    st.metric("Average Interest Rate", f"{debt_analysis['average_interest']:.1f}%")

        elif page == "Bills & Subscriptions":
            st.header("Bills & Subscriptions")
        
            # Add New Bill
            st.subheader("Add New Bill")
            bill_name = st.text_input("Bill Name")
            bill_amount = st.number_input("Amount", min_value=0.0, step=10.0)
            due_date = st.date_input("Due Date")
            frequency = st.selectbox("Frequency", ["Monthly", "Weekly", "Yearly"])
            reminder = st.number_input("Reminder (days before)", min_value=0, max_value=30)
            
            if st.button("Add Bill"):
                conn = sqlite3.connect('financial_planner.db')
                c = conn.cursor()
                c.execute("""
                    INSERT INTO bills (username, name, amount, due_date, frequency, reminder_days)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (st.session_state.username, bill_name, bill_amount,
                    due_date.strftime('%Y-%m-%d'), frequency, reminder))
                conn.commit()
                conn.close()
                st.success("Bill added successfully!")
            
            # Display Upcoming Bills
            st.subheader("Upcoming Bills")
            upcoming = bill_manager.get_upcoming_bills()
            if not upcoming.empty:
                st.dataframe(upcoming[['name', 'amount', 'due_date', 'frequency']])

        elif page == "Reports & Analysis":
            st.header("Financial Reports & Analysis")
        
            # Generate Report
            report_type = st.selectbox("Report Type", ["Monthly", "Quarterly", "Annual"])
            if st.button("Generate Report"):
                with st.spinner("Generating report..."):
                    report = report_generator.generate_monthly_report()
                    
                    # Display report sections
                    if report['income_expense']:
                        st.subheader("Income vs Expenses")
                        st.write(report['income_expense'])
                    
                    if report['budget_performance']:
                        st.subheader("Budget Performance")
                        st.write(report['budget_performance'])
                    
                    if report['investment_performance']:
                        st.subheader("Investment Performance")
                        st.write(report['investment_performance'])
                    
                    if report['recommendations']:
                        st.subheader("Recommendations")
                        for rec in report['recommendations']:
                            st.write(f"• {rec}")
                            
            # In the "Reports & Analysis" section
            st.subheader("Export Data")
            export_format = st.selectbox("Export Format", ["CSV", "Excel"])
            if st.button("Export Transactions"):
                conn = sqlite3.connect('financial_planner.db')
                transactions = pd.read_sql_query("""
                    SELECT date, category, amount, description 
                    FROM transactions 
                    WHERE username=?
                """, conn, params=(st.session_state.username,))
                conn.close()
                
                if export_format == "CSV":
                    csv = transactions.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="transactions.csv",
                        mime="text/csv"
                    )
                else:  # Excel
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        transactions.to_excel(writer, index=False)
                    excel_data = output.getvalue()
                    st.download_button(
                        label="Download Excel",
                        data=excel_data,
                        file_name="transactions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        elif page == "Settings":
            st.header("Settings")

            # User Preferences
            st.subheader("Preferences")
            currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY", "GHS"], 
                                index=["USD", "EUR", "GBP", "JPY", "GHS"].index(
                                    user_settings.settings.get('currency', 'GHS')))
            
            theme = st.selectbox("Theme", ["light", "dark"], 
                            index=["light", "dark"].index(
                                user_settings.settings.get('theme', 'light')))
            
            # Notification Settings
            st.subheader("Notifications")
            notifications = user_settings.settings.get('notifications', {})
            email_notifications = st.checkbox("Email Notifications", 
                                        value=notifications.get('email', True))
            bill_reminders = st.checkbox("Bill Reminders", 
                                    value=notifications.get('bill_reminders', True))
           
            
            if st.button("Save Settings"):
                try:
                    # Update settings dictionary
                    user_settings.settings.update({
                        'currency': currency,
                        'theme': theme,
                        'notifications': {
                            'email': email_notifications,
                            'bill_reminders': bill_reminders
                        }
                    })
                    
                    # Save settings using a single database connection
                    conn = sqlite3.connect('financial_planner.db')
                    try:
                        c = conn.cursor()
                        c.execute("""
                            UPDATE users 
                            SET settings = ?
                            WHERE username = ?
                        """, (json.dumps(user_settings.settings), st.session_state.username))
                        conn.commit()
                        st.success("Settings saved successfully!")
                    except sqlite3.Error as e:
                        st.error(f"Database error: {e}")
                        conn.rollback()
                    finally:
                        conn.close()
                except Exception as e:
                    st.error(f"Error saving settings: {e}")

if __name__ == "__main__":
    main()

            
