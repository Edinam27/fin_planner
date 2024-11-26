-- Users table

CREATE TABLE IF NOT EXISTS users
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
                  reset_token_expiry TEXT);


-- Transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    date TEXT,
    category TEXT,
    subcategory TEXT,
    amount REAL,
    description TEXT,
    payment_method TEXT,
    recurring BOOLEAN,
    tags TEXT,
    attachment_path TEXT,
    FOREIGN KEY (username) REFERENCES users(username)
);

ALTER TABLE transactions ADD COLUMN interest_rate REAL;
ALTER TABLE transactions ADD COLUMN transaction_type TEXT;

-- Investment portfolios table
CREATE TABLE IF NOT EXISTS portfolios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    symbol TEXT,
    quantity REAL,
    purchase_price REAL,
    purchase_date TEXT,
    portfolio_type TEXT,
    FOREIGN KEY (username) REFERENCES users(username)
);

-- Bills and subscriptions table
CREATE TABLE IF NOT EXISTS bills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    name TEXT,
    amount REAL,
    due_date TEXT,
    frequency TEXT,
    category TEXT,
    auto_pay BOOLEAN,
    reminder_days INTEGER,
    FOREIGN KEY (username) REFERENCES users(username)
);

-- Financial reports table
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    report_type TEXT,
    report_date TEXT,
    report_data TEXT,
    FOREIGN KEY (username) REFERENCES users(username)
);


UPDATE transactions SET transaction_type = 
    CASE 
        WHEN category IN ('Salary', 'Investments', 'Business', 'Freelance', 'Rental', 'Other Income') THEN 'Income'
        ELSE 'Expense'
    END
    WHERE transaction_type IS NULL;

