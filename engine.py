import hashlib
import hmac
import math
import os
import random
import secrets
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DB = ROOT / "pilot.sqlite3"
POLICY = (ROOT / "policy" / "pilot_policy.md").read_text(encoding="utf-8")
POLICY_VERSION = "illustrative-pilot-1"
TARGET_CUSTOMERS = int(os.getenv("SYNTHETIC_CUSTOMER_COUNT", "10000"))


def _password_hash(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 210_000
    ).hex()


def connect():
    url = os.getenv("DATABASE_URL")
    if url:
        import psycopg
        return psycopg.connect(url)
    return sqlite3.connect(DB)


def execute(conn, sql, params=()):
    if os.getenv("DATABASE_URL"):
        sql = sql.replace("?", "%s")
    return conn.execute(sql, params)


def init_db():
    with connect() as conn:
        execute(conn, """CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY, name TEXT, age INTEGER, income REAL, current_emi REAL,
            bureau INTEGER, utilization REAL, delinquencies INTEGER, aa_cashflow REAL,
            kyc INTEGER, consent INTEGER, contact_hour INTEGER, requested_amount REAL,
            cohort TEXT, response INTEGER, dpd INTEGER)""")
        execute(conn, """CREATE TABLE IF NOT EXISTS applications (
            customer_id TEXT PRIMARY KEY, status TEXT, amount REAL, rate REAL, term INTEGER,
            source TEXT, updated_at TEXT)""")
        execute(conn, """CREATE TABLE IF NOT EXISTS app_users (
            username TEXT PRIMARY KEY, display_name TEXT, role TEXT, branch_code TEXT,
            password_salt TEXT, password_hash TEXT, active INTEGER)""")
        event_id = "BIGSERIAL PRIMARY KEY" if os.getenv("DATABASE_URL") else "INTEGER PRIMARY KEY"
        execute(conn, f"""CREATE TABLE IF NOT EXISTS journey_events (
            id {event_id}, customer_id TEXT, stage TEXT, action TEXT, detail TEXT,
            actor TEXT, policy_version TEXT, created_at TEXT)""")
        extra_columns = {
            "branch_code": "TEXT", "city": "TEXT", "state": "TEXT", "product": "TEXT",
            "outstanding": "REAL", "loan_status": "TEXT", "emi_amount": "REAL",
            "tenure_months": "INTEGER", "origination_date": "TEXT", "last_payment_date": "TEXT",
            "collection_amount_due": "REAL", "collection_amount_received": "REAL",
            "customer_segment": "TEXT", "preferred_channel": "TEXT",
        }
        existing = {d[0] for d in execute(conn, "SELECT * FROM customers LIMIT 0").description}
        for column, data_type in extra_columns.items():
            if column not in existing:
                execute(conn, f"ALTER TABLE customers ADD COLUMN {column} {data_type}")

        n = execute(conn, "SELECT COUNT(*) FROM customers").fetchone()[0]
        rng = random.Random(24)
        first_names = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Ananya", "Vikram", "Isha", "Arjun", "Nisha", "Aditya", "Kavya", "Rahul", "Sneha", "Varun", "Pooja"]
        last_names = ["Sharma", "Patel", "Iyer", "Singh", "Gupta", "Nair", "Reddy", "Mehta", "Joshi", "Das", "Khan", "Rao"]
        locations = [
            ("BR001", "Mumbai", "Maharashtra"), ("BR002", "Delhi", "Delhi"),
            ("BR003", "Bengaluru", "Karnataka"), ("BR004", "Chennai", "Tamil Nadu"),
            ("BR005", "Kolkata", "West Bengal"), ("BR006", "Hyderabad", "Telangana"),
            ("BR007", "Pune", "Maharashtra"), ("BR008", "Ahmedabad", "Gujarat"),
            ("BR009", "Jaipur", "Rajasthan"), ("BR010", "Lucknow", "Uttar Pradesh"),
            ("BR011", "Kochi", "Kerala"), ("BR012", "Indore", "Madhya Pradesh"),
        ]
        products = ["Personal Loan", "Home Loan", "Vehicle Loan", "Education Loan", "MSME Loan", "Agri Loan"]
        statuses = ["Active", "Active", "Active", "Active", "Closed", "Delinquent", "New"]
        segments = ["Salaried", "Self-employed", "Professional", "Micro enterprise", "Affluent"]
        channels = ["WhatsApp", "SMS", "Voice", "Email"]
        columns = "customer_id,name,age,income,current_emi,bureau,utilization,delinquencies,aa_cashflow,kyc,consent,contact_hour,requested_amount,cohort,response,dpd,branch_code,city,state,product,outstanding,loan_status,emi_amount,tenure_months,origination_date,last_payment_date,collection_amount_due,collection_amount_received,customer_segment,preferred_channel"
        marks = ",".join(["?"] * 30)
        for i in range(n + 1, TARGET_CUSTOMERS + 1):
            income = rng.randrange(18000, 250001, 1000)
            bureau = rng.randrange(590, 841)
            requested = rng.randrange(50000, 2000001, 25000)
            dpd = rng.choices([0, 5, 15, 30, 45, 60, 90], weights=[72, 7, 6, 5, 4, 3, 3], k=1)[0]
            branch, city, state = locations[(i - 1) % len(locations)]
            product = products[(i + rng.randrange(len(products))) % len(products)]
            outstanding = round(requested * rng.uniform(.18, .96), 2)
            emi = round(max(1000, outstanding / rng.choice([12, 24, 36, 48, 60])), 2)
            due = round(emi * max(1, math.ceil(dpd / 30)), 2) if dpd else 0.0
            received = round(due * rng.uniform(0, .95), 2) if due else 0.0
            originated = date.today() - timedelta(days=rng.randrange(10, 1096))
            values = (
                f"C{i:05}", f"{first_names[(i-1) % len(first_names)]} {last_names[(i*7) % len(last_names)]}",
                rng.randrange(21, 66), income, rng.randrange(0, max(1000, int(income * .38)), 1000), bureau,
                round(rng.uniform(.05, .94), 2), int(dpd >= 30), round(rng.uniform(.55, 1.42), 2),
                int(rng.random() > .05), int(rng.random() > .11), rng.randrange(9, 20), requested,
                "pilot" if i % 5 == 0 else ("control" if i % 5 == 1 else "outside"),
                int(rng.random() < .27), dpd, branch, city, state, product, outstanding,
                "Delinquent" if dpd else rng.choice(statuses), emi, rng.choice([12, 24, 36, 48, 60, 84, 120, 180, 240]),
                originated.isoformat(), (date.today() - timedelta(days=rng.randrange(1, 46))).isoformat(),
                due, received, rng.choice(segments), rng.choice(channels),
            )
            execute(conn, f"INSERT INTO customers ({columns}) VALUES ({marks})", values)

        backfills = {
            "branch_code": "BR001", "city": "Mumbai", "state": "Maharashtra",
            "product": "Personal Loan", "loan_status": "Active", "origination_date": date.today().isoformat(),
            "last_payment_date": date.today().isoformat(), "customer_segment": "Salaried",
            "preferred_channel": "WhatsApp",
        }
        for column, value in backfills.items():
            execute(conn, f"UPDATE customers SET {column}=? WHERE {column} IS NULL", (value,))
        for column in ("outstanding", "emi_amount", "collection_amount_due", "collection_amount_received"):
            execute(conn, f"UPDATE customers SET {column}=0 WHERE {column} IS NULL")
        execute(conn, "UPDATE customers SET tenure_months=36 WHERE tenure_months IS NULL")

        user_count = execute(conn, "SELECT COUNT(*) FROM app_users").fetchone()[0]
        if not user_count:
            demo_users = [("bankho", "Head Office User", "HO", None)]
            demo_users.extend(
                (f"bankuser{i:03}", f"{city} Branch User", "BRANCH", branch)
                for i, (branch, city, _state) in enumerate(locations, start=1)
            )
            for username, display_name, role, branch_code in demo_users:
                salt = secrets.token_hex(16)
                password_hash = _password_hash(username, salt)
                execute(
                    conn,
                    "INSERT INTO app_users (username,display_name,role,branch_code,password_salt,password_hash,active) VALUES (?,?,?,?,?,?,1)",
                    (username, display_name, role, branch_code, salt, password_hash),
                )


def rows(sql, params=()):
    with connect() as conn:
        cur = execute(conn, sql, params)
        names = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=names)


def event(cid, stage, action, detail, actor="workflow"):
    with connect() as conn:
        execute(conn, "INSERT INTO journey_events (customer_id,stage,action,detail,actor,policy_version,created_at) VALUES (?,?,?,?,?,?,?)",
                (cid, stage, action, detail, actor, POLICY_VERSION, datetime.now(timezone.utc).isoformat()))


def score(c):
    income = float(c["income"])
    bureau = int(c["bureau"])
    emi = float(c["current_emi"])
    util = float(c["utilization"])
    delin = int(c["delinquencies"])
    cash = float(c["aa_cashflow"])
    reasons = []
    if not 21 <= int(c["age"]) <= 65: reasons.append("AGE_RANGE")
    if income < 20000: reasons.append("LOW_INCOME")
    if bureau < 650: reasons.append("BUREAU_FLOOR")
    if delin: reasons.append("ACTIVE_DELINQUENCY")
    if not int(c["kyc"]): reasons.append("KYC_PENDING")
    risk_points = max(0, (780-bureau)*.33) + util*34 + delin*38 + max(0, 1-cash)*28 + emi/income*25
    risk = min(.85, max(.015, 1/(1+math.exp(-(risk_points-35)/12))))
    propensity = min(.91, max(.04, .16 + (bureau-650)/700 + (income-20000)/250000 + cash*.15 - util*.13 - risk*.11))
    headroom = max(0, income*.45-emi)
    rate = 12.5 if risk < .20 else 15.5 if risk < .40 else 19.0
    monthly = rate/1200
    max_amount = headroom * (1-(1+monthly)**-36)/monthly if monthly else headroom*36
    amount = min(float(c["requested_amount"]), max_amount, 750000)
    amount = max(0, math.floor(amount/5000)*5000)
    if risk >= .60: reasons.append("HIGH_DEFAULT_RISK")
    if amount < 25000: reasons.append("AFFORDABILITY")
    if reasons:
        segment = "review" if reasons == ["KYC_PENDING"] else "decline"
    elif risk < .20 and bureau >= 740 and cash >= .9:
        segment = "pre-approved"
    elif risk < .45:
        segment = "pre-qualified"
    else:
        segment = "underwriting"
    if not reasons and risk >= .45:
        reasons.append("MANUAL_UNDERWRITING")
    return dict(risk=round(risk, 3), propensity=round(propensity, 3), amount=amount,
                rate=rate, segment=segment, reasons=", ".join(reasons) or "PASS")


def scored_customers(branch_code=None):
    if branch_code:
        df = rows("SELECT * FROM customers WHERE branch_code=?", (branch_code,))
    else:
        df = rows("SELECT * FROM customers")
    return pd.concat([df, df.apply(score, axis=1, result_type="expand")], axis=1)


def authenticate_user(username, password):
    username = (username or "").strip().lower()
    if not username or not password:
        return None
    user = rows(
        "SELECT username,display_name,role,branch_code,password_salt,password_hash,active "
        "FROM app_users WHERE username=?",
        (username,),
    )
    if user.empty or not int(user.iloc[0]["active"]):
        return None
    record = user.iloc[0]
    candidate = _password_hash(password, record["password_salt"])
    if not hmac.compare_digest(candidate, record["password_hash"]):
        return None
    return {
        "username": record["username"],
        "display_name": record["display_name"],
        "role": record["role"],
        "branch_code": record["branch_code"] if pd.notna(record["branch_code"]) else None,
    }


def upsert_application(cid, status, amount, rate, term, source):
    with connect() as conn:
        execute(conn, """INSERT INTO applications (customer_id,status,amount,rate,term,source,updated_at)
            VALUES (?,?,?,?,?,?,?) ON CONFLICT(customer_id) DO UPDATE SET status=excluded.status,
            amount=excluded.amount,rate=excluded.rate,term=excluded.term,source=excluded.source,
            updated_at=excluded.updated_at""",
            (cid,status,amount,rate,term,source,datetime.now(timezone.utc).isoformat()))


def retrieve_policy(query, top=3):
    sections = [s.strip() for s in POLICY.split("\n## ") if s.strip()]
    terms = set(query.lower().replace("-", " ").split())
    ranked = sorted(sections, key=lambda s: len(terms & set(s.lower().replace("-", " ").split())), reverse=True)
    return ranked[:top]
