#!/usr/bin/env python3
"""
Fundfolio — Indian Stock Market Complete Platform (Unified)

Combines:
  • stock_market_platform.py — Full analysis, PDF extraction, backtesting,
    portfolio management, tax compliance, alerts, education center
  • stock_selection_engine.py — End-to-end 3-phase stock selection pipeline
    (Momentum Scan → Pre-Open Gap → Live ORB/VWAP)

Enhanced version with improved caching, error handling, and full integration.
Run: streamlit run fundfolio_platform.py
"""

import re
import logging
import math
import os
import tempfile
import shutil
import gc
import time
import random
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import concurrent.futures
from io import BytesIO
import queue
import requests

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import streamlit as st
import pdfplumber
import yfinance as yf

# Optional Plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# -------------------------------------------------------
# CONFIGURATION & LOGGING SETUP
# -------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional OCR
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.info("OCR not available. Scanned PDFs will not be processed.")

# Optional vectorbt for backtesting
try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    logger.warning("vectorbt not installed. Backtesting module disabled.")

@dataclass
class Config:
    default_max_pages: int = 120
    max_reasonable_value: float = 1e12
    numeric_column_threshold: float = 0.3

    stcg_equity_rate: float = 15.0
    ltcg_equity_rate: float = 10.0
    ltcg_exemption: float = 100000
    stt_equity: float = 0.1
    brokerage_pct: float = 0.05

    risk_free_rate: float = 7.0
    graham_base_yield: float = 4.4

    stock_cache_ttl: int = 600
    index_cache_ttl: int = 600
    movers_cache_ttl: int = 600
    tech_cache_ttl: int = 300

    api_min_interval: float = 0.5
    max_pdf_size_mb: int = 50

    enable_performance_logging: bool = True
    slow_operation_threshold: float = 2.0

    max_concurrent_workers: int = 3
    alert_check_interval: int = 60

CONFIG = Config()

def validate_config() -> None:
    assert CONFIG.stock_cache_ttl >= 60, "Cache TTL too short"
    assert CONFIG.api_min_interval >= 0.1, "API interval too short"
    assert CONFIG.default_max_pages <= 500, "Max pages too high"
    logger.info("Configuration validated")

validate_config()

IST = timezone(timedelta(hours=5, minutes=30))

def get_ist_time() -> datetime:
    return datetime.now(IST)

def is_market_open() -> bool:
    now = get_ist_time()
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


@st.cache_data(ttl=60, show_spinner=False)
def fetch_nse_option_chain(symbol: str = "NIFTY") -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    try:
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.nseindia.com/option-chain",
            "Connection": "keep-alive",
        }
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol.upper()}"
        resp = session.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        records = payload.get("records", {})
        data = records.get("data", [])
        underlying = records.get("underlyingValue")
        expiry_dates = records.get("expiryDates", [])

        rows = []
        for item in data:
            strike = item.get("strikePrice")
            ce = item.get("CE", {}) or {}
            pe = item.get("PE", {}) or {}
            if strike is None:
                continue
            rows.append({
                "Strike": strike,
                "Call OI": ce.get("openInterest"),
                "Call Chg OI": ce.get("changeinOpenInterest"),
                "Call LTP": ce.get("lastPrice"),
                "Call Volume": ce.get("totalTradedVolume"),
                "Put LTP": pe.get("lastPrice"),
                "Put Volume": pe.get("totalTradedVolume"),
                "Put Chg OI": pe.get("changeinOpenInterest"),
                "Put OI": pe.get("openInterest"),
                "Call IV": ce.get("impliedVolatility"),
                "Put IV": pe.get("impliedVolatility"),
            })

        df = pd.DataFrame(rows).sort_values("Strike") if rows else pd.DataFrame()
        meta = {
            "underlying": underlying,
            "expiry_dates": expiry_dates,
            "timestamp": records.get("timestamp"),
            "symbol": symbol.upper(),
        }
        return df, meta
    except Exception as exc:
        logger.warning(f"Option chain fetch failed for {symbol}: {exc}")
        return None, {"error": str(exc), "symbol": symbol.upper()}

# -------------------------------------------------------
# PERFORMANCE LOGGING
# -------------------------------------------------------
def log_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not CONFIG.enable_performance_logging:
            return func(*args, **kwargs)
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        if elapsed > CONFIG.slow_operation_threshold:
            logger.warning(f"Slow operation: {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

# -------------------------------------------------------
# PRE-COMPILED REGEX PATTERNS
# -------------------------------------------------------
HEADER_PATTERN = re.compile(r'(?i)^\s*(?:fy\s*|as at|note\s*\d*|particulars|year|\d{4}[-\s]?\d{2,4})\s*$')
COMPANY_SUFFIX_PATTERN = re.compile(r'\b(limited|ltd|private|pvt|industries|corporation|company|co\.)\b', re.IGNORECASE)
ALL_CAPS_PATTERN = re.compile(r'^[A-Z][A-Z\s\.\-&]{4,}$')
NOISE_PATTERN = re.compile(r'^\d+$|page|contents|index|table of', re.IGNORECASE)
METADATA_FILTER_PATTERN = re.compile(r'(microsoft|adobe|acrobat|pdf|www\.)', re.IGNORECASE)
YEAR_PATTERN = re.compile(r'(20\d{2})')
STOCK_FILENAME_CLEAN = re.compile(r'(?i)(annual|report|ar|financial|statement|fy|_\d{4}.*)')
UNIT_MULTIPLIER_PATTERN = re.compile(r'(?:in\s+)?(?:rs\.?\s*|₹\s*)?(crores?|lakhs?|millions?)', re.IGNORECASE)

# -------------------------------------------------------
# HELPER FUNCTIONS – FORMATTING
# -------------------------------------------------------
def format_indian_number(value: Optional[float], is_share_price: bool = False) -> str:
    if value is None or pd.isna(value):
        return "-"
    if is_share_price:
        return f"₹{value:,.2f}"
    if abs(value) >= 1e7:
        return f"₹{value/1e7:,.2f} Cr"
    elif abs(value) >= 1e5:
        return f"₹{value/1e5:,.2f} L"
    else:
        return f"₹{value:,.2f}"

def format_indian_series(series: pd.Series, is_share_price: bool = False) -> pd.Series:
    if is_share_price:
        return series.map('₹{:,.2f}'.format).fillna("-")
    def format_val(val):
        if pd.isna(val):
            return "-"
        if abs(val) >= 1e7:
            return f"₹{val/1e7:,.2f} Cr"
        elif abs(val) >= 1e5:
            return f"₹{val/1e5:,.2f} L"
        else:
            return f"₹{val:,.2f}"
    return series.map(format_val)

def format_percent(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}%"

def safe_divide(numerator: pd.Series, denominator: pd.Series, default=None) -> pd.Series:
    result = pd.Series(index=numerator.index, dtype=float)
    mask = (denominator != 0) & denominator.notna() & numerator.notna()
    result[mask] = numerator[mask] / denominator[mask]
    result[~mask] = default
    return result

def sanitize_ticker(ticker: str) -> str:
    if not ticker:
        return ""
    sanitized = re.sub(r'[^A-Z0-9.\-^]', '', ticker.upper().strip())
    return sanitized[:20]

def normalize_ticker(ticker: str, exchange: str = "NSE") -> str:
    ticker = sanitize_ticker(ticker)
    if not ticker:
        return ""
    if ticker.startswith('^'):
        return ticker
    if '.' in ticker:
        return ticker
    suffix = ".NS" if exchange == "NSE" else ".BO"
    return ticker + suffix

def validate_number_input(value: float, min_val: float = 0, max_val: float = float('inf')) -> bool:
    if value is None or pd.isna(value):
        return False
    return min_val <= value <= max_val

# -------------------------------------------------------
# OPTIONS ANALYZER (Black-Scholes)
# -------------------------------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0 or sigma <= 0:
        if option_type == 'call':
            return max(0.0, S - K)
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def analyze_option(S: float, K: float, premium: float, T_years: float, option_type: str = 'call') -> Dict[str, Any]:
    if option_type == 'call':
        intrinsic_val = max(0.0, S - K)
        status = 'In the Money' if S > K else 'Out of the Money'
        break_even = K + premium
    else:
        intrinsic_val = max(0.0, K - S)
        status = 'In the Money' if S < K else 'Out of the Money'
        break_even = K - premium

    return {
        'type': option_type.upper(),
        'spot': S,
        'strike': K,
        'premium': premium,
        'time_years': T_years,
        'time_days': round(T_years * 365),
        'intrinsic_value': intrinsic_val,
        'break_even': break_even,
        'status': status,
        'buyer_max_loss': premium,
        'buyer_profit_note': 'Unlimited upside' if option_type == 'call' else 'Profit improves as spot falls',
        'seller_max_profit': premium,
        'seller_risk_note': 'Unlimited risk' if option_type == 'call' else 'Substantial downside risk',
    }


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Dict[str, float]:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf = _norm_pdf(d1)
    sqrt_t = math.sqrt(T)
    gamma = pdf / (S * sigma * sqrt_t)
    vega = S * pdf * sqrt_t / 100.0

    if option_type == 'call':
        delta = _norm_cdf(d1)
        theta = (-(S * pdf * sigma) / (2 * sqrt_t) - r * K * math.exp(-r * T) * _norm_cdf(d2)) / 365.0
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = (-(S * pdf * sigma) / (2 * sqrt_t) + r * K * math.exp(-r * T) * _norm_cdf(-d2)) / 365.0

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
    }


def option_payoff_at_expiry(spot: float, strike: float, premium: float, option_type: str = 'call', side: str = 'long') -> float:
    if option_type == 'call':
        intrinsic = max(0.0, spot - strike)
    else:
        intrinsic = max(0.0, strike - spot)

    payoff = intrinsic - premium
    if side == 'short':
        payoff = -payoff
    return payoff


def build_payoff_frame(spot: float, strike: float, call_premium: float, put_premium: float, option_type: str) -> pd.DataFrame:
    lower = max(0.0, spot * 0.5)
    upper = max(spot * 1.5, strike * 1.5)
    prices = np.linspace(lower, upper, 31)
    rows = []
    for px in prices:
        rows.append({
            'Spot': px,
            'Long Call': option_payoff_at_expiry(px, strike, call_premium, 'call', 'long'),
            'Short Call': option_payoff_at_expiry(px, strike, call_premium, 'call', 'short'),
            'Long Put': option_payoff_at_expiry(px, strike, put_premium, 'put', 'long'),
            'Short Put': option_payoff_at_expiry(px, strike, put_premium, 'put', 'short'),
        })
    df = pd.DataFrame(rows)
    if option_type == 'call':
        df['Selected'] = df['Long Call']
    else:
        df['Selected'] = df['Long Put']
    return df


def strategy_breakeven(strategy: str, strike: float, call_premium: float, put_premium: float, stock_entry: float) -> Union[float, Tuple[float, float]]:
    if strategy in {"Long Call", "Short Call"}:
        return strike + call_premium
    if strategy in {"Long Put", "Short Put"}:
        return strike - put_premium
    if strategy == "Long Straddle":
        total = call_premium + put_premium
        return (strike - total, strike + total)
    if strategy == "Short Straddle":
        total = call_premium + put_premium
        return (strike - total, strike + total)
    if strategy == "Covered Call":
        return stock_entry + call_premium
    if strategy == "Protective Put":
        return stock_entry + put_premium
    return strike

# -------------------------------------------------------
# THREAD-SAFE RATE LIMITING DECORATOR (per‑user) with global semaphore
# -------------------------------------------------------
_rate_limit_last_called = {}
_global_semaphore = threading.Semaphore(CONFIG.max_concurrent_workers)

def rate_limit(min_interval: float = 0.5):
    def decorator(func):
        lock = threading.Lock()
        @wraps(func)
        def wrapper(*args, **kwargs):
            with _global_semaphore:
                key = func.__name__
                user_id = st.session_state.get('user_id', 'global')
                full_key = f"{user_id}:{key}"
                with lock:
                    now = time.time()
                    if full_key in _rate_limit_last_called:
                        elapsed = now - _rate_limit_last_called[full_key]
                        if elapsed < min_interval:
                            time.sleep(min_interval - elapsed)
                    _rate_limit_last_called[full_key] = time.time()
                return func(*args, **kwargs)
        return wrapper
    return decorator

# -------------------------------------------------------
# PDF EXTRACTION (with OCR fallback)
# -------------------------------------------------------
def extract_text_with_ocr(pdf_path: str, page_num: int) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        images = convert_from_path(pdf_path, dpi=300, first_page=page_num+1, last_page=page_num+1)
        if images:
            return pytesseract.image_to_string(images[0])
    except Exception as e:
        logger.warning(f"OCR failed on page {page_num}: {e}")
    return ""

def clean_number(text: Any) -> Optional[float]:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip().replace('₹', '').replace(',', '')
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    cleaned = re.sub(r'[^\d\.\-]', '', s)
    try:
        val = float(cleaned)
        if abs(val) > CONFIG.max_reasonable_value:
            return None
        return val
    except Exception as e:
        logger.debug(f"clean_number failed on {text}: {e}")
        return None

def detect_unit_multiplier(text: str) -> Optional[float]:
    text_lower = text.lower()
    if re.search(r'(?:in\s+)?(?:rs\.?\s*|₹\s*)?crores?\b', text_lower):
        return 10_000_000
    if re.search(r'(?:in\s+)?(?:rs\.?\s*|₹\s*)?lakhs?\b', text_lower):
        return 100_000
    if re.search(r'(?:in\s+)?(?:rs\.?\s*|₹\s*)?millions?\b', text_lower):
        return 1_000_000
    return None

def _is_header_cell(cell: Any) -> bool:
    if cell is None:
        return False
    s = str(cell).strip()
    if not s:
        return False
    return bool(HEADER_PATTERN.match(s))

def find_numeric_columns(table: List[List[Any]], threshold: float = 0.3) -> List[int]:
    if not table:
        return[]
    num_rows = len(table)
    num_cols = max(len(row) for row in table) if table else 0
    candidates =[]
    for col in range(num_cols):
        count = 0
        for row in table:
            if col < len(row) and row[col] is not None:
                cell = str(row[col]).strip()
                if cell and not _is_header_cell(cell) and clean_number(cell) is not None:
                    count += 1
        if count >= num_rows * threshold:
            candidates.append(col)
    return candidates

def _normalise_row_text(row: List[Any]) -> str:
    parts =[]
    for cell in row:
        if cell is None:
            continue
        s = str(cell).strip()
        if s:
            s = re.sub(r'[\n\r\t]+', ' ', s)
            s = re.sub(r'\s{2,}', ' ', s)
            parts.append(s.strip())
    return ' '.join(parts).lower()

def extract_metric_from_table(table: List[List[Any]], keywords: List[str]) -> Optional[float]:
    if not table:
        return None
    numeric_cols = find_numeric_columns(table)
    if not numeric_cols:
        return None
    keywords_lower =[kw.lower() for kw in keywords]
    for row in table:
        row_text = _normalise_row_text(row)
        for kw in keywords_lower:
            if kw in row_text:
                for col in numeric_cols:
                    if col < len(row):
                        val = clean_number(row[col])
                        if val is not None:
                            return val
    return None

def _extract_company_from_page_text(text: str) -> Optional[str]:
    lines = text.split('\n')
    candidates = []
    for line in lines[:20]:
        line = line.strip()
        if not line or len(line) > 80:
            continue
        if NOISE_PATTERN.search(line):
            continue
        if COMPANY_SUFFIX_PATTERN.search(line):
            cleaned = re.sub(r'\s+', ' ', line).strip()
            candidates.append(cleaned)
        elif ALL_CAPS_PATTERN.match(line) and len(line) > 5:
            candidates.append(line)
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]
    return None

def extract_company_year_from_pdf(pdf_path: str, max_pages_to_scan: int = 10) -> Tuple[str, str]:
    company = "Unknown"
    year = "Unknown"
    base = os.path.basename(pdf_path)
    y_match = YEAR_PATTERN.search(base)
    if y_match:
        year = y_match.group(1)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            meta = pdf.metadata or {}
            for field in ('Title', 'Author', 'Subject', 'Creator'):
                val = meta.get(field, '').strip()
                if val and len(val) > 3:
                    if not METADATA_FILTER_PATTERN.search(val):
                        if company == "Unknown":
                            company = val[:80]
                        elif len(val) > len(company) and len(val) < 80:
                            company = val[:80]
                if year == "Unknown":
                    y = YEAR_PATTERN.search(val)
                    if y:
                        year = y.group(1)
            for page_idx in range(min(max_pages_to_scan, len(pdf.pages))):
                text = extract_text_safely(pdf.pages[page_idx])
                if not text.strip():
                    if OCR_AVAILABLE and page_idx < 5:
                        text = extract_text_with_ocr(pdf_path, page_idx)
                    if not text.strip():
                        continue
                if year == "Unknown":
                    y = YEAR_PATTERN.search(text)
                    if y:
                        year = y.group(1)
                if company == "Unknown":
                    extracted = _extract_company_from_page_text(text)
                    if extracted:
                        company = extracted[:80]
                if company != "Unknown" and year != "Unknown":
                    break
    except Exception as e:
        logger.warning(f"PDF extraction error for {base}: {e}")
    if company == "Unknown":
        stem = os.path.splitext(base)[0]
        clean_stem = STOCK_FILENAME_CLEAN.sub('', stem).strip()
        if clean_stem and len(clean_stem) > 3:
            company = clean_stem[:80]
    return company, year

def extract_tables_safely(page) -> List[List[Any]]:
    try:
        tables = page.extract_tables()
        return tables if tables is not None else[]
    except Exception as e:
        logger.warning(f"Table extraction failed: {e}")
        return[]

def extract_text_safely(page) -> str:
    try:
        text = page.extract_text()
        return text if text is not None else ""
    except Exception as e:
        logger.warning(f"Text extraction failed: {e}")
        return ""

def find_financial_pages(pdf, max_pages: int) -> List[int]:
    target_pages =[]
    for i in range(min(max_pages, len(pdf.pages))):
        try:
            text = extract_text_safely(pdf.pages[i])
            if any(kw in text.lower() for kw in ['balance sheet', 'profit and loss', 'income statement']):
                target_pages.append(i)
        except:
            continue
    return target_pages

def extract_financial_data(pdf_path: str, max_pages: int = 120) -> Dict[str, Any]:
    data = {}
    company, year = extract_company_year_from_pdf(pdf_path)
    data['Company'] = company
    data['Year'] = year
    data['_extraction_success'] = False
    data['_period'] = 'annual'
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_multipliers = {}
            for i in range(min(10, len(pdf.pages))):
                text = extract_text_safely(pdf.pages[i])
                m = detect_unit_multiplier(text)
                if m:
                    page_multipliers[i] = m
            default_multiplier = 1.0
            data['_unit_multiplier'] = default_multiplier

            target_pages = find_financial_pages(pdf, max_pages)
            if not target_pages and OCR_AVAILABLE:
                st.warning("⚠️ No text‑based financial pages found. OCR can only extract company name, not table data. Consider using OCR‑processed PDFs for better results.")
            metrics_found = 0
            for page_num in target_pages:
                if metrics_found >= len(METRIC_KEYWORDS) * 0.8:
                    break
                page = pdf.pages[page_num]
                tables = extract_tables_safely(page)
                page_text = extract_text_safely(page)
                multiplier = page_multipliers.get(page_num, default_multiplier)
                for table in tables:
                    for metric, keywords in METRIC_KEYWORDS.items():
                        if metric not in data:
                            val = extract_metric_from_table(table, keywords)
                            if val is not None:
                                scale = 1.0 if metric in PER_SHARE_METRICS else multiplier
                                data[metric] = val * scale
                                data['_extraction_success'] = True
                                metrics_found += 1
                for metric, pattern in TEXT_PATTERNS.items():
                    if metric not in data:
                        match = re.search(pattern, page_text, re.IGNORECASE)
                        if match:
                            val = clean_number(match.group(1))
                            if val is not None:
                                scale = 1.0 if metric in PER_SHARE_METRICS else multiplier
                                data[metric] = val * scale
                                data['_extraction_success'] = True
                                metrics_found += 1
            if 'Share Capital' in data and data.get('Face Value', 0) > 0.01:
                data['_share_count'] = data['Share Capital'] / data['Face Value']
            elif 'EPS' in data and 'Net Profit' in data and data.get('EPS', 0) != 0:
                data['_share_count'] = data['Net Profit'] / data['EPS']
            if not metrics_found and OCR_AVAILABLE:
                st.warning("⚠️ No financial metrics could be extracted from scanned pages. The PDF may be image‑based.")
    except Exception as e:
        logger.error(f"Extraction failed for {pdf_path}: {e}")
        data['_error'] = str(e)
    finally:
        gc.collect()
    return data

METRIC_KEYWORDS: Dict[str, List[str]] = {
    'Revenue':['revenue from operations', 'total revenue', 'net sales', 'sales'],
    'Net Profit':['profit for the year', 'profit after tax', 'net profit'],
    'Equity':["total shareholders' funds", "shareholders' funds", "shareholder's equity", 'total equity'],
    'Inventory':['total inventories', 'inventories', 'closing stock'],
    'Receivables': ['trade receivables', 'accounts receivable', 'debtors'],
    'Fixed Assets': ['property plant and equipment', 'tangible assets', 'net block', 'fixed assets'],
    'Current Assets': ['total current assets', 'current assets'],
    'Current Liabilities':['total current liabilities', 'current liabilities'],
    'Total Assets': ['total assets'],
    'COGS':['cost of goods sold', 'cost of materials consumed'],
    'EPS': ['earnings per share', 'eps'],
    'EBIT': ['profit before interest and tax', 'operating profit', 'ebit'],
    'EBITDA': ['ebitda'],
    'Share Capital': ['equity share capital', 'share capital'],
    'Reserves': ['reserves and surplus', 'reserves'],
    'Face Value':['face value per share', 'face value'],
    'Total Debt': ['total borrowings', 'total debt'],
    'Operating Cash Flow': ['net cash from operating activities', 'cash generated from operations'],
    'Interest':['finance costs', 'interest expense'],
    'Capex': ['purchase of property plant and equipment', 'capital expenditure'],
    'Cash': ['cash and cash equivalents', 'cash and bank balances'],
    'Depreciation':['depreciation and amortisation expense', 'depreciation'],
}
PER_SHARE_METRICS: Set[str] = {'Face Value', 'EPS'}

TEXT_PATTERNS: Dict[str, str] = {
    'Revenue': r'revenue from operations\s*[:\-–—\s]+([\d,]+(?:\.\d+)?)',
    'Net Profit': r'profit\s+(?:for the year|after tax)\s*[:\-–—\s]+([\d,]+(?:\.\d+)?)',
    'EPS': r'earnings\s+per\s+share\s*[:\-–—\s]*([\d,]+(?:\.\d+)?)',
    'EBIT': r'(?:profit before (?:interest and )?tax|operating profit|ebit)\s*[:\-–—\s]+([\d,]+(?:\.\d+)?)',
    'EBITDA': r'ebitda\s*[:\-–—\s]+([\d,]+(?:\.\d+)?)',
    'Total Debt': r'total\s+(?:debt|borrowings)\s*[:\-–—\s]+([\d,]+(?:\.\d+)?)',
    'Operating Cash Flow': r'net cash (?:from|generated from) operating activities\s*[:\-–—\s]+([\d,]+(?:\.\d+)?)',
    'Cash': r'cash and (?:cash equivalents|bank balances?)\s*[:\-–—\s]+([\d,]+(?:\.\d+)?)',
    'Interest': r'(?:finance costs?|interest expense)\s*[:\-–—\s]+([\d,]+(?:\.\d+)?)',
    'Capex': r'purchase of (?:fixed assets|property[,\s]+plant)\s*[:\-–—\s]+([\d,]+(?:\.\d+)?)',
    'Face Value': r'face value(?:\s+per\s+share)?\s*[:\-–—\s]*[₹Rs.]*\s*([\d,]+(?:\.\d+)?)',
}

# -------------------------------------------------------
# CACHING LAYER (with market hours adjustment)
# -------------------------------------------------------
def get_cache_ttl():
    if is_market_open():
        return 60
    else:
        return CONFIG.stock_cache_ttl

@st.cache_data(ttl=CONFIG.stock_cache_ttl)
def fetch_stock_data(ticker: str, period: str = '1y') -> Optional[pd.DataFrame]:
    return fetch_live_data(ticker, period)

@st.cache_data(ttl=CONFIG.tech_cache_ttl)
def compute_technical_indicators_cached(ticker: str, period: str) -> pd.DataFrame:
    df = fetch_stock_data(ticker, period)
    if df is None or df.empty:
        return df
    return calculate_technical_indicators(df.copy())

@st.cache_data(ttl=CONFIG.index_cache_ttl)
def get_cached_index_data(ticker: str):
    return get_index_data(ticker)

@st.cache_data(ttl=CONFIG.movers_cache_ttl)
def get_cached_top_movers(limit: int = 5):
    return get_top_movers(limit)

# -------------------------------------------------------
# YFINANCE HELPERS
# -------------------------------------------------------
@rate_limit(min_interval=CONFIG.api_min_interval)
@log_performance
def fetch_live_data(ticker: str, period: str = '1y', max_retries: int = 3) -> Optional[pd.DataFrame]:
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df is None or df.empty:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt + random.uniform(0, 1))
                    continue
                logger.warning(f"No data for {ticker}")
                return None
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required):
                logger.error(f"Missing columns for {ticker}")
                return None
            return df
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed for {ticker}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.uniform(0, 1))
            else:
                return None
    return None

@rate_limit(min_interval=CONFIG.api_min_interval)
def get_index_data(ticker: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1mo')
        if data is None or len(data) < 2:
            return None, None, None
        data = data.dropna()
        if len(data) < 2:
            return None, None, None
        current = float(data['Close'].iloc[-1])
        prev_close = float(data['Close'].iloc[-2])
        change = current - prev_close
        pct = (change / prev_close) * 100
        return current, change, pct
    except Exception as e:
        logger.error(f"Error fetching index {ticker}: {e}")
        return None, None, None

def get_current_price(ticker: str) -> float:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d')
        if not hist.empty and 'Close' in hist.columns:
            return float(hist['Close'].iloc[-1])
        hist = stock.history(period='5d')
        if not hist.empty and 'Close' in hist.columns:
            return float(hist['Close'].iloc[-1])
        info = stock.info
        for key in['currentPrice', 'regularMarketPrice', 'previousClose']:
            if key in info and info[key]:
                return float(info[key])
        logger.warning(f"Could not fetch price for {ticker}")
        return 0.0
    except Exception as e:
        logger.error(f"Error fetching price for {ticker}: {e}")
        return 0.0

def safe_concurrent_fetch(tickers: list, fetch_func, max_workers: int = None, timeout: int = 30):
    if max_workers is None:
        max_workers = CONFIG.max_concurrent_workers
    results =[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(fetch_func, t): t for t in tickers}
        try:
            for future in concurrent.futures.as_completed(future_to_ticker, timeout=timeout):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Timeout fetching {ticker}")
                except Exception as e:
                    logger.error(f"Error fetching {ticker}: {e}")
        except concurrent.futures.TimeoutError:
            logger.warning("Global timeout reached in concurrent fetch")
    return results

@log_performance
def get_top_movers(limit: int = 5) -> Tuple[List[Dict], List[Dict]]:
    logger.info(f"Fetching top {limit} movers from {len(NIFTY_50_DATA)} stocks")
    start = time.time()
    gainers, losers = [],[]
    tickers = list(NIFTY_50_DATA.keys())
    def process_stock(ticker):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='5d')
            if len(hist) >= 2:
                current = float(hist['Close'].iloc[-1])
                prev = None
                for i in range(2, len(hist)+1):
                    candidate = hist['Close'].iloc[-i]
                    if not pd.isna(candidate):
                        prev = float(candidate)
                        break
                if prev is None:
                    return None
                change_pct = ((current - prev) / prev) * 100
                name, sector = NIFTY_50_DATA[ticker]
                return {'Stock': name, 'Price': current, 'Change %': change_pct}
            return None
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            return None
    results = safe_concurrent_fetch(tickers, process_stock, max_workers=CONFIG.max_concurrent_workers)
    for res in results:
        if res:
            if res['Change %'] > 0:
                gainers.append(res)
            elif res['Change %'] < 0:
                losers.append(res)
    gainers.sort(key=lambda x: x['Change %'], reverse=True)
    losers.sort(key=lambda x: x['Change %'])
    elapsed = time.time() - start
    logger.info(f"Top movers fetched in {elapsed:.2f}s: {len(gainers)} gainers, {len(losers)} losers")
    return gainers[:limit], losers[:limit]

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        # ROC indicators
        df['ROC_5'] = df['Close'].pct_change(5) * 100
        df['ROC_20'] = df['Close'].pct_change(20) * 100
        # Volume ratio
        df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Volume_SMA_20'].replace(0, np.nan)
        # 52-week high/low
        df['52W_High'] = df['High'].rolling(252).max()
        df['52W_Low'] = df['Low'].rolling(252).min()
        df['Pct_From_52W_High'] = ((df['Close'] - df['52W_High']) / df['52W_High']) * 100
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        # Wilder smoothing (EWM with alpha=1/14)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (std * 2)
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        range_ = (high_14 - low_14).replace(0, np.nan)
        df['Stochastic_%K'] = 100 * (df['Close'] - low_14) / range_
        df['Stochastic_%D'] = df['Stochastic_%K'].rolling(3).mean()
    except Exception as e:
        logger.error(f"Indicator error: {e}")
    return df

# -------------------------------------------------------
# CANDLESTICK PATTERN DETECTION
# -------------------------------------------------------
class PatternSignal(Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"

@dataclass
class PatternResult:
    name: str
    signal: PatternSignal
    strength: str
    description: str
    index: int

class CandlestickPatterns:
    @staticmethod
    def _body_size(row: pd.Series) -> float:
        return abs(row['Close'] - row['Open'])
    @staticmethod
    def _upper_shadow(row: pd.Series) -> float:
        return row['High'] - max(row['Open'], row['Close'])
    @staticmethod
    def _lower_shadow(row: pd.Series) -> float:
        return min(row['Open'], row['Close']) - row['Low']
    @staticmethod
    def _total_range(row: pd.Series) -> float:
        return row['High'] - row['Low']
    @staticmethod
    def _is_bullish(row: pd.Series) -> bool:
        return row['Close'] > row['Open']
    @staticmethod
    def _is_bearish(row: pd.Series) -> bool:
        return row['Close'] < row['Open']

    @staticmethod
    def doji(df: pd.DataFrame, idx: int) -> Optional[PatternResult]:
        row = df.iloc[idx]
        body = CandlestickPatterns._body_size(row)
        total_range = CandlestickPatterns._total_range(row)
        if total_range == 0:
            return None
        if body / total_range < 0.1:
            return PatternResult("Doji", PatternSignal.NEUTRAL, "Medium",
                                 "Indecision - potential reversal signal", idx)
        return None

    @staticmethod
    def hammer(df: pd.DataFrame, idx: int) -> Optional[PatternResult]:
        row = df.iloc[idx]
        body = CandlestickPatterns._body_size(row)
        lower = CandlestickPatterns._lower_shadow(row)
        upper = CandlestickPatterns._upper_shadow(row)
        total_range = CandlestickPatterns._total_range(row)
        if total_range == 0:
            return None
        if lower >= 2 * body and upper < body * 0.3 and body / total_range < 0.3:
            return PatternResult("Hammer", PatternSignal.BULLISH, "Strong",
                                 "Bullish reversal - buying pressure after decline", idx)
        return None

    @staticmethod
    def shooting_star(df: pd.DataFrame, idx: int) -> Optional[PatternResult]:
        row = df.iloc[idx]
        body = CandlestickPatterns._body_size(row)
        lower = CandlestickPatterns._lower_shadow(row)
        upper = CandlestickPatterns._upper_shadow(row)
        total_range = CandlestickPatterns._total_range(row)
        if total_range == 0:
            return None
        if upper >= 2 * body and lower < body * 0.3 and body / total_range < 0.3:
            return PatternResult("Shooting Star", PatternSignal.BEARISH, "Strong",
                                 "Bearish reversal - selling pressure after rally", idx)
        return None

    @staticmethod
    def spinning_top(df: pd.DataFrame, idx: int) -> Optional[PatternResult]:
        row = df.iloc[idx]
        body = CandlestickPatterns._body_size(row)
        lower = CandlestickPatterns._lower_shadow(row)
        upper = CandlestickPatterns._upper_shadow(row)
        total_range = CandlestickPatterns._total_range(row)
        if total_range == 0:
            return None
        if body / total_range < 0.3 and lower > body * 0.5 and upper > body * 0.5:
            return PatternResult("Spinning Top", PatternSignal.NEUTRAL, "Weak",
                                 "Indecision - equal buying and selling pressure", idx)
        return None

    @staticmethod
    def marubozu(df: pd.DataFrame, idx: int) -> Optional[PatternResult]:
        row = df.iloc[idx]
        body = CandlestickPatterns._body_size(row)
        lower = CandlestickPatterns._lower_shadow(row)
        upper = CandlestickPatterns._upper_shadow(row)
        total_range = CandlestickPatterns._total_range(row)
        if total_range == 0:
            return None
        if body / total_range > 0.85 and lower < total_range * 0.05 and upper < total_range * 0.05:
            signal = PatternSignal.BULLISH if CandlestickPatterns._is_bullish(row) else PatternSignal.BEARISH
            direction = "Bullish" if signal == PatternSignal.BULLISH else "Bearish"
            return PatternResult(f"{direction} Marubozu", signal, "Strong",
                                 f"Strong {direction.lower()} momentum - clear directional move", idx)
        return None

    @staticmethod
    def engulfing(df: pd.DataFrame, idx: int) -> Optional[PatternResult]:
        if idx < 1:
            return None
        prev = df.iloc[idx-1]
        curr = df.iloc[idx]
        prev_body = CandlestickPatterns._body_size(prev)
        curr_body = CandlestickPatterns._body_size(curr)
        if CandlestickPatterns._is_bearish(prev) and CandlestickPatterns._is_bullish(curr) and curr['Open'] < prev['Close'] and curr['Close'] > prev['Open'] and curr_body > prev_body:
            return PatternResult("Bullish Engulfing", PatternSignal.BULLISH, "Strong",
                                 "Strong bullish reversal - buyers overwhelm sellers", idx)
        if CandlestickPatterns._is_bullish(prev) and CandlestickPatterns._is_bearish(curr) and curr['Open'] > prev['Close'] and curr['Close'] < prev['Open'] and curr_body > prev_body:
            return PatternResult("Bearish Engulfing", PatternSignal.BEARISH, "Strong",
                                 "Strong bearish reversal - sellers overwhelm buyers", idx)
        return None

    @staticmethod
    def harami(df: pd.DataFrame, idx: int) -> Optional[PatternResult]:
        if idx < 1:
            return None
        prev = df.iloc[idx-1]
        curr = df.iloc[idx]
        prev_body = CandlestickPatterns._body_size(prev)
        curr_body = CandlestickPatterns._body_size(curr)
        if (curr['Open'] < max(prev['Open'], prev['Close']) and curr['Close'] > min(prev['Open'], prev['Close']) and
            curr['Open'] > min(prev['Open'], prev['Close']) and curr['Close'] < max(prev['Open'], prev['Close']) and
            curr_body < prev_body * 0.7):
            if CandlestickPatterns._is_bearish(prev) and CandlestickPatterns._is_bullish(curr):
                return PatternResult("Bullish Harami", PatternSignal.BULLISH, "Medium",
                                     "Bullish reversal - weakening downtrend", idx)
            if CandlestickPatterns._is_bullish(prev) and CandlestickPatterns._is_bearish(curr):
                return PatternResult("Bearish Harami", PatternSignal.BEARISH, "Medium",
                                     "Bearish reversal - weakening uptrend", idx)
        return None

    @staticmethod
    def piercing_line(df: pd.DataFrame, idx: int) -> Optional[PatternResult]:
        if idx < 1:
            return None
        prev = df.iloc[idx-1]
        curr = df.iloc[idx]
        prev_mid = (prev['Open'] + prev['Close']) / 2
        if (CandlestickPatterns._is_bearish(prev) and CandlestickPatterns._is_bullish(curr) and
            curr['Open'] < prev['Low'] and curr['Close'] > prev_mid and curr['Close'] < prev['Open']):
            return PatternResult("Piercing Line", PatternSignal.BULLISH, "Strong",
                                 "Bullish reversal - strong buying after gap down", idx)
        return None

    @staticmethod
    def dark_cloud_cover(df: pd.DataFrame, idx: int) -> Optional[PatternResult]:
        if idx < 1:
            return None
        prev = df.iloc[idx-1]
        curr = df.iloc[idx]
        prev_mid = (prev['Open'] + prev['Close']) / 2
        if (CandlestickPatterns._is_bullish(prev) and CandlestickPatterns._is_bearish(curr) and
            curr['Open'] > prev['High'] and curr['Close'] < prev_mid and curr['Close'] > prev['Open']):
            return PatternResult("Dark Cloud Cover", PatternSignal.BEARISH, "Strong",
                                 "Bearish reversal - strong selling after gap up", idx)
        return None

    @staticmethod
    def morning_star(df: pd.DataFrame, idx: int) -> Optional[PatternResult]:
        if idx < 2:
            return None
        first = df.iloc[idx-2]
        second = df.iloc[idx-1]
        third = df.iloc[idx]
        first_body = CandlestickPatterns._body_size(first)
        second_body = CandlestickPatterns._body_size(second)
        if (CandlestickPatterns._is_bearish(first) and second_body < first_body * 0.5 and
            CandlestickPatterns._is_bullish(third) and third['Close'] > (first['Open'] + first['Close']) / 2):
            return PatternResult("Morning Star", PatternSignal.BULLISH, "Strong",
                                 "Strong bullish reversal - downtrend ending", idx)
        return None

    @staticmethod
    def evening_star(df: pd.DataFrame, idx: int) -> Optional[PatternResult]:
        if idx < 2:
            return None
        first = df.iloc[idx-2]
        second = df.iloc[idx-1]
        third = df.iloc[idx]
        first_body = CandlestickPatterns._body_size(first)
        second_body = CandlestickPatterns._body_size(second)
        if (CandlestickPatterns._is_bullish(first) and second_body < first_body * 0.5 and
            CandlestickPatterns._is_bearish(third) and third['Close'] < (first['Open'] + first['Close']) / 2):
            return PatternResult("Evening Star", PatternSignal.BEARISH, "Strong",
                                 "Strong bearish reversal - uptrend ending", idx)
        return None

    @staticmethod
    def three_white_soldiers(df: pd.DataFrame, idx: int) -> Optional[PatternResult]:
        if idx < 2:
            return None
        first = df.iloc[idx-2]
        second = df.iloc[idx-1]
        third = df.iloc[idx]
        if (CandlestickPatterns._is_bullish(first) and CandlestickPatterns._is_bullish(second) and CandlestickPatterns._is_bullish(third) and
            second['Close'] > first['Close'] and third['Close'] > second['Close'] and
            second['Open'] > first['Open'] and third['Open'] > second['Open']):
            return PatternResult("Three White Soldiers", PatternSignal.BULLISH, "Strong",
                                 "Strong bullish momentum - sustained buying pressure", idx)
        return None

    @staticmethod
    def three_black_crows(df: pd.DataFrame, idx: int) -> Optional[PatternResult]:
        if idx < 2:
            return None
        first = df.iloc[idx-2]
        second = df.iloc[idx-1]
        third = df.iloc[idx]
        if (CandlestickPatterns._is_bearish(first) and CandlestickPatterns._is_bearish(second) and CandlestickPatterns._is_bearish(third) and
            second['Close'] < first['Close'] and third['Close'] < second['Close'] and
            second['Open'] < first['Open'] and third['Open'] < second['Open']):
            return PatternResult("Three Black Crows", PatternSignal.BEARISH, "Strong",
                                 "Strong bearish momentum - sustained selling pressure", idx)
        return None

class PatternScanner:
    PATTERNS =[
        CandlestickPatterns.doji, CandlestickPatterns.hammer, CandlestickPatterns.shooting_star,
        CandlestickPatterns.spinning_top, CandlestickPatterns.marubozu, CandlestickPatterns.engulfing,
        CandlestickPatterns.harami, CandlestickPatterns.piercing_line, CandlestickPatterns.dark_cloud_cover,
        CandlestickPatterns.morning_star, CandlestickPatterns.evening_star,
        CandlestickPatterns.three_white_soldiers, CandlestickPatterns.three_black_crows,
    ]
    @staticmethod
    def scan(df: pd.DataFrame, last_n: int = 20) -> List[PatternResult]:
        if df is None or df.empty or len(df) < 3:
            return []
        patterns =[]
        start = max(0, len(df) - last_n)
        for idx in range(start, len(df)):
            for pfunc in PatternScanner.PATTERNS:
                res = pfunc(df, idx)
                if res:
                    patterns.append(res)
        return patterns
    @staticmethod
    def get_latest_pattern(df: pd.DataFrame) -> Optional[PatternResult]:
        pats = PatternScanner.scan(df, last_n=5)
        return max(pats, key=lambda p: p.index) if pats else None
    @staticmethod
    def get_bullish_patterns(df: pd.DataFrame, last_n: int = 20) -> List[PatternResult]:
        return[p for p in PatternScanner.scan(df, last_n) if p.signal == PatternSignal.BULLISH]
    @staticmethod
    def get_bearish_patterns(df: pd.DataFrame, last_n: int = 20) -> List[PatternResult]:
        return[p for p in PatternScanner.scan(df, last_n) if p.signal == PatternSignal.BEARISH]
    @staticmethod
    def format_pattern_summary(patterns: List[PatternResult]) -> str:
        if not patterns:
            return "No patterns detected"
        lines =[]
        for p in patterns:
            emoji = "🟢" if p.signal == PatternSignal.BULLISH else "🔴" if p.signal == PatternSignal.BEARISH else "🟡"
            lines.append(f"{emoji} {p.name} ({p.strength}) - {p.description}")
        return "\n".join(lines)

# -------------------------------------------------------
# DATABASE INTEGRATION (PostgreSQL / MongoDB)
# -------------------------------------------------------
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
    from sqlalchemy.orm import sessionmaker
    try:
        # SQLAlchemy 2.x
        from sqlalchemy.orm import declarative_base
    except ImportError:
        # SQLAlchemy 1.x fallback
        from sqlalchemy.ext.declarative import declarative_base
    SQLALCHEMY_AVAILABLE = True
    Base = declarative_base()
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    from pymongo import MongoClient
    from bson.objectid import ObjectId
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

logger_db = logging.getLogger(__name__)

if SQLALCHEMY_AVAILABLE:
    class Portfolio(Base):
        __tablename__ = 'portfolios'
        id = Column(Integer, primary_key=True)
        user_id = Column(String(100), index=True)
        stock_name = Column(String(200))
        ticker = Column(String(50))
        shares = Column(Integer)
        buy_price = Column(Float)
        current_price = Column(Float)
        total_cost = Column(Float)
        current_value = Column(Float)
        pnl = Column(Float)
        pnl_pct = Column(Float)
        sector = Column(String(100))
        purchase_date = Column(String(50))
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    class Trade(Base):
        __tablename__ = 'trades'
        id = Column(Integer, primary_key=True)
        user_id = Column(String(100), index=True)
        trade_date = Column(String(50))
        trade_type = Column(String(20))
        stock = Column(String(200))
        quantity = Column(Integer)
        entry = Column(Float)
        exit_price = Column(Float, nullable=True)
        stop_loss = Column(Float)
        target = Column(Float)
        strategy = Column(String(100))
        notes = Column(String(1000))
        pnl = Column(Float, nullable=True)
        pnl_pct = Column(Float, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)

    class Alert(Base):
        __tablename__ = 'alerts'
        id = Column(Integer, primary_key=True)
        user_id = Column(String(100), index=True)
        ticker = Column(String(50))
        alert_type = Column(String(50))
        threshold = Column(Float)
        is_active = Column(Integer, default=1)
        email = Column(String(200))
        last_triggered = Column(DateTime, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)


def get_postgres_connection():
    try:
        conn = st.connection("postgresql", type="sql")
        return conn
    except Exception:
        return None

class PostgreSQLDatabase:
    def __init__(self, connection_string: Optional[str] = None):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy not installed")
        self.connection_string = connection_string or os.getenv('DATABASE_URL', '')
        if not self.connection_string:
            try:
                self.connection_string = st.secrets["DATABASE_URL"]
            except (KeyError, FileNotFoundError):
                logger.error("No database connection string provided.")
                raise ValueError("Database connection string missing. Please add DATABASE_URL to .streamlit/secrets.toml")
        self._conn = None
        self._engine = None
        self._session = None
        self._transaction = False
        logger_db.info("PostgreSQL connection created (pooled)")

    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine(self.connection_string)
        return self._engine

    @property
    def session(self):
        if self._session is None:
            Session = sessionmaker(bind=self.engine)
            self._session = Session()
        return self._session

    def begin_transaction(self) -> None:
        if not self._transaction:
            self.session.begin()
            self._transaction = True
            logger_db.info("Transaction started")

    def commit(self) -> None:
        if self._transaction:
            self.session.commit()
            self._transaction = False
            logger_db.info("Transaction committed")

    def rollback(self) -> None:
        if self._transaction:
            self.session.rollback()
            self._transaction = False
            logger_db.warning("Transaction rolled back")

    def save_portfolio(self, user_id: str, positions: List[Dict]) -> bool:
        try:
            self.session.query(Portfolio).filter_by(user_id=user_id).delete()
            for pos in positions:
                portfolio = Portfolio(
                    user_id=user_id,
                    stock_name=pos.get('Stock'),
                    ticker=pos.get('Ticker', '').upper(),
                    shares=pos.get('Shares'),
                    buy_price=pos.get('Buy Price'),
                    current_price=pos.get('Current Price'),
                    total_cost=pos.get('Total Cost'),
                    current_value=pos.get('Current Value'),
                    pnl=pos.get('P&L'),
                    pnl_pct=pos.get('P&L %'),
                    sector=pos.get('Sector'),
                    purchase_date=pos.get('Date')
                )
                self.session.add(portfolio)
            self.session.commit()
            logger_db.info(f"Saved {len(positions)} positions for user {user_id}")
            return True
        except Exception as e:
            self.session.rollback()
            logger_db.error(f"Error saving portfolio: {e}")
            return False

    def load_portfolio(self, user_id: str) -> List[Dict]:
        try:
            positions = self.session.query(Portfolio).filter_by(user_id=user_id).all()
            return[{
                'Stock': p.stock_name, 'Ticker': p.ticker, 'Shares': p.shares,
                'Buy Price': p.buy_price, 'Current Price': p.current_price,
                'Total Cost': p.total_cost, 'Current Value': p.current_value,
                'P&L': p.pnl, 'P&L %': p.pnl_pct, 'Sector': p.sector,
                'Date': p.purchase_date
            } for p in positions]
        except Exception as e:
            logger_db.error(f"Error loading portfolio: {e}")
            return[]

    def add_position(self, user_id: str, position: Dict) -> bool:
        try:
            portfolio = Portfolio(
                user_id=user_id,
                stock_name=position.get('Stock'),
                ticker=position.get('Ticker', '').upper(),
                shares=position.get('Shares'),
                buy_price=position.get('Buy Price'),
                current_price=position.get('Current Price'),
                total_cost=position.get('Total Cost'),
                current_value=position.get('Current Value'),
                pnl=position.get('P&L'),
                pnl_pct=position.get('P&L %'),
                sector=position.get('Sector'),
                purchase_date=position.get('Date')
            )
            self.session.add(portfolio)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            logger_db.error(f"Error adding position: {e}")
            return False

    def delete_position(self, user_id: str, ticker: str) -> bool:
        try:
            ticker = ticker.upper()
            self.session.query(Portfolio).filter_by(user_id=user_id, ticker=ticker).delete()
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            logger_db.error(f"Error deleting position: {e}")
            return False

    def save_trade(self, user_id: str, trade_data: Dict) -> bool:
        try:
            trade = Trade(
                user_id=user_id,
                trade_date=trade_data.get('Date'),
                trade_type=trade_data.get('Type'),
                stock=trade_data.get('Stock'),
                quantity=trade_data.get('Quantity'),
                entry=trade_data.get('Entry'),
                exit_price=trade_data.get('Exit'),
                stop_loss=trade_data.get('Stop Loss'),
                target=trade_data.get('Target'),
                strategy=trade_data.get('Strategy'),
                notes=trade_data.get('Notes'),
                pnl=trade_data.get('P&L'),
                pnl_pct=trade_data.get('P&L %')
            )
            self.session.add(trade)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            logger_db.error(f"Error saving trade: {e}")
            return False

    def load_trades(self, user_id: str) -> List[Dict]:
        try:
            trades = self.session.query(Trade).filter_by(user_id=user_id).all()
            return[{
                'Date': t.trade_date, 'Type': t.trade_type, 'Stock': t.stock,
                'Quantity': t.quantity, 'Entry': t.entry, 'Exit': t.exit_price,
                'Stop Loss': t.stop_loss, 'Target': t.target, 'Strategy': t.strategy,
                'Notes': t.notes, 'P&L': t.pnl, 'P&L %': t.pnl_pct
            } for t in trades]
        except Exception as e:
            logger_db.error(f"Error loading trades: {e}")
            return[]

    def save_portfolio_with_trades(self, user_id: str, portfolio: List[Dict], trades: List[Dict]) -> bool:
        try:
            self.begin_transaction()
            self.save_portfolio(user_id, portfolio)
            for trade in trades:
                self.save_trade(user_id, trade)
            self.commit()
            return True
        except Exception as e:
            self.rollback()
            logger.error(f"Atomic save failed: {e}")
            return False

    def close(self):
        if self._session:
            self._session.close()
        if self._engine:
            self._engine.dispose()


class MongoDBDatabase:
    def __init__(self, connection_string: Optional[str] = None):
        if not MONGODB_AVAILABLE:
            raise ImportError("pymongo not installed")
        self.connection_string = connection_string or os.getenv('MONGODB_URL', '')
        if not self.connection_string:
            try:
                self.connection_string = st.secrets["MONGODB_URL"]
            except (KeyError, FileNotFoundError):
                logger.error("No MongoDB connection string provided.")
                raise ValueError("MongoDB connection string missing. Please add MONGODB_URL to .streamlit/secrets.toml")
        self.client = MongoClient(self.connection_string)
        self.db = self.client['stock_platform']
        self.portfolios = self.db['portfolios']
        self.trades = self.db['trades']
        self.alerts = self.db['alerts']
        self.portfolios.create_index([('user_id', 1), ('ticker', 1)])
        self.trades.create_index([('user_id', 1), ('Date', -1)])
        self.alerts.create_index([('user_id', 1), ('is_active', 1)])
        self._transaction = False
        logger_db.info("MongoDB connection established")

    def begin_transaction(self):
        self._transaction = True
        logger_db.info("Simulated transaction started")

    def commit(self):
        self._transaction = False
        logger_db.info("Simulated transaction committed")

    def rollback(self):
        self._transaction = False
        logger_db.warning("Simulated transaction rolled back")

    def save_portfolio(self, user_id: str, positions: List[Dict]) -> bool:
        try:
            self.portfolios.delete_many({'user_id': user_id})
            for pos in positions:
                doc = {
                    **pos,
                    "ticker": pos.get("Ticker", "").upper(),
                    "user_id": user_id,
                    "created_at": datetime.utcnow()
                }
                doc.pop("Ticker", None)
                self.portfolios.insert_one(doc)
            logger_db.info(f"Saved {len(positions)} positions for user {user_id}")
            return True
        except Exception as e:
            logger_db.error(f"Error saving portfolio: {e}")
            return False

    def load_portfolio(self, user_id: str) -> List[Dict]:
        try:
            docs = list(self.portfolios.find({'user_id': user_id}, {'_id':0, 'user_id':0, 'created_at':0}))
            for doc in docs:
                if 'ticker' in doc:
                    doc['Ticker'] = doc.pop('ticker')
            return docs
        except Exception as e:
            logger_db.error(f"Error loading portfolio: {e}")
            return[]

    def add_position(self, user_id: str, position: Dict) -> bool:
        try:
            doc = {
                **position,
                "ticker": position.get("Ticker", "").upper(),
                "user_id": user_id,
                "created_at": datetime.utcnow()
            }
            doc.pop("Ticker", None)
            self.portfolios.insert_one(doc)
            return True
        except Exception as e:
            logger_db.error(f"Error adding position: {e}")
            return False

    def delete_position(self, user_id: str, ticker: str) -> bool:
        try:
            ticker = ticker.upper()
            self.portfolios.delete_one({'user_id': user_id, 'ticker': ticker})
            return True
        except Exception as e:
            logger_db.error(f"Error deleting position: {e}")
            return False

    def save_trade(self, user_id: str, trade_data: Dict) -> bool:
        try:
            doc = trade_data.copy()
            doc['user_id'] = user_id
            doc['created_at'] = datetime.utcnow()
            self.trades.insert_one(doc)
            return True
        except Exception as e:
            logger_db.error(f"Error saving trade: {e}")
            return False

    def load_trades(self, user_id: str) -> List[Dict]:
        try:
            return list(self.trades.find({'user_id': user_id}, {'_id':0, 'user_id':0, 'created_at':0}).sort('Date', -1))
        except Exception as e:
            logger_db.error(f"Error loading trades: {e}")
            return[]

    def save_portfolio_with_trades(self, user_id: str, portfolio: List[Dict], trades: List[Dict]) -> bool:
        try:
            self.begin_transaction()
            ok_port = self.save_portfolio(user_id, portfolio)
            ok_trades = all(self.save_trade(user_id, t) for t in trades)
            if ok_port and ok_trades:
                self.commit()
                return True
            else:
                self.rollback()
                return False
        except Exception as e:
            self.rollback()
            logger.error(f"Atomic save failed: {e}")
            return False

    def close(self):
        self.client.close()


def get_database(db_type: str = 'postgresql', connection_string: Optional[str] = None):
    if db_type.lower() == 'postgresql':
        return PostgreSQLDatabase(connection_string)
    elif db_type.lower() == 'mongodb':
        return MongoDBDatabase(connection_string)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

# -------------------------------------------------------
# ALERT SYSTEM (Synchronous)
# -------------------------------------------------------
class AlertManager:
    def __init__(self):
        if 'alerts' not in st.session_state:
            st.session_state.alerts =[]
        if 'alert_queue' not in st.session_state:
            st.session_state.alert_queue = []  # plain list avoids Streamlit serialization warnings
        if 'last_alert_check' not in st.session_state:
            st.session_state.last_alert_check = 0

    def check_alerts(self):
        if not st.session_state.alerts:
            return
            
        now = time.time()
        if now - st.session_state.last_alert_check < CONFIG.alert_check_interval:
            return
        st.session_state.last_alert_check = now

        tickers = set(alert['ticker'] for alert in st.session_state.alerts if not alert['triggered'])
        if tickers:
            prices = {}
            for t in tickers:
                try:
                    price = get_current_price(t)
                    if price > 0:
                        prices[t] = price
                except:
                    pass
            triggered = []
            for alert in st.session_state.alerts:
                if alert['triggered'] or alert['ticker'] not in prices:
                    continue
                current_price = prices[alert['ticker']]
                
                if alert['type'] == 'price_above' and current_price > alert['threshold']:
                    should_trigger = True
                elif alert['type'] == 'price_below' and current_price < alert['threshold']:
                    should_trigger = True
                else:
                    should_trigger = False
                    
                if should_trigger:
                    alert['triggered'] = True
                    alert['triggered_at'] = get_ist_time()
                    alert['triggered_price'] = current_price
                    triggered.append(alert)
                    
            for alert in triggered:
                st.session_state.alert_queue.append(alert)
                logger.info(f"Alert triggered: {alert}")

    def add_alert(self, ticker: str, alert_type: str, threshold: float, message: str = "") -> None:
        alert = {
            'id': str(uuid.uuid4()),
            'ticker': ticker,
            'type': alert_type,
            'threshold': threshold,
            'message': message,
            'created': get_ist_time(),
            'triggered': False,
            'triggered_at': None,
            'triggered_price': None
        }
        st.session_state.alerts.append(alert)
        logger.info(f"Alert created: {alert}")

    def display_alerts_ui(self) -> None:
        st.subheader("🔔 Price Alerts")
        pending = st.session_state.alert_queue[:]
        st.session_state.alert_queue.clear()
        for alert in pending:
            st.success(f"🔔 **Alert Triggered!** {alert['ticker']} {alert['type'].replace('_', ' ')} {alert['threshold']:.2f} at {alert['triggered_price']:.2f}")

        with st.expander("➕ Add New Alert"):
            with st.form("alert_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    ticker = st.text_input("Ticker", key="alert_ticker")
                with col2:
                    alert_type = st.selectbox("Condition", ["price_above", "price_below"])
                with col3:
                    threshold = st.number_input("Price", min_value=0.0, step=0.01)
                submitted = st.form_submit_button("Create Alert")
                if submitted and ticker:
                    self.add_alert(ticker, alert_type, threshold)
                    st.success("Alert created! The system will check conditions periodically while you browse.")
        if st.session_state.alerts:
            df_alerts = pd.DataFrame(st.session_state.alerts)
            st.dataframe(df_alerts[['ticker', 'type', 'threshold', 'triggered', 'triggered_price', 'triggered_at']])
        else:
            st.info("No alerts set")

# -------------------------------------------------------
# RATIO CALCULATIONS
# -------------------------------------------------------
def compute_ratios(group: pd.DataFrame, price: float, period: str = 'annual') -> pd.DataFrame:
    day_mult = {'quarterly': 90, 'half-yearly': 180, 'annual': 365}.get(period, 365)

    if price > 0 and '_share_count' in group.columns:
        shares = group['_share_count'].replace(0, np.nan)
        if 'Equity' in group.columns:
            bvps = safe_divide(group['Equity'], shares)
            group['P/B'] = safe_divide(pd.Series([price] * len(group)), bvps)
        if 'Revenue' in group.columns:
            sps = safe_divide(group['Revenue'], shares)
            group['P/S'] = safe_divide(pd.Series([price] * len(group)), sps)
        if 'EPS' in group.columns:
            group['P/E'] = safe_divide(pd.Series([price] * len(group)), group['EPS'])

    if 'Net Profit' in group.columns and 'Revenue' in group.columns:
        group['Net Profit Margin (%)'] = safe_divide(group['Net Profit'], group['Revenue'], 0) * 100
    if 'Net Profit' in group.columns and 'Equity' in group.columns:
        group['ROE (%)'] = safe_divide(group['Net Profit'], group['Equity'], 0) * 100
    if 'Net Profit' in group.columns and 'Total Assets' in group.columns:
        group['ROA (%)'] = safe_divide(group['Net Profit'], group['Total Assets'], 0) * 100

    if 'Total Debt' in group.columns and 'Equity' in group.columns:
        group['Debt-to-Equity'] = safe_divide(group['Total Debt'], group['Equity'])
    if 'Current Assets' in group.columns and 'Current Liabilities' in group.columns:
        group['Current Ratio'] = safe_divide(group['Current Assets'], group['Current Liabilities'])
    if 'Interest' in group.columns and 'EBIT' in group.columns:
        group['Interest Coverage'] = safe_divide(group['EBIT'], group['Interest'])

    if 'Operating Cash Flow' in group.columns and 'Capex' in group.columns:
        group['Free Cash Flow'] = group['Operating Cash Flow'] - group['Capex']
    if 'Operating Cash Flow' in group.columns and 'Net Profit' in group.columns:
        group['OCF/Net Profit'] = safe_divide(group['Operating Cash Flow'], group['Net Profit'])

    if 'Inventory' in group.columns and 'COGS' in group.columns:
        inv_avg = (group['Inventory'] + group['Inventory'].shift(1)) / 2
        group['Inventory Days'] = safe_divide(inv_avg, group['COGS'], 0) * day_mult
    if 'Receivables' in group.columns and 'Revenue' in group.columns:
        rec_avg = (group['Receivables'] + group['Receivables'].shift(1)) / 2
        group['Receivable Days (DSO)'] = safe_divide(rec_avg, group['Revenue'], 0) * day_mult

    if all(col in group.columns for col in ['EBIT', 'Total Assets', 'Current Liabilities']):
        capital_employed = group['Total Assets'] - group['Current Liabilities']
        group['ROCE (%)'] = safe_divide(group['EBIT'], capital_employed, 0) * 100

    return group

def generate_analysis_from_pdfs(files: List[str], stock_prices: Dict[str, float],
                                max_pages: int = 120, progress_callback=None) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
    logger.info(f"Starting PDF analysis: {len(files)} files, max_pages={max_pages}")
    raw_records =[]
    total_files = len(files)
    for idx, f in enumerate(files):
        if progress_callback:
            progress_callback(idx, total_files, f"Processing {os.path.basename(f)}")
        try:
            data = extract_financial_data(f, max_pages=max_pages)
            if data.get('_extraction_success'):
                raw_records.append(data)
        except Exception as e:
            logger.error(f"Failed to process {f}: {e}")
    if not raw_records:
        logger.warning("No financial data extracted from any PDF")
        return pd.DataFrame(), {}, []
    df = pd.DataFrame(raw_records)
    numeric_cols =['Revenue', 'Net Profit', 'Equity', 'Total Assets', 'EPS',
                    'Current Assets', 'Current Liabilities', 'Total Debt',
                    'Operating Cash Flow', 'Capex', '_share_count']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    results =[]
    latest_metrics = {}
    for company, group in df.groupby('Company'):
        period = 'annual'
        if '_period' in group.columns:
            periods = group['_period'].value_counts()
            period = periods.index[0] if not periods.empty else 'annual'
        group = group.sort_values('Year').reset_index(drop=True)
        price = stock_prices.get(company, 0.0)
        group = compute_ratios(group, price, period)
        if not group.empty:
            latest = group.iloc[-1]
            latest_metrics[company] = {
                'eps': latest.get('EPS'),
                'free_cash_flow': latest.get('Free Cash Flow'),
                'share_count': latest.get('_share_count'),
                'net_profit': latest.get('Net Profit'),
                'revenue': latest.get('Revenue'),
                'equity': latest.get('Equity'),
                'period': period,
            }
        results.append(group)
    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.round(2)
    ordered = ['Company', 'Year', 'Revenue']
    ratio_cols =['P/E', 'P/B', 'P/S', 'Net Profit Margin (%)', 'ROE (%)', 'ROA (%)',
                  'ROCE (%)', 'Debt-to-Equity', 'Current Ratio', 'Free Cash Flow']
    for col in ratio_cols:
        if col in final_df.columns and col not in ordered:
            ordered.append(col)
    ordered +=[c for c in final_df.columns if c not in ordered and not c.startswith('_')]
    final_df = final_df[[c for c in ordered if c in final_df.columns]]
    if final_df['Company'].nunique() == 1:
        logger.info(f"PDF analysis complete: {len(raw_records)} records extracted")
        return final_df.set_index('Year').drop(columns=['Company'], errors='ignore'), latest_metrics, raw_records
    logger.info(f"PDF analysis complete: {len(raw_records)} records extracted")
    return final_df.set_index(['Company', 'Year']), latest_metrics, raw_records

# -------------------------------------------------------
# TRADING & RISK FUNCTIONS
# -------------------------------------------------------
def calculate_position_size(portfolio_value: float, risk_per_trade: float,
                            entry_price: float, stop_loss: float, trade_type: str = "Long") -> Optional[Dict]:
    risk_amount = portfolio_value * (risk_per_trade / 100)
    
    if trade_type == "Long":
        price_risk = entry_price - stop_loss
    else:
        price_risk = stop_loss - entry_price
        
    if price_risk <= 0:
        return None
        
    shares = int(risk_amount / price_risk)
    if shares == 0:
        return None
        
    position_value = shares * entry_price
    return {
        'shares': shares,
        'position_value': position_value,
        'risk_amount': risk_amount,
        'position_pct': (position_value / portfolio_value) * 100
    }

def calculate_tax(buy_price: float, sell_price: float, shares: int, holding_days: int) -> Dict:
    profit = (sell_price - buy_price) * shares
    if profit <= 0:
        return {
            'gross_profit': profit,
            'stt': 0,
            'capital_gains_tax': 0,
            'total_tax': 0,
            'net_profit': profit,
            'tax_type': 'Loss',
            'holding_days': holding_days
        }
    stt = (buy_price * shares + sell_price * shares) * (CONFIG.stt_equity / 100)
    if holding_days >= 365:
        taxable = max(0, profit - CONFIG.ltcg_exemption)
        tax = taxable * (CONFIG.ltcg_equity_rate / 100)
        tax_type = 'LTCG'
    else:
        tax = profit * (CONFIG.stcg_equity_rate / 100)
        tax_type = 'STCG'
    total_tax = stt + tax
    net_profit = profit - total_tax
    return {
        'gross_profit': profit,
        'stt': stt,
        'capital_gains_tax': tax,
        'total_tax': total_tax,
        'net_profit': net_profit,
        'tax_type': tax_type,
        'holding_days': holding_days
    }

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 7.0) -> float:
    excess = returns - (risk_free_rate / 252)
    if excess.std() == 0:
        return 0
    return (excess.mean() / excess.std()) * np.sqrt(252)

def calculate_max_drawdown(prices: pd.Series) -> float:
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return drawdown.min() * 100

def graham_intrinsic_value(eps: float, growth: float, yield_rate: float = 7.5) -> Optional[float]:
    if eps <= 0:
        return None
    return eps * (8.5 + 2 * growth) * (CONFIG.graham_base_yield / yield_rate)

def dcf_intrinsic_value(fcf: float, growth: float, discount: float, terminal: float, shares: float) -> Optional[float]:
    if fcf <= 0 or shares <= 0:
        return None
    g = growth / 100
    r = discount / 100
    g_t = terminal / 100
    if r <= g_t:
        return None
    projected = [fcf * (1 + g) ** yr for yr in range(1, 11)]
    pv_projected = sum(f / (1 + r) ** yr for yr, f in enumerate(projected, 1))
    terminal_fcf = projected[-1] * (1 + g_t)
    pv_terminal = (terminal_fcf / (r - g_t)) / (1 + r) ** 10
    return (pv_projected + pv_terminal) / shares

def calculate_transaction_cost(buy_price: float, sell_price: float, shares: int) -> float:
    turnover = (buy_price + sell_price) * shares
    stt = turnover * (CONFIG.stt_equity / 100)
    brokerage = turnover * (CONFIG.brokerage_pct / 100)
    return stt + brokerage

def calculate_ltcg_grandfathering(buy_price: float, sell_price: float, shares: int,
                                   buy_date: datetime, sell_date: datetime) -> float:
    cutoff = datetime(2018, 1, 31, tzinfo=IST)
    if buy_date >= cutoff:
        profit = (sell_price - buy_price) * shares
        return max(0, profit - CONFIG.ltcg_exemption)
    else:
        fmv = buy_price  # Placeholder; should be fetched
        cost = max(buy_price, fmv)
        profit = (sell_price - cost) * shares
        return max(0, profit - CONFIG.ltcg_exemption)

# -------------------------------------------------------
# DATA EXPORT FUNCTIONS
# -------------------------------------------------------
def export_portfolio_to_excel(portfolio: List[Dict]) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_portfolio = pd.DataFrame(portfolio)
        df_portfolio.to_excel(writer, sheet_name='Portfolio', index=False)

        if portfolio:
            df_portfolio['P&L %'] = pd.to_numeric(df_portfolio['P&L %'], errors='coerce')
            metrics = {
                'Total Investment': df_portfolio['Total Cost'].sum(),
                'Current Value': df_portfolio['Current Value'].sum(),
                'Total P&L': df_portfolio['Current Value'].sum() - df_portfolio['Total Cost'].sum(),
                'Win Rate': (df_portfolio['P&L %'] > 0).mean() * 100 if len(df_portfolio) > 0 else 0,
                'Best Performer': df_portfolio.nlargest(1, 'P&L %')['Stock'].iloc[0] if not df_portfolio.empty else '',
                'Worst Performer': df_portfolio.nsmallest(1, 'P&L %')['Stock'].iloc[0] if not df_portfolio.empty else '',
            }
            pd.DataFrame([metrics]).to_excel(writer, sheet_name='Metrics', index=False)

        if portfolio and 'Sector' in df_portfolio.columns:
            sector_allocation = df_portfolio.groupby('Sector')['Current Value'].sum().reset_index()
            sector_allocation.to_excel(writer, sheet_name='Sector Allocation', index=False)
    output.seek(0)
    return output

def export_trades_to_csv(trades: List[Dict]) -> str:
    df = pd.DataFrame(trades)
    return df.to_csv(index=False)

def export_tax_report(trades: List[Dict], fy_start_year: int) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df = pd.DataFrame(trades)
        df['Date'] = pd.to_datetime(df['Date'])
        start_date = pd.Timestamp(f"{fy_start_year}-04-01")
        end_date = pd.Timestamp(f"{fy_start_year+1}-03-31")
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        intraday = df[df['Type'] == 'Intraday'].copy()
        if not intraday.empty:
            intraday['Holding Days'] = 0
            intraday['Tax Type'] = 'Intraday'
            intraday.to_excel(writer, sheet_name='Intraday', index=False)

        delivery = df[df['Type'] != 'Intraday'].copy()
        if not delivery.empty:
            if 'Holding Days' not in delivery.columns:
                delivery['Holding Days'] = 0
            delivery['Tax Type'] = delivery['Holding Days'].apply(lambda x: 'STCG' if x < 365 else 'LTCG')
            stcg = delivery[delivery['Tax Type'] == 'STCG']
            ltcg = delivery[delivery['Tax Type'] == 'LTCG']
            if not stcg.empty:
                stcg.to_excel(writer, sheet_name='STCG', index=False)
            if not ltcg.empty:
                ltcg.to_excel(writer, sheet_name='LTCG', index=False)

        summary = {
            'Financial Year':[f"{fy_start_year}-{fy_start_year+1}"],
            'Total Intraday Trades': [len(intraday)],
            'Total STCG Trades':[len(stcg) if 'stcg' in locals() else 0],
            'Total LTCG Trades':[len(ltcg) if 'ltcg' in locals() else 0],
            'Net Taxable LTCG (after exemption)': [0],
            'Tax Payable': [0]
        }
        pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)
    output.seek(0)
    return output

# -------------------------------------------------------
# STREAMLIT GUI
# -------------------------------------------------------
INDIAN_SECTORS =[
    'Banking', 'IT', 'Pharmaceuticals', 'FMCG', 'Auto', 'Metals',
    'Oil & Gas', 'Infrastructure', 'Cement', 'Power', 'Telecom',
    'Real Estate', 'Media', 'Capital Goods', 'Consumer Durables'
]

# ── Tier 1: Nifty 50 ─────────────────────────────────────────────────────────
NIFTY_50: Dict[str, Tuple[str, str]] = {
    "RELIANCE.NS":   ("Reliance Industries",           "Oil & Gas"),
    "TCS.NS":        ("Tata Consultancy Services",      "IT"),
    "HDFCBANK.NS":   ("HDFC Bank",                     "Banking"),
    "INFY.NS":       ("Infosys",                       "IT"),
    "ICICIBANK.NS":  ("ICICI Bank",                    "Banking"),
    "HINDUNILVR.NS": ("Hindustan Unilever",             "FMCG"),
    "ITC.NS":        ("ITC",                           "FMCG"),
    "SBIN.NS":       ("State Bank of India",            "Banking"),
    "BHARTIARTL.NS": ("Bharti Airtel",                 "Telecom"),
    "KOTAKBANK.NS":  ("Kotak Mahindra Bank",            "Banking"),
    "LT.NS":         ("Larsen & Toubro",                "Infrastructure"),
    "AXISBANK.NS":   ("Axis Bank",                     "Banking"),
    "ASIANPAINT.NS": ("Asian Paints",                  "Consumer Durables"),
    "MARUTI.NS":     ("Maruti Suzuki",                 "Auto"),
    "SUNPHARMA.NS":  ("Sun Pharma",                    "Pharma"),
    "TITAN.NS":      ("Titan Company",                 "Consumer Durables"),
    "ULTRACEMCO.NS": ("UltraTech Cement",              "Cement"),
    "BAJFINANCE.NS": ("Bajaj Finance",                 "NBFC"),
    "WIPRO.NS":      ("Wipro",                         "IT"),
    "HCLTECH.NS":    ("HCL Technologies",              "IT"),
    "ONGC.NS":       ("ONGC",                          "Oil & Gas"),
    "NTPC.NS":       ("NTPC",                          "Power"),
    "POWERGRID.NS":  ("Power Grid Corporation",         "Power"),
    "M&M.NS":        ("Mahindra & Mahindra",           "Auto"),
    "TATAMOTORS.NS": ("Tata Motors",                   "Auto"),
    "TATASTEEL.NS":  ("Tata Steel",                    "Metals"),
    "JSWSTEEL.NS":   ("JSW Steel",                     "Metals"),
    "BRITANNIA.NS":  ("Britannia Industries",           "FMCG"),
    "NESTLEIND.NS":  ("Nestle India",                  "FMCG"),
    "TATACONSUM.NS": ("Tata Consumer Products",         "FMCG"),
    "BAJAJFINSV.NS": ("Bajaj Finserv",                 "NBFC"),
    "ADANIPORTS.NS": ("Adani Ports",                   "Infrastructure"),
    "GRASIM.NS":     ("Grasim Industries",             "Cement"),
    "DIVISLAB.NS":   ("Divi's Laboratories",           "Pharma"),
    "DRREDDY.NS":    ("Dr. Reddy's Laboratories",      "Pharma"),
    "CIPLA.NS":      ("Cipla",                         "Pharma"),
    "APOLLOHOSP.NS": ("Apollo Hospitals",              "Healthcare"),
    "HEROMOTOCO.NS": ("Hero MotoCorp",                 "Auto"),
    "BAJAJ-AUTO.NS": ("Bajaj Auto",                    "Auto"),
    "EICHERMOT.NS":  ("Eicher Motors",                 "Auto"),
    "COALINDIA.NS":  ("Coal India",                    "Metals"),
    "IOC.NS":        ("Indian Oil Corporation",         "Oil & Gas"),
    "BPCL.NS":       ("Bharat Petroleum",              "Oil & Gas"),
    "HINDALCO.NS":   ("Hindalco Industries",           "Metals"),
    "SHREECEM.NS":   ("Shree Cement",                  "Cement"),
    "TECHM.NS":      ("Tech Mahindra",                 "IT"),
    "INDUSINDBK.NS": ("IndusInd Bank",                 "Banking"),
    "UPL.NS":        ("UPL",                           "Chemicals"),
    "ADANIENT.NS":   ("Adani Enterprises",             "Diversified"),
    "SBILIFE.NS":    ("SBI Life Insurance",            "Insurance"),
}

# ── Tier 2: Nifty Next 50 ─────────────────────────────────────────────────────
NIFTY_NEXT50: Dict[str, Tuple[str, str]] = {
    "DMART.NS":       ("Avenue Supermarts",             "Retail"),
    "PIDILITIND.NS":  ("Pidilite Industries",           "Chemicals"),
    "SIEMENS.NS":     ("Siemens",                      "Capital Goods"),
    "HAVELLS.NS":     ("Havells India",                 "Consumer Durables"),
    "DABUR.NS":       ("Dabur India",                   "FMCG"),
    "MARICO.NS":      ("Marico",                        "FMCG"),
    "GODREJCP.NS":    ("Godrej Consumer Products",      "FMCG"),
    "BERGEPAINT.NS":  ("Berger Paints",                 "Consumer Durables"),
    "MUTHOOTFIN.NS":  ("Muthoot Finance",               "NBFC"),
    "CHOLAFIN.NS":    ("Cholamandalam Investment",      "NBFC"),
    "TORNTPHARM.NS":  ("Torrent Pharma",                "Pharma"),
    "LUPIN.NS":       ("Lupin",                         "Pharma"),
    "BIOCON.NS":      ("Biocon",                        "Pharma"),
    "AUROPHARMA.NS":  ("Aurobindo Pharma",              "Pharma"),
    "BOSCHLTD.NS":    ("Bosch",                         "Auto Ancillary"),
    "NAUKRI.NS":      ("Info Edge (Naukri)",             "IT"),
    "ZOMATO.NS":      ("Zomato",                        "Consumer Internet"),
    "PAYTM.NS":       ("Paytm (One97 Comm)",             "Fintech"),
    "NYKAA.NS":       ("FSN E-Commerce (Nykaa)",         "Retail"),
    "POLICYBZR.NS":   ("PB Fintech (PolicyBazaar)",      "Fintech"),
    "DELHIVERY.NS":   ("Delhivery",                     "Logistics"),
    "TATAPOWER.NS":   ("Tata Power",                    "Power"),
    "ADANIGREEN.NS":  ("Adani Green Energy",             "Power"),
    "ADANIPOWER.NS":  ("Adani Power",                   "Power"),
    "RECLTD.NS":      ("REC Limited",                   "NBFC"),
    "PFC.NS":         ("Power Finance Corp",             "NBFC"),
    "IRCTC.NS":       ("IRCTC",                         "Travel"),
    "IRFC.NS":        ("IRFC",                          "NBFC"),
    "HAL.NS":         ("Hindustan Aeronautics",          "Defence"),
    "BEL.NS":         ("Bharat Electronics",             "Defence"),
    "BHEL.NS":        ("Bharat Heavy Electricals",       "Capital Goods"),
    "CUMMINSIND.NS":  ("Cummins India",                  "Capital Goods"),
    "ABB.NS":         ("ABB India",                      "Capital Goods"),
    "PERSISTENT.NS":  ("Persistent Systems",             "IT"),
    "MPHASIS.NS":     ("Mphasis",                        "IT"),
    "COFORGE.NS":     ("Coforge",                        "IT"),
    "LTIM.NS":        ("LTIMindtree",                    "IT"),
    "OFSS.NS":        ("Oracle Financial Services",       "IT"),
    "KPITTECH.NS":    ("KPIT Technologies",              "IT"),
    "TATAELXSI.NS":   ("Tata Elxsi",                     "IT"),
    "BALKRISIND.NS":  ("Balkrishna Industries",          "Auto Ancillary"),
    "MOTHERSON.NS":   ("Samvardhana Motherson",          "Auto Ancillary"),
    "MINDA.NS":       ("UNO Minda",                      "Auto Ancillary"),
    "SUNDARMFIN.NS":  ("Sundaram Finance",               "NBFC"),
    "MAXHEALTH.NS":   ("Max Healthcare",                 "Healthcare"),
    "FORTIS.NS":      ("Fortis Healthcare",              "Healthcare"),
    "LALPATHLAB.NS":  ("Dr. Lal PathLabs",               "Healthcare"),
    "METROPOLIS.NS":  ("Metropolis Healthcare",          "Healthcare"),
    "INDUSTOWER.NS":  ("Indus Towers",                   "Telecom"),
    "VODAFONE.NS":    ("Vodafone Idea",                  "Telecom"),
}

# ── Tier 3: Nifty Midcap 100 (key liquid stocks) ─────────────────────────────
NIFTY_MIDCAP: Dict[str, Tuple[str, str]] = {
    "PAGEIND.NS":     ("Page Industries",               "Consumer Durables"),
    "VOLTAS.NS":      ("Voltas",                        "Consumer Durables"),
    "WHIRLPOOL.NS":   ("Whirlpool India",               "Consumer Durables"),
    "BLUESTARCO.NS":  ("Blue Star",                     "Consumer Durables"),
    "CROMPTON.NS":    ("Crompton Greaves Consumer",      "Consumer Durables"),
    "BATAINDIA.NS":   ("Bata India",                    "Consumer Durables"),
    "RELAXO.NS":      ("Relaxo Footwears",              "Consumer Durables"),
    "KAJARIACER.NS":  ("Kajaria Ceramics",              "Cement"),
    "RAMCOCEM.NS":    ("Ramco Cements",                 "Cement"),
    "JKCEMENT.NS":    ("JK Cement",                     "Cement"),
    "AIAENG.NS":      ("AIA Engineering",               "Capital Goods"),
    "GRINDWELL.NS":   ("Grindwell Norton",              "Capital Goods"),
    "APLAPOLLO.NS":   ("APL Apollo Tubes",              "Metals"),
    "RATNAMANI.NS":   ("Ratnamani Metals",              "Metals"),
    "NATIONALUM.NS":  ("National Aluminium",            "Metals"),
    "SAILS.NS":       ("SAIL",                          "Metals"),
    "GMRINFRA.NS":    ("GMR Airports Infra",            "Infrastructure"),
    "IRB.NS":         ("IRB Infrastructure",            "Infrastructure"),
    "KNRCON.NS":      ("KNR Constructions",             "Infrastructure"),
    "ASHOKA.NS":      ("Ashoka Buildcon",               "Infrastructure"),
    "DELTACORP.NS":   ("Delta Corp",                    "Real Estate"),
    "OBEROIRLTY.NS":  ("Oberoi Realty",                 "Real Estate"),
    "GODREJPROP.NS":  ("Godrej Properties",             "Real Estate"),
    "PRESTIGE.NS":    ("Prestige Estates",              "Real Estate"),
    "BRIGADE.NS":     ("Brigade Enterprises",           "Real Estate"),
    "JSWENERGY.NS":   ("JSW Energy",                    "Power"),
    "CESC.NS":        ("CESC",                          "Power"),
    "TORNTPOWER.NS":  ("Torrent Power",                 "Power"),
    "SUZLON.NS":      ("Suzlon Energy",                 "Power"),
    "INOXWIND.NS":    ("Inox Wind",                     "Power"),
    "SUNDRCFAST.NS":  ("Sundram Fasteners",             "Auto Ancillary"),
    "SUPRAJIT.NS":    ("Suprajit Engineering",          "Auto Ancillary"),
    "GABRIEL.NS":     ("Gabriel India",                 "Auto Ancillary"),
    "AAPL.NS":        ("Amara Raja Energy",             "Auto Ancillary"),
    "DEEPAKNITR.NS":  ("Deepak Nitrite",                "Chemicals"),
    "AARTI.NS":       ("Aarti Industries",              "Chemicals"),
    "SRF.NS":         ("SRF",                           "Chemicals"),
    "NAVINFLUOR.NS":  ("Navin Fluorine",                "Chemicals"),
    "VINATIORGA.NS":  ("Vinati Organics",               "Chemicals"),
    "FINPIPE.NS":     ("Finolex Industries",            "Chemicals"),
    "SYNGENE.NS":     ("Syngene International",         "Pharma"),
    "IPCA.NS":        ("IPCA Laboratories",             "Pharma"),
    "ALKEM.NS":       ("Alkem Laboratories",            "Pharma"),
    "ABBOTINDIA.NS":  ("Abbott India",                  "Pharma"),
    "PFIZER.NS":      ("Pfizer India",                  "Pharma"),
    "BANKBARODA.NS":  ("Bank of Baroda",                "Banking"),
    "PNB.NS":         ("Punjab National Bank",          "Banking"),
    "CANBK.NS":       ("Canara Bank",                   "Banking"),
    "UNIONBANK.NS":   ("Union Bank of India",           "Banking"),
    "FEDERALBNK.NS":  ("Federal Bank",                  "Banking"),
    "IDFCFIRSTB.NS":  ("IDFC First Bank",               "Banking"),
    "BANDHANBNK.NS":  ("Bandhan Bank",                  "Banking"),
    "RBLBANK.NS":     ("RBL Bank",                      "Banking"),
    "KARURVYSYA.NS":  ("Karur Vysya Bank",              "Banking"),
    "BAJAJHFL.NS":    ("Bajaj Housing Finance",         "NBFC"),
    "MANAPPURAM.NS":  ("Manappuram Finance",            "NBFC"),
    "M&MFIN.NS":      ("M&M Financial Services",        "NBFC"),
    "SHRIRAMFIN.NS":  ("Shriram Finance",               "NBFC"),
    "POONAWALLA.NS":  ("Poonawalla Fincorp",            "NBFC"),
}

# ── Combined universe (used by the engine) ────────────────────────────────────
FULL_UNIVERSE: Dict[str, Tuple[str, str]] = {**NIFTY_50, **NIFTY_NEXT50, **NIFTY_MIDCAP}

UNIVERSE_TIERS = {
    "Nifty 50": NIFTY_50,
    "Nifty Next 50": NIFTY_NEXT50,
    "Nifty Midcap 100": NIFTY_MIDCAP,
    "All (~160 stocks)": FULL_UNIVERSE,
}

# ── Indian + Global indices ───────────────────────────────────────────────────
INDIAN_INDICES = {
    "Nifty 50":    "^NSEI",
    "Bank Nifty":  "^NSEBANK",
    "Sensex":      "^BSESN",
    "Nifty IT":    "^CNXIT",
    "Nifty Pharma":"^CNXPHARMA",
}

GLOBAL_INDICES = {
    "S&P 500":    "^GSPC",
    "Nasdaq":     "^IXIC",
    "Dow Jones":  "^DJI",
    "FTSE 100":   "^FTSE",
    "Nikkei 225": "^N225",
    "Hang Seng":  "^HSI",
    "SGX Nifty":  "^NSEI",  # best proxy via yfinance
}

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ──────────────────────────────────────────────────────────────────────────────



def is_pre_open() -> bool:
    now = get_ist_time()
    if now.weekday() >= 5:
        return False
    return now.hour == 9 and now.minute < 15

_api_lock = threading.Lock()
_last_call: Dict[str, float] = {}

def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns returned by yf.download() in yfinance >= 0.2.x.
    Single-ticker downloads may return columns like ('Close', 'RELIANCE.NS').
    This reduces them to plain strings like 'Close'.
    """
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def rate_limited_fetch(ticker: str, period: str = "3mo",
                       interval: str = "1d",
                       min_interval: float = 0.4) -> Optional[pd.DataFrame]:
    """Thread-safe, rate-limited yfinance fetch with retry."""
    with _api_lock:
        now = time.time()
        key = f"{ticker}:{period}:{interval}"
        wait = min_interval - (now - _last_call.get(key, 0))
        if wait > 0:
            time.sleep(wait)
        _last_call[key] = time.time()

    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            df = flatten_yf_columns(df)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                return df
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {ticker}: {e}")
            time.sleep(2 ** attempt)
    return None

def parallel_fetch(tickers: List[str],
                   period: str = "3mo",
                   interval: str = "1d",
                   max_workers: int = 4) -> Dict[str, pd.DataFrame]:
    """Fetch multiple tickers in parallel. Returns {ticker: df}."""
    results: Dict[str, pd.DataFrame] = {}
    def _fetch(t):
        df = rate_limited_fetch(t, period, interval)
        if df is not None:
            results[t] = df
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch, t): t for t in tickers}
        concurrent.futures.wait(futures, timeout=120)
    return results

# ──────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATOR ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all indicators needed by the selection engine.
    Input: raw OHLCV DataFrame from yfinance.
    Output: same df enriched with indicator columns.
    """
    if df is None or len(df) < 20:
        return df

    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    v = df["Volume"].squeeze()

    # ── Moving Averages ───────────────────────────────────────────────────────
    df["SMA_10"]  = c.rolling(10).mean()
    df["SMA_20"]  = c.rolling(20).mean()
    df["SMA_50"]  = c.rolling(50).mean()
    df["SMA_200"] = c.rolling(200).mean()
    df["EMA_9"]   = c.ewm(span=9,  adjust=False).mean()
    df["EMA_21"]  = c.ewm(span=21, adjust=False).mean()

    # ── MACD ─────────────────────────────────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ── RSI (Wilder's smoothing) ──────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid  = c.rolling(20).mean()
    bb_std  = c.rolling(20).std()
    df["BB_Upper"]  = bb_mid + 2 * bb_std
    df["BB_Lower"]  = bb_mid - 2 * bb_std
    df["BB_Middle"] = bb_mid
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / bb_mid  # volatility proxy

    # ── ATR (14) ──────────────────────────────────────────────────────────────
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # ── Volume indicators ─────────────────────────────────────────────────────
    df["Vol_SMA20"]  = v.rolling(20).mean()
    df["Vol_Ratio"]  = v / df["Vol_SMA20"]      # >1.5 = volume surge

    # ── VWAP (rolling daily proxy — proper VWAP needs intraday data) ──────────
    typical = (h + l + c) / 3
    df["VWAP_Proxy"] = (typical * v).rolling(20).sum() / v.rolling(20).sum()

    # ── Momentum / Rate of Change ─────────────────────────────────────────────
    df["ROC_5"]  = c.pct_change(5)  * 100
    df["ROC_10"] = c.pct_change(10) * 100
    df["ROC_20"] = c.pct_change(20) * 100

    # ── Squeeze / Consolidation flag ─────────────────────────────────────────
    # Keltner Channels
    kc_upper = df["EMA_21"] + 1.5 * df["ATR"]
    kc_lower = df["EMA_21"] - 1.5 * df["ATR"]
    df["In_Squeeze"] = (df["BB_Upper"] < kc_upper) & (df["BB_Lower"] > kc_lower)

    # ── 52-week high/low proximity ────────────────────────────────────────────
    rolling_high = c.rolling(252).max()
    rolling_low  = c.rolling(252).min()
    df["Pct_From_52W_High"] = ((c - rolling_high) / rolling_high) * 100
    df["Pct_From_52W_Low"]  = ((c - rolling_low)  / rolling_low)  * 100

    return df


def detect_support_resistance(df: pd.DataFrame, window: int = 10,
                               lookback: int = 60) -> Tuple[List[float], List[float]]:
    """
    Returns (support_levels, resistance_levels) as price lists.
    Uses local minima/maxima over the last `lookback` candles.
    """
    if df is None or len(df) < window * 3:
        return [], []
    sub = df.tail(lookback)
    c   = sub["Close"].values

    supports    = []
    resistances = []
    for i in range(window, len(c) - window):
        segment = c[i - window: i + window + 1]
        if c[i] == segment.min():
            supports.append(round(float(c[i]), 2))
        if c[i] == segment.max():
            resistances.append(round(float(c[i]), 2))

    # Deduplicate levels within 0.5% of each other
    def dedupe(levels):
        levels = sorted(set(levels))
        merged = []
        for lvl in levels:
            if not merged or abs(lvl - merged[-1]) / merged[-1] > 0.005:
                merged.append(lvl)
        return merged

    return dedupe(supports[-5:]), dedupe(resistances[-5:])


def get_breakout_price(df: pd.DataFrame, lookback: int = 20) -> float:
    """Returns the recent resistance (breakout level) price."""
    if df is None or df.empty:
        return 0.0
    return round(float(df["High"].tail(lookback).max()), 2)


def get_breakdown_price(df: pd.DataFrame, lookback: int = 20) -> float:
    """Returns the recent support (breakdown level) price."""
    if df is None or df.empty:
        return 0.0
    return round(float(df["Low"].tail(lookback).min()), 2)

# ──────────────────────────────────────────────────────────────────────────────
# MOMENTUM SCORING ENGINE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StockScore:
    ticker:          str
    name:            str
    sector:          str
    tier:            str
    price:           float = 0.0
    momentum_score:  int   = 0
    trend_score:     int   = 0
    volume_score:    int   = 0
    total_score:     int   = 0
    rsi:             float = 0.0
    macd_signal:     str   = "Neutral"
    ema_trend:       str   = "Neutral"
    vol_ratio:       float = 0.0
    roc_5:           float = 0.0
    roc_20:          float = 0.0
    near_52w_high:   bool  = False
    in_squeeze:      bool  = False
    breakout_price:  float = 0.0
    support_price:   float = 0.0
    atr:             float = 0.0
    bias:            str   = "Watch"       # Long / Short / Watch
    reasons:         List[str] = field(default_factory=list)
    above_sma20:     bool  = False
    above_sma50:     bool  = False
    above_sma200:    bool  = False
    pct_from_52w_high: float = 0.0


def score_stock(ticker: str, name: str, sector: str, tier: str,
                df: pd.DataFrame) -> Optional[StockScore]:
    """
    Score a single stock across momentum, trend, and volume dimensions.
    Returns a StockScore dataclass.
    """
    if df is None or len(df) < 30:
        return None
    try:
        df = compute_indicators(df.copy())
        if df is None or df.empty:
            return None

        last      = df.iloc[-1]
        c         = df["Close"].squeeze()
        score     = StockScore(ticker=ticker, name=name, sector=sector, tier=tier)
        score.price = round(float(last["Close"]), 2)

        reasons   = []
        mom_sc = trend_sc = vol_sc = 0

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi = float(last.get("RSI", 50)) if not pd.isna(last.get("RSI", np.nan)) else 50.0
        score.rsi = round(rsi, 1)
        if 55 <= rsi <= 75:
            mom_sc += 2; reasons.append(f"RSI {rsi:.0f} — bullish momentum zone ✅")
        elif rsi > 75:
            mom_sc += 1; reasons.append(f"RSI {rsi:.0f} — overbought (caution) ⚠️")
        elif rsi < 30:
            mom_sc -= 1; reasons.append(f"RSI {rsi:.0f} — oversold / weak ❌")

        # ── MACD ─────────────────────────────────────────────────────────────
        macd_val  = float(last.get("MACD",        0)) if not pd.isna(last.get("MACD",       np.nan)) else 0.0
        macd_sig  = float(last.get("MACD_Signal", 0)) if not pd.isna(last.get("MACD_Signal",np.nan)) else 0.0
        macd_hist = float(last.get("MACD_Hist",   0)) if not pd.isna(last.get("MACD_Hist",  np.nan)) else 0.0
        if macd_val > macd_sig:
            trend_sc += 2; score.macd_signal = "Bullish"; reasons.append("MACD above signal — bullish ✅")
        else:
            trend_sc -= 1; score.macd_signal = "Bearish"

        # Histogram expanding (momentum building)
        if len(df) >= 3:
            h2 = float(df["MACD_Hist"].iloc[-2]) if not pd.isna(df["MACD_Hist"].iloc[-2]) else 0.0
            if macd_hist > 0 and macd_hist > h2:
                mom_sc += 1; reasons.append("MACD histogram expanding ✅")

        # ── EMA Trend ─────────────────────────────────────────────────────────
        price = score.price
        ema9  = float(last.get("EMA_9",  price)) if not pd.isna(last.get("EMA_9",  np.nan)) else price
        ema21 = float(last.get("EMA_21", price)) if not pd.isna(last.get("EMA_21", np.nan)) else price
        sma20 = float(last.get("SMA_20", price)) if not pd.isna(last.get("SMA_20", np.nan)) else price
        sma50 = float(last.get("SMA_50", price)) if not pd.isna(last.get("SMA_50", np.nan)) else price
        sma200= float(last.get("SMA_200",price)) if not pd.isna(last.get("SMA_200",np.nan)) else price

        score.above_sma20  = price > sma20
        score.above_sma50  = price > sma50
        score.above_sma200 = price > sma200

        if price > ema9 > ema21:
            trend_sc += 2; score.ema_trend = "Strong Up"; reasons.append("EMA 9 > EMA 21 — uptrend ✅")
        elif price > ema21:
            trend_sc += 1; score.ema_trend = "Up"
        else:
            trend_sc -= 1; score.ema_trend = "Down"

        if price > sma20:  trend_sc += 1; reasons.append("Price > SMA20 ✅")
        if price > sma50:  trend_sc += 1; reasons.append("Price > SMA50 ✅")
        if price > sma200: trend_sc += 1; reasons.append("Price > SMA200 (bull mkt) ✅")

        # Golden cross detection (SMA50 crosses above SMA200 in last 5 candles)
        if len(df) >= 6:
            for i in range(-5, 0):
                s50_prev = float(df["SMA_50"].iloc[i-1]) if not pd.isna(df["SMA_50"].iloc[i-1]) else 0
                s200_prev= float(df["SMA_200"].iloc[i-1]) if not pd.isna(df["SMA_200"].iloc[i-1]) else 0
                s50_curr = float(df["SMA_50"].iloc[i]) if not pd.isna(df["SMA_50"].iloc[i]) else 0
                s200_curr= float(df["SMA_200"].iloc[i]) if not pd.isna(df["SMA_200"].iloc[i]) else 0
                if s50_prev < s200_prev and s50_curr >= s200_curr:
                    trend_sc += 3; reasons.append("⚡ GOLDEN CROSS (SMA50 > SMA200) — major signal!")
                    break

        # ── Volume ────────────────────────────────────────────────────────────
        vol_ratio = float(last.get("Vol_Ratio", 1.0)) if not pd.isna(last.get("Vol_Ratio", np.nan)) else 1.0
        score.vol_ratio = round(vol_ratio, 2)
        if vol_ratio >= 3.0:
            vol_sc += 3; reasons.append(f"Volume {vol_ratio:.1f}× avg — very high surge 🔥")
        elif vol_ratio >= 2.0:
            vol_sc += 2; reasons.append(f"Volume {vol_ratio:.1f}× avg — high surge ✅")
        elif vol_ratio >= 1.5:
            vol_sc += 1; reasons.append(f"Volume {vol_ratio:.1f}× avg — above avg ✅")
        elif vol_ratio < 0.7:
            vol_sc -= 1  # thin volume, skip

        # ── ROC / Momentum ────────────────────────────────────────────────────
        roc5  = float(last.get("ROC_5",  0)) if not pd.isna(last.get("ROC_5",  np.nan)) else 0.0
        roc20 = float(last.get("ROC_20", 0)) if not pd.isna(last.get("ROC_20", np.nan)) else 0.0
        score.roc_5  = round(roc5,  2)
        score.roc_20 = round(roc20, 2)
        if roc5 >= 3:
            mom_sc += 2; reasons.append(f"5-day ROC +{roc5:.1f}% ✅")
        elif roc5 >= 1:
            mom_sc += 1
        if roc20 >= 10:
            mom_sc += 2; reasons.append(f"20-day ROC +{roc20:.1f}% — strong 1-month momentum ✅")
        elif roc20 >= 5:
            mom_sc += 1; reasons.append(f"20-day ROC +{roc20:.1f}%")

        # ── 52-Week High Proximity ────────────────────────────────────────────
        pct_52h = float(last.get("Pct_From_52W_High", -50)) if not pd.isna(last.get("Pct_From_52W_High", np.nan)) else -50.0
        score.pct_from_52w_high = round(pct_52h, 2)
        if -5 <= pct_52h <= 0:
            trend_sc += 3; score.near_52w_high = True
            reasons.append(f"Near 52-week high ({pct_52h:.1f}%) — breakout candidate 🔥")
        elif -10 <= pct_52h < -5:
            trend_sc += 1; reasons.append(f"Within 10% of 52-week high ✅")

        # ── Squeeze ───────────────────────────────────────────────────────────
        in_squeeze = bool(last.get("In_Squeeze", False)) if not pd.isna(last.get("In_Squeeze", np.nan)) else False
        score.in_squeeze = in_squeeze
        if in_squeeze:
            mom_sc += 1; reasons.append("Bollinger Squeeze — breakout may be imminent ⚡")

        # ── Breakout / Support Levels ─────────────────────────────────────────
        score.breakout_price = get_breakout_price(df, lookback=20)
        score.support_price  = get_breakdown_price(df, lookback=20)

        # ATR
        atr = float(last.get("ATR", 0)) if not pd.isna(last.get("ATR", np.nan)) else 0.0
        score.atr = round(atr, 2)

        # ── Bias determination ────────────────────────────────────────────────
        score.momentum_score = mom_sc
        score.trend_score    = trend_sc
        score.volume_score   = vol_sc
        score.total_score    = mom_sc + trend_sc + vol_sc
        score.reasons        = reasons

        total = score.total_score
        if total >= 8:
            score.bias = "Long"
        elif total >= 4:
            score.bias = "Long"      # still tradeable, lower conviction
        elif total <= -2:
            score.bias = "Short"
        else:
            score.bias = "Watch"

        return score

    except Exception as e:
        logger.error(f"Error scoring {ticker}: {e}")
        return None

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — MOMENTUM SCANNER
# ──────────────────────────────────────────────────────────────────────────────

def run_momentum_scan(universe: Dict[str, Tuple[str, str]],
                      tier_name: str = "All",
                      period: str = "3mo",
                      min_total_score: int = 6,
                      min_volume_ratio: float = 1.0,
                      progress_callback=None) -> pd.DataFrame:
    """
    Phase 1 core scanner.
    Returns a DataFrame of ranked stocks sorted by total_score desc.
    """
    tickers = list(universe.keys())
    total   = len(tickers)
    results: List[StockScore] = []

    data_map = parallel_fetch(tickers, period=period, interval="1d", max_workers=4)

    for idx, (ticker, (name, sector)) in enumerate(universe.items()):
        if progress_callback:
            progress_callback(idx + 1, total, f"Scanning {name}…")
        df = data_map.get(ticker)
        score = score_stock(ticker, name, sector, tier_name, df)
        if score and score.total_score >= min_total_score and score.vol_ratio >= min_volume_ratio:
            results.append(score)

    if not results:
        return pd.DataFrame()

    rows = []
    for s in sorted(results, key=lambda x: x.total_score, reverse=True):
        rows.append({
            "Score":          s.total_score,
            "Bias":           s.bias,
            "Ticker":         s.ticker,
            "Name":           s.name,
            "Sector":         s.sector,
            "Tier":           s.tier,
            "Price (₹)":      s.price,
            "RSI":            s.rsi,
            "MACD":           s.macd_signal,
            "EMA Trend":      s.ema_trend,
            "Vol Ratio":      s.vol_ratio,
            "ROC 5d %":       s.roc_5,
            "ROC 20d %":      s.roc_20,
            "Near 52W High":  "✅" if s.near_52w_high else "",
            "Squeeze":        "⚡" if s.in_squeeze else "",
            "Above SMA20":    "✅" if s.above_sma20  else "❌",
            "Above SMA50":    "✅" if s.above_sma50  else "❌",
            "Above SMA200":   "✅" if s.above_sma200 else "❌",
            "Breakout (₹)":   s.breakout_price,
            "Support (₹)":    s.support_price,
            "ATR (₹)":        s.atr,
            "% from 52W High":s.pct_from_52w_high,
        })

    df_out = pd.DataFrame(rows)
    return df_out


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — PRE-OPEN GAP + PRIORITY SCORER
# ──────────────────────────────────────────────────────────────────────────────

def run_preopen_gap_scan(tickers: List[str],
                         universe_lookup: Dict[str, Tuple[str, str]],
                         gap_threshold: float = 1.0,
                         progress_callback=None) -> Tuple[pd.DataFrame, str]:
    """
    Scans for gap-up / gap-down vs previous close.
    Returns (gap_df, market_bias_string).
    """
    results = []
    total   = len(tickers)
    data_map = parallel_fetch(tickers, period="5d", interval="1d", max_workers=4)

    for idx, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(idx + 1, total, f"Checking {ticker}…")
        name, sector = universe_lookup.get(ticker, (ticker, "Unknown"))
        df = data_map.get(ticker)
        if df is None or len(df) < 2:
            continue
        try:
            prev_close  = float(df["Close"].iloc[-2])
            today_open  = float(df["Open"].iloc[-1])
            today_close = float(df["Close"].iloc[-1])
            today_vol   = float(df["Volume"].iloc[-1])
            prev_vol    = float(df["Volume"].iloc[-2])
            gap_pct     = (today_open - prev_close) / prev_close * 100
            intra_move  = (today_close - today_open) / today_open * 100
            vol_ratio   = today_vol / prev_vol if prev_vol > 0 else 1.0

            if abs(gap_pct) >= gap_threshold:
                results.append({
                    "Ticker":         ticker,
                    "Name":           name,
                    "Sector":         sector,
                    "Prev Close (₹)": round(prev_close, 2),
                    "Open (₹)":       round(today_open,  2),
                    "LTP (₹)":        round(today_close, 2),
                    "Gap %":          round(gap_pct, 2),
                    "Intraday %":     round(intra_move, 2),
                    "Vol Ratio":      round(vol_ratio, 2),
                    "Type":           "⬆️ Gap Up" if gap_pct > 0 else "⬇️ Gap Down",
                })
        except Exception as e:
            logger.warning(f"Gap scan error {ticker}: {e}")

    if not results:
        return pd.DataFrame(), "🟡 Neutral"

    df_out = pd.DataFrame(results).sort_values("Gap %", ascending=False)
    ups    = (df_out["Gap %"] > 0).sum()
    downs  = (df_out["Gap %"] < 0).sum()
    if ups > downs * 1.5:
        bias = "🟢 Bullish Bias — More gap-ups than gap-downs"
    elif downs > ups * 1.5:
        bias = "🔴 Bearish Bias — More gap-downs than gap-ups"
    else:
        bias = "🟡 Mixed / Neutral — No clear directional bias"
    return df_out, bias


def prioritise_watchlist(watchlist: List[Dict],
                          universe_lookup: Dict[str, Tuple[str, str]],
                          progress_callback=None) -> List[Dict]:
    """
    Phase 2: Takes the pre-market watchlist and enriches / scores each stock
    using latest available price data. Returns list sorted by combined score.
    """
    tickers  = [w["ticker"] for w in watchlist]
    data_map = parallel_fetch(tickers, period="5d", interval="1d", max_workers=4)
    enriched = []
    total    = len(watchlist)

    for idx, item in enumerate(watchlist):
        if progress_callback:
            progress_callback(idx + 1, total, f"Scoring {item['ticker']}…")

        ticker    = item["ticker"]
        df        = data_map.get(ticker)
        score     = 0
        reasons   = []
        live_price = 0.0
        gap_pct    = 0.0
        vol_ratio  = 1.0

        if df is not None and len(df) >= 2:
            prev_close  = float(df["Close"].iloc[-2])
            today_open  = float(df["Open"].iloc[-1])
            live_price  = float(df["Close"].iloc[-1])
            gap_pct     = (today_open - prev_close) / prev_close * 100
            vol_ratio   = (float(df["Volume"].iloc[-1]) /
                           float(df["Volume"].iloc[-2])) \
                          if float(df["Volume"].iloc[-2]) > 0 else 1.0

            bias     = item.get("bias", "Long")
            breakout = item.get("breakout", 0)
            sl       = item.get("stop_loss", 0)
            target   = item.get("target", 0)

            # Gap alignment
            if bias == "Long":
                if gap_pct >= 0:
                    score += 2; reasons.append(f"Gap {gap_pct:+.2f}% aligns with Long ✅")
                else:
                    score -= 1; reasons.append(f"Gap {gap_pct:+.2f}% against Long ⚠️")
            else:
                if gap_pct <= 0:
                    score += 2; reasons.append(f"Gap {gap_pct:+.2f}% aligns with Short ✅")
                else:
                    score -= 1; reasons.append(f"Gap {gap_pct:+.2f}% against Short ⚠️")

            # Breakout proximity
            if breakout > 0:
                dist_pct = abs(live_price - breakout) / breakout * 100
                if dist_pct <= 0.5:
                    score += 3; reasons.append("⚡ AT breakout level — high conviction!")
                elif dist_pct <= 2.0:
                    score += 2; reasons.append(f"Within {dist_pct:.1f}% of breakout ✅")
                if bias == "Long" and live_price > breakout:
                    score += 2; reasons.append("Breakout already triggered 🚀")

            # Volume
            if vol_ratio >= 2.0:
                score += 2; reasons.append(f"Vol {vol_ratio:.1f}× — institutional interest 🔥")
            elif vol_ratio >= 1.5:
                score += 1; reasons.append(f"Vol {vol_ratio:.1f}× — above avg ✅")

            # R:R
            rr = 0.0
            if bias == "Long" and breakout > sl > 0 and target > breakout:
                rr = (target - breakout) / (breakout - sl)
            elif bias == "Short" and sl > breakout > target > 0:
                rr = (breakout - target) / (sl - breakout)
            if rr >= 2.5:
                score += 3; reasons.append(f"R:R 1:{rr:.1f} — excellent ✅")
            elif rr >= 2.0:
                score += 2; reasons.append(f"R:R 1:{rr:.1f} — good ✅")
            elif rr >= 1.5:
                score += 1; reasons.append(f"R:R 1:{rr:.1f}")
        else:
            reasons.append("Could not fetch live data")
            rr = 0.0

        label = "🔥 HIGH" if score >= 6 else "⚡ MEDIUM" if score >= 3 else "👁️ WATCH"
        enriched.append({
            **item,
            "live_price": round(live_price, 2),
            "gap_pct":    round(gap_pct, 2),
            "vol_ratio":  round(vol_ratio, 2),
            "rr":         round(rr, 2),
            "score":      score,
            "priority":   label,
            "reasons":    reasons,
        })

    enriched.sort(key=lambda x: x["score"], reverse=True)
    return enriched


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 — LIVE MARKET SCANNER
# ──────────────────────────────────────────────────────────────────────────────

def run_live_orb_scan(tickers: List[str],
                      universe_lookup: Dict[str, Tuple[str, str]],
                      orb_minutes: int = 15,
                      progress_callback=None) -> pd.DataFrame:
    """
    Opening Range Breakout (ORB) scanner.
    Uses intraday (5-minute) data to find stocks breaking above/below
    the first `orb_minutes` range after 9:15 AM.
    """
    results = []
    total   = len(tickers)
    data_map = parallel_fetch(tickers, period="1d", interval="5m", max_workers=4)

    for idx, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(idx + 1, total, f"ORB: {ticker}…")
        name, sector = universe_lookup.get(ticker, (ticker, "Unknown"))
        df = data_map.get(ticker)
        if df is None or df.empty:
            continue
        try:
            df.index = df.index.tz_localize("UTC").tz_convert(IST) \
                if df.index.tzinfo is None else df.index.tz_convert(IST)
            today      = get_ist_time().date()
            today_data = df[df.index.date == today]
            if today_data.empty:
                continue
            market_open_time = today_data.index[0]
            orb_end_time     = market_open_time + timedelta(minutes=orb_minutes)
            orb_candles      = today_data[today_data.index <= orb_end_time]
            if len(orb_candles) < 1:
                continue

            orb_high  = float(orb_candles["High"].max())
            orb_low   = float(orb_candles["Low"].min())
            last_price= float(today_data["Close"].iloc[-1])
            last_vol  = float(today_data["Volume"].iloc[-1])
            orb_range = orb_high - orb_low

            # Check breakout
            if last_price > orb_high and orb_range > 0:
                breakout_type = "⬆️ ORB Breakout"
                move_pct = (last_price - orb_high) / orb_high * 100
            elif last_price < orb_low and orb_range > 0:
                breakout_type = "⬇️ ORB Breakdown"
                move_pct = (orb_low - last_price) / orb_low * 100
            else:
                continue  # still inside ORB, skip

            # Volume surge during breakout
            avg_vol_5m = today_data["Volume"].mean()
            vol_surge  = last_vol / avg_vol_5m if avg_vol_5m > 0 else 1.0

            results.append({
                "Ticker":         ticker,
                "Name":           name,
                "Sector":         sector,
                "Type":           breakout_type,
                "LTP (₹)":        round(last_price, 2),
                "ORB High (₹)":   round(orb_high, 2),
                "ORB Low (₹)":    round(orb_low,  2),
                "ORB Range (₹)":  round(orb_range, 2),
                "Move %":         round(move_pct, 2),
                "Vol Surge":      round(vol_surge, 2),
                "Signal Strength":("🔥 Strong" if move_pct > 1 and vol_surge > 2
                                   else "⚡ Moderate" if move_pct > 0.5
                                   else "👁️ Weak"),
            })
        except Exception as e:
            logger.warning(f"ORB error {ticker}: {e}")

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("Move %", ascending=False)


def run_vwap_momentum_scan(tickers: List[str],
                            universe_lookup: Dict[str, Tuple[str, str]],
                            progress_callback=None) -> pd.DataFrame:
    """
    Scans for stocks with price significantly above/below intraday VWAP.
    Uses 5-min intraday data.
    """
    results  = []
    total    = len(tickers)
    data_map = parallel_fetch(tickers, period="1d", interval="5m", max_workers=4)

    for idx, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(idx + 1, total, f"VWAP: {ticker}…")
        name, sector = universe_lookup.get(ticker, (ticker, "Unknown"))
        df = data_map.get(ticker)
        if df is None or df.empty:
            continue
        try:
            df.index = df.index.tz_localize("UTC").tz_convert(IST) \
                if df.index.tzinfo is None else df.index.tz_convert(IST)
            today      = get_ist_time().date()
            today_data = df[df.index.date == today].copy()
            if len(today_data) < 3:
                continue

            typical  = (today_data["High"] + today_data["Low"] + today_data["Close"]) / 3
            cumvol   = today_data["Volume"].cumsum()
            cum_tp_v = (typical * today_data["Volume"]).cumsum()
            today_data["VWAP"] = cum_tp_v / cumvol

            last_price = float(today_data["Close"].iloc[-1])
            vwap       = float(today_data["VWAP"].iloc[-1])
            pct_vs_vwap= (last_price - vwap) / vwap * 100
            last_vol   = float(today_data["Volume"].iloc[-1])
            avg_vol    = float(today_data["Volume"].mean())
            vol_ratio  = last_vol / avg_vol if avg_vol > 0 else 1.0

            if abs(pct_vs_vwap) >= 0.5:  # Only list if meaningful deviation
                results.append({
                    "Ticker":       ticker,
                    "Name":         name,
                    "Sector":       sector,
                    "LTP (₹)":      round(last_price, 2),
                    "VWAP (₹)":     round(vwap, 2),
                    "% vs VWAP":    round(pct_vs_vwap, 2),
                    "Signal":       ("⬆️ Above VWAP" if pct_vs_vwap > 0 else "⬇️ Below VWAP"),
                    "Vol Ratio":    round(vol_ratio, 2),
                    "Bias":         ("Long" if pct_vs_vwap > 0 else "Short"),
                })
        except Exception as e:
            logger.warning(f"VWAP error {ticker}: {e}")

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("% vs VWAP", ascending=False)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL INDICES TREND CHECKER (Phase 1 helper)
# ──────────────────────────────────────────────────────────────────────────────

def get_market_trend_summary() -> Dict[str, Any]:
    """
    Fetches Indian + global indices and returns a trend summary dict.
    Used in Phase 1 to validate overall market direction.
    """
    summary = {"indian": {}, "global": {}, "overall_bias": "Neutral"}
    all_indices = {**INDIAN_INDICES, **GLOBAL_INDICES}
    data_map    = parallel_fetch(list(all_indices.values()), period="5d",
                                  interval="1d", max_workers=6)
    bullish = bearish = 0
    for name, ticker in all_indices.items():
        df = data_map.get(ticker)
        if df is None or len(df) < 2:
            continue
        try:
            prev  = float(df["Close"].iloc[-2])
            curr  = float(df["Close"].iloc[-1])
            chg   = curr - prev
            pct   = chg / prev * 100
            direction = "🟢 Up" if pct > 0.2 else ("🔴 Down" if pct < -0.2 else "🟡 Flat")
            entry = {"price": round(curr,2), "change": round(chg,2),
                     "pct": round(pct,2), "direction": direction}
            if name in INDIAN_INDICES:
                summary["indian"][name] = entry
            else:
                summary["global"][name] = entry
            if pct > 0.2:  bullish += 1
            elif pct < -0.2: bearish += 1
        except Exception:
            pass

    if bullish > bearish * 1.5:
        summary["overall_bias"] = "🟢 Broadly Bullish — prefer Longs"
    elif bearish > bullish * 1.5:
        summary["overall_bias"] = "🔴 Broadly Bearish — prefer Shorts / avoid longs"
    else:
        summary["overall_bias"] = "🟡 Mixed — trade selectively"

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# WATCHLIST TRADE PLAN GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def generate_trade_plan(ticker: str, price: float, atr: float,
                        bias: str = "Long",
                        atr_sl_mult: float = 1.5,
                        atr_target_mult: float = 3.0) -> Dict[str, float]:
    """
    Generates stop-loss and target based on ATR multiples.
    Returns dict with entry, sl, target, rr.
    """
    if atr <= 0:
        atr = price * 0.01  # fallback: 1% of price
    if bias == "Long":
        sl     = round(price - atr_sl_mult * atr, 2)
        target = round(price + atr_target_mult * atr, 2)
    else:
        sl     = round(price + atr_sl_mult * atr, 2)
        target = round(price - atr_target_mult * atr, 2)
    risk   = abs(price - sl)
    reward = abs(target - price)
    rr     = round(reward / risk, 2) if risk > 0 else 0
    return {"entry": price, "sl": sl, "target": target, "rr": rr}


# ──────────────────────────────────────────────────────────────────────────────
# ▌ADD-TO-EXISTING (SECTION B) — STREAMLIT PAGE FUNCTION
#   Paste this function into stock_market_platform.py inside main(),
#   then add "🎯 Stock Selection Engine" to the sidebar radio and
#   add:  elif page == "🎯 Stock Selection Engine": show_stock_selection_engine()
#   to the PAGE DISPATCH ROUTER at the bottom.
# ──────────────────────────────────────────────────────────────────────────────

def show_stock_selection_engine():
    """
    Full Streamlit UI for the end-to-end stock selection pipeline.
    Three phases: Pre-Market → Pre-Open → Live Market.
    """
    st.header("🎯 Stock Selection Engine")
    st.markdown(
        "**End-to-end pipeline: Momentum Scan → Market Trend → "
        "Gap Scanner → Watchlist → Live ORB / VWAP**"
    )

    now_ist = get_ist_time()
    hour, minute = now_ist.hour, now_ist.minute

    # ── Phase status banner ───────────────────────────────────────────────────
    if hour < 9:
        mins_to_open = (9 - hour) * 60 - minute
        st.success(f"🌅 **PRE-MARKET PHASE** — Market opens in {mins_to_open} min. Run Phase 1 now.")
    elif hour == 9 and minute < 15:
        st.warning(f"⚡ **PRE-OPEN SESSION ACTIVE** — {15 - minute} min until market opens. Run Phase 2.")
    elif is_market_open():
        st.error(f"🔴 **LIVE MARKET** — {now_ist.strftime('%I:%M %p IST')}. Run Phase 3 scanners.")
    else:
        st.info(f"🔵 Market closed — {now_ist.strftime('%I:%M %p IST')}. Review or prepare for tomorrow.")

    st.divider()

    phase1_tab, phase2_tab, phase3_tab, settings_tab = st.tabs([
        "🌅 Phase 1 — Pre-Market Scan",
        "⚡ Phase 2 — Pre-Open Prioritiser",
        "🔴 Phase 3 — Live Market Scanner",
        "⚙️ Settings",
    ])

    # ────────────────────────────────────────────────────────────────────────
    # PHASE 1 — PRE-MARKET (Before 9 AM)
    # ────────────────────────────────────────────────────────────────────────
    with phase1_tab:
        st.markdown("### 🌅 Phase 1: Previous Day Scan (Run before 9:00 AM)")
        st.markdown(
            "> **What this does:** Scans your selected universe for stocks with "
            "momentum, volume surge, trend alignment and proximity to breakout levels. "
            "This replaces manually using Streak / ChartInk — everything is automated here."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            selected_tier = st.selectbox(
                "📊 Universe",
                list(UNIVERSE_TIERS.keys()),
                index=0,
                key="p1_tier",
                help="Start with Nifty 50 for speed; use All (~160) for full scan"
            )
        with col2:
            min_score = st.slider("Min Total Score", 0, 15, 6, key="p1_min_score",
                                  help="Higher = fewer but stronger signals")
        with col3:
            min_vol   = st.slider("Min Volume Ratio", 0.5, 3.0, 1.0, 0.1, key="p1_min_vol",
                                  help="Volume vs 20-day average")

        col4, col5 = st.columns(2)
        with col4:
            sector_filter = st.multiselect(
                "Filter by Sector (optional)",
                sorted(set(v[1] for v in FULL_UNIVERSE.values())),
                key="p1_sectors"
            )
        with col5:
            scan_period = st.selectbox("Lookback Period", ["1mo", "3mo", "6mo"], index=1, key="p1_period")

        st.divider()

        # ── Market Trend Section ──────────────────────────────────────────────
        st.markdown("#### 🌏 Step A: Market Trend Check (Global + Indian Indices)")
        if st.button("📡 Fetch Market Trend", key="p1_market_trend", use_container_width=True):
            with st.spinner("Fetching index data…"):
                trend = get_market_trend_summary()
            st.markdown(f"### Overall Market Bias: {trend['overall_bias']}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🇮🇳 Indian Indices**")
                for idx_name, data in trend["indian"].items():
                    color = "#00d084" if "Up" in data["direction"] else \
                            "#ff4444" if "Down" in data["direction"] else "#ffa500"
                    st.markdown(
                        f"<div style='border-left:3px solid {color};"
                        f"padding:6px 12px;margin:4px 0;background:#1a1d2e;border-radius:4px;'>"
                        f"<b>{idx_name}</b>: {data['price']:,.0f} "
                        f"<span style='color:{color};'>{data['pct']:+.2f}% {data['direction']}</span>"
                        f"</div>", unsafe_allow_html=True
                    )
            with c2:
                st.markdown("**🌍 Global Indices**")
                for idx_name, data in trend["global"].items():
                    color = "#00d084" if "Up" in data["direction"] else \
                            "#ff4444" if "Down" in data["direction"] else "#ffa500"
                    st.markdown(
                        f"<div style='border-left:3px solid {color};"
                        f"padding:6px 12px;margin:4px 0;background:#1a1d2e;border-radius:4px;'>"
                        f"<b>{idx_name}</b>: {data['price']:,.0f} "
                        f"<span style='color:{color};'>{data['pct']:+.2f}% {data['direction']}</span>"
                        f"</div>", unsafe_allow_html=True
                    )

            st.session_state["market_trend"] = trend

        if "market_trend" in st.session_state:
            mt = st.session_state["market_trend"]
            st.info(f"📊 Last fetched market bias: **{mt['overall_bias']}**")

        st.divider()

        # ── Momentum Scanner ──────────────────────────────────────────────────
        st.markdown("#### 🔍 Step B: Momentum + Technical Scanner")
        if st.button("🚀 Run Full Momentum Scan", type="primary",
                     key="p1_run_scan", use_container_width=True):

            universe = UNIVERSE_TIERS[selected_tier]
            if sector_filter:
                universe = {k: v for k, v in universe.items() if v[1] in sector_filter}

            st.info(f"Scanning {len(universe)} stocks… This may take 60–120 seconds for large universes.")
            progress_bar  = st.progress(0)
            status_text   = st.empty()

            def cb(cur, total, msg):
                progress_bar.progress(cur / total)
                status_text.text(msg)

            with st.spinner(""):
                df_results = run_momentum_scan(
                    universe, tier_name=selected_tier,
                    period=scan_period,
                    min_total_score=min_score,
                    min_volume_ratio=min_vol,
                    progress_callback=cb
                )

            progress_bar.empty()
            status_text.empty()

            if df_results.empty:
                st.warning("No stocks met the criteria. Try lowering the score or volume threshold.")
            else:
                st.success(f"✅ Found **{len(df_results)} stocks** matching criteria")
                st.session_state["p1_scan_results"] = df_results

        # ── Display scan results ──────────────────────────────────────────────
        if "p1_scan_results" in st.session_state:
            df_r = st.session_state["p1_scan_results"]
            st.markdown(f"#### 📋 Scan Results — {len(df_r)} Stocks")

            # Colour-code the Score column
            def highlight_score(val):
                if isinstance(val, (int, float)):
                    if val >= 12: return "background-color:#1e3a2e;color:#00d084;"
                    if val >= 8:  return "background-color:#2e2e1e;color:#ffd700;"
                    if val >= 5:  return "background-color:#1e1e3a;color:#a78bfa;"
                return ""

            st.dataframe(
                df_r.style.applymap(highlight_score, subset=["Score"]),
                use_container_width=True, hide_index=True
            )

            # ── Add to watchlist ──────────────────────────────────────────────
            st.markdown("#### 📌 Add to Pre-Market Watchlist")
            selected_tickers = st.multiselect(
                "Select stocks to add to watchlist",
                options=df_r["Ticker"].tolist(),
                default=df_r["Ticker"].head(5).tolist() if len(df_r) >= 5 else df_r["Ticker"].tolist(),
                key="p1_selected"
            )
            atr_sl_mult  = st.slider("SL = Price − (ATR ×)", 0.5, 3.0, 1.5, 0.1, key="p1_sl_mult")
            atr_tgt_mult = st.slider("Target = Price + (ATR ×)", 1.0, 6.0, 3.0, 0.5, key="p1_tgt_mult")

            if st.button("📋 Add Selected to Watchlist", key="p1_add_wl", use_container_width=True):
                if "pre_watchlist" not in st.session_state:
                    st.session_state.pre_watchlist = []
                added = 0
                for ticker in selected_tickers:
                    row = df_r[df_r["Ticker"] == ticker].iloc[0]
                    price = float(row["Price (₹)"])
                    atr   = float(row["ATR (₹)"])
                    plan  = generate_trade_plan(ticker, price, atr, bias=row["Bias"],
                                               atr_sl_mult=atr_sl_mult,
                                               atr_target_mult=atr_tgt_mult)
                    new_entry = {
                        "ticker":    ticker,
                        "bias":      row["Bias"],
                        "breakout":  float(row["Breakout (₹)"]),
                        "stop_loss": plan["sl"],
                        "target":    plan["target"],
                        "notes":     f"Auto-added: Score {row['Score']} | {row['Sector']}",
                        "added_at":  now_ist.strftime("%H:%M IST"),
                        "score_p1":  int(row["Score"]),
                    }
                    existing = [w["ticker"] for w in st.session_state.pre_watchlist]
                    if ticker not in existing:
                        st.session_state.pre_watchlist.append(new_entry)
                        added += 1
                st.success(f"✅ Added {added} stocks to watchlist! Go to Phase 2 for pre-open prioritisation.")

            # ── Export ────────────────────────────────────────────────────────
            csv = df_r.to_csv(index=False)
            st.download_button(
                "⬇️ Export Scan Results (CSV)",
                data=csv,
                file_name=f"momentum_scan_{now_ist.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="p1_export"
            )

        # ── External scanner links (ChartInk, Streak, NSE OI) ─────────────────
        st.divider()
        st.markdown("#### 🔗 External Resources (as per Fundfolio methodology)")
        ext_cols = st.columns(3)
        with ext_cols[0]:
            st.markdown("**📡 Scanners**")
            for label, url in {
                "ChartInk Screener":    "https://chartink.com/screener/",
                "Streak Scanner":       "https://streak.tech/",
                "Tickertape Screener":  "https://www.tickertape.in/screener",
                "Trendlyne Screener":   "https://trendlyne.com/screener/",
            }.items():
                st.markdown(f"• [{label}]({url})")
        with ext_cols[1]:
            st.markdown("**📊 NSE OI Data**")
            for label, url in {
                "NSE Option Chain":     "https://www.nseindia.com/option-chain",
                "NSE Market Activity":  "https://www.nseindia.com/market-data/most-active-securities",
                "NSE FII/DII Data":     "https://www.nseindia.com/research/fiidiitrends",
                "NSE Bulk Deals":       "https://www.nseindia.com/market-data/bulk-deals",
            }.items():
                st.markdown(f"• [{label}]({url})")
        with ext_cols[2]:
            st.markdown("**🌏 Global Indices**")
            for label, url in {
                "CNBC World Markets":   "https://www.cnbctv18.com/market/world-market",
                "Gift Nifty (SGX)":     "https://www.nseindia.com/market-data/giftCity",
                "Investing.com":        "https://www.investing.com/indices/world-indices",
                "TradingEconomics":     "https://tradingeconomics.com/stocks",
            }.items():
                st.markdown(f"• [{label}]({url})")

    # ────────────────────────────────────────────────────────────────────────
    # PHASE 2 — PRE-OPEN (9:00–9:15 AM)
    # ────────────────────────────────────────────────────────────────────────
    with phase2_tab:
        st.markdown("### ⚡ Phase 2: Pre-Open Session (9:00 – 9:15 AM)")
        st.markdown(
            "> **What this does:** Scans your watchlist + selected universe for gap-ups/downs, "
            "cross-checks against your Phase 1 watchlist, and produces a final priority-ranked "
            "list so you know exactly which stocks to focus on the moment the market opens."
        )

        p2_tabs = st.tabs([
            "📈 Gap Scanner",
            "🏆 Watchlist Prioritiser",
        ])

        # ── Gap Scanner ───────────────────────────────────────────────────────
        with p2_tabs[0]:
            st.markdown("#### 📈 Gap Up / Gap Down Scanner")
            st.info(
                "NSE pre-open data (9:00–9:15 AM) determines the day's opening prices. "
                "Gap-ups from your watchlist = high priority. "
                "Gap-downs = caution (wait for support or go short)."
            )

            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                gap_universe_choice = st.selectbox(
                    "Scan Universe", list(UNIVERSE_TIERS.keys()), key="p2_gap_universe"
                )
            with gc2:
                gap_threshold = st.slider("Min Gap %", 0.5, 5.0, 1.0, 0.25, key="p2_gap_thresh")
            with gc3:
                include_watchlist = st.checkbox("Include Watchlist stocks", value=True, key="p2_incl_wl")

            if st.button("🔍 Scan for Gaps", type="primary", key="p2_run_gap", use_container_width=True):
                gap_tickers = list(UNIVERSE_TIERS[gap_universe_choice].keys())
                if include_watchlist and "pre_watchlist" in st.session_state:
                    wl_tickers  = [w["ticker"] for w in st.session_state.pre_watchlist]
                    gap_tickers = list(set(gap_tickers + wl_tickers))

                lookup = {**FULL_UNIVERSE}
                if "pre_watchlist" in st.session_state:
                    for w in st.session_state.pre_watchlist:
                        if w["ticker"] not in lookup:
                            lookup[w["ticker"]] = (w["ticker"], "Watchlist")

                progress_bar = st.progress(0)
                status_text  = st.empty()

                def gap_cb(cur, total, msg):
                    progress_bar.progress(cur / total)
                    status_text.text(msg)

                with st.spinner(""):
                    gap_df, market_bias = run_preopen_gap_scan(
                        gap_tickers, lookup, gap_threshold, gap_cb
                    )

                progress_bar.empty()
                status_text.empty()

                if gap_df.empty:
                    st.info(f"No stocks with gap ≥ {gap_threshold}%. Market may be flat.")
                else:
                    st.markdown(f"### Market Opening Bias: **{market_bias}**")
                    wl_tickers_set = set(w["ticker"] for w in st.session_state.get("pre_watchlist", []))

                    gap_ups   = gap_df[gap_df["Gap %"] > 0]
                    gap_downs = gap_df[gap_df["Gap %"] < 0]

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("⬆️ Gap Ups",   len(gap_ups))
                    mc2.metric("⬇️ Gap Downs",  len(gap_downs))
                    wl_in_gap = gap_df[gap_df["Ticker"].isin(wl_tickers_set)]
                    mc3.metric("⭐ In Watchlist", len(wl_in_gap))

                    if not wl_in_gap.empty:
                        st.success(
                            f"⭐ **{len(wl_in_gap)} watchlist stock(s) have significant gaps — PRIORITY!**"
                        )
                        st.dataframe(wl_in_gap, use_container_width=True, hide_index=True)
                        st.divider()

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### ⬆️ Gap Ups")
                        if not gap_ups.empty:
                            st.dataframe(gap_ups, use_container_width=True, hide_index=True)
                        else:
                            st.info("No significant gap-ups.")
                    with c2:
                        st.markdown("#### ⬇️ Gap Downs")
                        if not gap_downs.empty:
                            st.dataframe(gap_downs, use_container_width=True, hide_index=True)
                        else:
                            st.info("No significant gap-downs.")

                    st.session_state["p2_gap_results"] = {
                        "df": gap_df, "bias": market_bias,
                        "timestamp": now_ist.strftime("%H:%M IST")
                    }

        # ── Watchlist Prioritiser ─────────────────────────────────────────────
        with p2_tabs[1]:
            st.markdown("#### 🏆 Watchlist Prioritiser")
            st.markdown(
                "Cross-references your Phase 1 watchlist with current pre-open data. "
                "Each stock is scored: gap alignment + breakout proximity + volume + R:R."
            )

            pre_wl = st.session_state.get("pre_watchlist", [])
            if not pre_wl:
                st.warning(
                    "Your watchlist is empty. Run Phase 1 Momentum Scan and add stocks to watchlist first."
                )
            else:
                st.info(f"Watchlist has {len(pre_wl)} stocks. Click below to score and prioritise.")
                if st.button("🔄 Prioritise Watchlist Now", type="primary",
                             key="p2_prioritise", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text  = st.empty()

                    def pri_cb(cur, total, msg):
                        progress_bar.progress(cur / total)
                        status_text.text(msg)

                    with st.spinner(""):
                        prioritised = prioritise_watchlist(pre_wl, FULL_UNIVERSE, pri_cb)

                    progress_bar.empty()
                    status_text.empty()
                    st.session_state["p2_prioritised"] = prioritised
                    st.success(f"✅ Prioritised {len(prioritised)} stocks!")

                prioritised = st.session_state.get("p2_prioritised", [])
                if prioritised:
                    st.markdown("### 🏆 Priority Ranked Trade List")

                    for rank, item in enumerate(prioritised, 1):
                        score       = item.get("score", 0)
                        bias        = item.get("bias", "Watch")
                        live_price  = item.get("live_price", 0)
                        gap_pct     = item.get("gap_pct", 0)
                        vol_ratio   = item.get("vol_ratio", 1)
                        breakout    = item.get("breakout", 0)
                        sl          = item.get("stop_loss", 0)
                        target      = item.get("target", 0)
                        rr          = item.get("rr", 0)
                        priority    = item.get("priority", "👁️ WATCH")
                        reasons     = item.get("reasons", [])

                        score_color = "#00d084" if score >= 6 else "#ffa500" if score >= 3 else "#888"
                        bias_color  = "#00d084" if bias == "Long" else "#ff4444" if bias == "Short" else "#a78bfa"
                        gap_color   = "#00d084" if gap_pct > 0 else "#ff4444"

                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#1e2535,#252b3f);
                                    border:1px solid #2d3548;border-left:5px solid {score_color};
                                    border-radius:10px;padding:16px;margin:10px 0;">
                          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
                            <div>
                              <span style="color:#888;font-size:0.85rem;">#{rank}</span>
                              <span style="color:#fff;font-size:1.15rem;font-weight:700;margin:0 10px;">{item['ticker']}</span>
                              <span style="background:{bias_color}22;color:{bias_color};
                                           padding:2px 8px;border-radius:10px;font-size:0.8rem;">{bias}</span>
                              <span style="margin-left:10px;color:{score_color};font-weight:600;">{priority}</span>
                            </div>
                            <div style="text-align:right;">
                              <span style="color:#fff;font-size:1.1rem;">₹{live_price:,.2f}</span>
                              <span style="margin-left:8px;color:{gap_color};">{gap_pct:+.2f}% gap</span>
                              <span style="margin-left:10px;background:{score_color}33;color:{score_color};
                                           padding:2px 8px;border-radius:8px;">Score {score}</span>
                            </div>
                          </div>
                          <div style="display:flex;gap:20px;margin-top:10px;font-size:0.83rem;color:#aaa;flex-wrap:wrap;">
                            <span>🎯 BO: <b style="color:#fff;">₹{breakout:,.2f}</b></span>
                            <span>🛑 SL: <b style="color:#ff4444;">₹{sl:,.2f}</b></span>
                            <span>🏁 Tgt: <b style="color:#00d084;">₹{target:,.2f}</b></span>
                            <span>R:R: <b style="color:{'#00d084' if rr>=2 else '#ffa500' if rr>=1.5 else '#888'};">
                              {'1:'+str(round(rr,1)) if rr>0 else '–'}</b></span>
                            <span>Vol: <b>{vol_ratio:.1f}×</b></span>
                          </div>
                          <div style="margin-top:8px;font-size:0.82rem;color:#aaa;">
                            {'  •  '.join(reasons[:4]) if reasons else '—'}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Export final prioritised list
                    export_rows = [{
                        "Rank":        rank + 1,
                        "Ticker":      x["ticker"],
                        "Bias":        x["bias"],
                        "Priority":    x["priority"],
                        "Score":       x["score"],
                        "Live Price":  x["live_price"],
                        "Gap %":       x["gap_pct"],
                        "Vol Ratio":   x["vol_ratio"],
                        "Breakout":    x["breakout"],
                        "Stop Loss":   x["stop_loss"],
                        "Target":      x["target"],
                        "R:R":         x["rr"],
                        "Reasons":     " | ".join(x["reasons"]),
                    } for rank, x in enumerate(prioritised)]
                    csv = pd.DataFrame(export_rows).to_csv(index=False)
                    st.download_button(
                        "⬇️ Export Priority List (CSV)",
                        data=csv,
                        file_name=f"priority_list_{now_ist.strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        key="p2_export"
                    )

    # ────────────────────────────────────────────────────────────────────────
    # PHASE 3 — LIVE MARKET (After 9:15 AM)
    # ────────────────────────────────────────────────────────────────────────
    with phase3_tab:
        st.markdown("### 🔴 Phase 3: Live Market Scanners (After 9:15 AM)")
        st.markdown(
            "> **What this does:** Runs real-time ORB and VWAP scans on your watchlist + broader universe. "
            "Only take trades that are confirmed by both Phase 1/2 preparation AND live breakout signals."
        )

        p3_tabs = st.tabs([
            "💥 ORB Scanner",
            "📊 VWAP Momentum",
        ])

        live_universe_choice = st.selectbox(
            "Live Scan Universe",
            ["Watchlist Only"] + list(UNIVERSE_TIERS.keys()),
            key="p3_universe"
        )

        def get_live_tickers():
            if live_universe_choice == "Watchlist Only":
                wl = st.session_state.get("pre_watchlist", [])
                if not wl:
                    return list(NIFTY_50.keys())[:20]
                return [w["ticker"] for w in wl]
            return list(UNIVERSE_TIERS[live_universe_choice].keys())

        # ── ORB Scanner ───────────────────────────────────────────────────────
        with p3_tabs[0]:
            st.markdown("#### 💥 Opening Range Breakout (ORB) Scanner")
            st.info(
                "ORB is one of the highest-probability intraday setups. "
                "The opening range is defined by the first N minutes of trade. "
                "A breakout above ORB high with volume = Long. Breakdown below ORB low = Short."
            )
            orb_mins = st.selectbox("Opening Range (minutes)", [5, 10, 15, 30], index=2, key="p3_orb_mins")

            if st.button("💥 Scan for ORB Breakouts", type="primary",
                         key="p3_run_orb", use_container_width=True):
                if not is_market_open():
                    st.warning(
                        "⚠️ Market is currently closed. ORB scanner requires live intraday data. "
                        "Results shown when market is open (9:15 AM – 3:30 PM IST)."
                    )
                tickers = get_live_tickers()
                lookup  = {**FULL_UNIVERSE}

                progress_bar = st.progress(0)
                status_text  = st.empty()

                def orb_cb(cur, total, msg):
                    progress_bar.progress(cur / total)
                    status_text.text(msg)

                with st.spinner(""):
                    orb_df = run_live_orb_scan(tickers, lookup,
                                               orb_minutes=orb_mins,
                                               progress_callback=orb_cb)
                progress_bar.empty()
                status_text.empty()

                if orb_df.empty:
                    st.info(
                        "No ORB breakouts found yet. This is normal early in the session. "
                        "Re-run after 9:30 AM once ranges are established."
                    )
                else:
                    st.success(f"✅ Found {len(orb_df)} ORB breakout/breakdown signals!")
                    breakouts  = orb_df[orb_df["Type"].str.contains("Breakout")]
                    breakdowns = orb_df[orb_df["Type"].str.contains("Breakdown")]

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### ⬆️ ORB Breakouts (Long Bias)")
                        if not breakouts.empty:
                            st.dataframe(breakouts, use_container_width=True, hide_index=True)
                        else:
                            st.info("No ORB breakouts yet.")
                    with c2:
                        st.markdown("#### ⬇️ ORB Breakdowns (Short Bias)")
                        if not breakdowns.empty:
                            st.dataframe(breakdowns, use_container_width=True, hide_index=True)
                        else:
                            st.info("No ORB breakdowns yet.")

        # ── VWAP Momentum ─────────────────────────────────────────────────────
        with p3_tabs[1]:
            st.markdown("#### 📊 VWAP Momentum Scanner")
            st.info(
                "VWAP (Volume-Weighted Average Price) is the key institutional benchmark. "
                "Price > VWAP = institutional buying. Price < VWAP = institutional selling. "
                "Best setups: breakout stocks from Phase 1 that are also above VWAP intraday."
            )

            if st.button("📊 Scan VWAP Momentum", type="primary",
                         key="p3_run_vwap", use_container_width=True):
                if not is_market_open():
                    st.warning(
                        "⚠️ Market is closed. VWAP scanner uses live intraday data. "
                        "Run during market hours (9:15 AM – 3:30 PM IST)."
                    )
                tickers = get_live_tickers()
                lookup  = {**FULL_UNIVERSE}

                progress_bar = st.progress(0)
                status_text  = st.empty()

                def vwap_cb(cur, total, msg):
                    progress_bar.progress(cur / total)
                    status_text.text(msg)

                with st.spinner(""):
                    vwap_df = run_vwap_momentum_scan(tickers, lookup, vwap_cb)

                progress_bar.empty()
                status_text.empty()

                if vwap_df.empty:
                    st.info("No significant VWAP deviation found. Try broadening the universe.")
                else:
                    st.success(f"✅ Found {len(vwap_df)} stocks with VWAP signals!")
                    above = vwap_df[vwap_df["Bias"] == "Long"]
                    below = vwap_df[vwap_df["Bias"] == "Short"]

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### ⬆️ Above VWAP (Long Bias)")
                        if not above.empty:
                            st.dataframe(above, use_container_width=True, hide_index=True)
                        else:
                            st.info("No stocks significantly above VWAP.")
                    with c2:
                        st.markdown("#### ⬇️ Below VWAP (Short Bias)")
                        if not below.empty:
                            st.dataframe(below, use_container_width=True, hide_index=True)
                        else:
                            st.info("No stocks significantly below VWAP.")

    # ────────────────────────────────────────────────────────────────────────
    # SETTINGS TAB
    # ────────────────────────────────────────────────────────────────────────
    with settings_tab:
        st.markdown("### ⚙️ Engine Settings & Scoring Reference")

        st.markdown("""
        #### 📊 How Scores Are Calculated

        Each stock is scored across three dimensions:

        | Dimension | Max Points | Key Signals |
        |-----------|-----------|-------------|
        | **Momentum** | ~10 | RSI 55–75, MACD histogram expanding, ROC 5d/20d, BB Squeeze |
        | **Trend**    | ~12 | EMA 9>21, Price>SMA20/50/200, Golden Cross, Near 52W High |
        | **Volume**   | ~6  | Volume ratio vs 20-day avg (1.5× = +1, 2× = +2, 3× = +3) |

        **Total Score interpretation:**
        - **≥ 12**: 🔥 Extremely high conviction — top breakout candidate
        - **8 – 11**: ✅ Strong setup — add to watchlist
        - **5 – 7**:  ⚡ Moderate — monitor, confirm with TA
        - **< 5**:    👁️ Weak — skip or watch only

        #### 🎯 Phase Priority Rules (Fundfolio Methodology)
        1. **Phase 1** → Never enter without a scan. Use this every evening.
        2. **Phase 2** → Cross-check your watchlist vs gap data at 9:00 AM.
           Only trade stocks where gap direction aligns with your bias.
        3. **Phase 3** → Entry only when ORB or VWAP confirms. 
           Do NOT chase. Wait for the breakout + volume.

        #### 📐 ATR-Based Trade Plan
        - **Stop Loss** = Entry − (1.5 × ATR) for Long, Entry + (1.5 × ATR) for Short
        - **Target**    = Entry + (3.0 × ATR) for Long, Entry − (3.0 × ATR) for Short
        - **R:R**       = Minimum 1:2 (adjust multipliers if needed)

        #### ⚠️ Risk Reminders
        - Never risk more than 1–2% of capital on a single trade.
        - If market trend (Phase 1, Step A) is Bearish, reduce position sizes or skip.
        - Set price alerts for your breakout levels before 9:15 AM.
        - Do not place market orders during pre-open (9:00–9:15 AM).
        """)

        st.divider()
        st.markdown("#### 🧹 Clear Engine Data")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Scan Results (Phase 1)", key="clr_p1"):
                st.session_state.pop("p1_scan_results", None)
                st.success("Phase 1 results cleared.")
        with col2:
            if st.button("🗑️ Clear Priority List (Phase 2)", key="clr_p2"):
                st.session_state.pop("p2_prioritised", None)
                st.session_state.pop("p2_gap_results", None)
                st.success("Phase 2 results cleared.")

# Alias for backward compatibility with platform functions
NIFTY_50_DATA = NIFTY_50


def main():
    st.set_page_config(
        page_title="Fundfolio — Indian Market Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
    /* ─── GLOBAL ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background:
            radial-gradient(circle at top, rgba(124,58,237,0.22) 0%, rgba(12,17,32,0.12) 28%, rgba(8,11,20,0.0) 52%),
            linear-gradient(160deg, #080b14 0%, #0b1020 45%, #090d18 100%);
        color: #edf1f8;
    }
    /* ─── SCROLLBAR ──────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0d1421; }
    ::-webkit-scrollbar-thumb { background: #2a3a5c; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a5080; }

    /* ─── SIDEBAR ────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(11, 16, 32, 0.72) !important;
        border-right: 1px solid rgba(160, 174, 198, 0.14);
        backdrop-filter: blur(18px) saturate(165%);
        -webkit-backdrop-filter: blur(18px) saturate(165%);
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #d0d8e6 !important;
        font-size: 0.88rem;
        padding: 4px 0;
        transition: color 0.2s;
    }
    section[data-testid="stSidebar"] .stRadio label:hover { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; font-size: 0.95rem !important; }

    .glass-panel {
        background: rgba(18, 24, 40, 0.58);
        border: 1px solid rgba(180, 190, 210, 0.14);
        border-radius: 18px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22);
        backdrop-filter: blur(18px) saturate(160%);
        -webkit-backdrop-filter: blur(18px) saturate(160%);
    }

    .glass-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #eef2ff;
        font-size: 0.78rem;
    }

    /* ─── METRIC CARDS ───────────────────────────────────── */
    .metric-card {
        background: rgba(16, 22, 36, 0.58);
        border: 1px solid rgba(180, 190, 210, 0.14);
        border-radius: 16px;
        padding: 18px 20px;
        margin: 6px 0;
        box-shadow: 0 10px 24px rgba(0,0,0,0.18);
        backdrop-filter: blur(16px) saturate(150%);
        -webkit-backdrop-filter: blur(16px) saturate(150%);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.26);
    }
    .metric-card .label { color: #8c98ad; font-size: 0.78rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
    .metric-card .value { font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
    .metric-card .change { font-size: 0.85rem; margin-top: 5px; font-weight: 500; }

    /* ─── INDEX CARD ─────────────────────────────────────── */
    .index-card {
        background: rgba(16, 22, 36, 0.62);
        border: 1px solid rgba(180, 190, 210, 0.14);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
        backdrop-filter: blur(16px) saturate(150%);
        -webkit-backdrop-filter: blur(16px) saturate(150%);
        transition: transform 0.2s;
    }
    .index-card:hover { transform: translateY(-3px); }
    .index-card .ix-name { color: #4a7090; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }
    .index-card .ix-value { color: #e2e8f0; font-size: 2.2rem; font-weight: 700; margin: 6px 0; }
    .index-card .ix-change { font-size: 0.9rem; font-weight: 600; }

    /* ─── SIGNAL CARDS ───────────────────────────────────── */
    .success-signal {
        background: rgba(14, 29, 24, 0.58);
        border-left: 3px solid #22c55e;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 6px 0;
    }
    .warning-signal {
        background: rgba(33, 26, 16, 0.58);
        border-left: 3px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 6px 0;
    }
    .danger-signal {
        background: rgba(33, 15, 20, 0.58);
        border-left: 3px solid #ef4444;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 6px 0;
    }
    .info-signal {
        background: rgba(17, 21, 41, 0.58);
        border-left: 3px solid #7dd3fc;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 6px 0;
    }

    /* ─── BUTTONS ────────────────────────────────────────── */
    .stButton > button {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
        border: 1px solid rgba(226, 232, 240, 0.18) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.2s !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.16) !important;
        backdrop-filter: blur(14px) saturate(150%);
        -webkit-backdrop-filter: blur(14px) saturate(150%);
    }
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.22) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, rgba(248,250,252,0.14) 0%, rgba(226,232,240,0.12) 100%) !important;
        border: 1px solid rgba(226,232,240,0.24) !important;
        box-shadow: 0 3px 12px rgba(0,0,0,0.18) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0.14) 100%) !important;
        box-shadow: 0 5px 18px rgba(0,0,0,0.22) !important;
    }

    /* ─── TABS ───────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 12px 12px 0 0 !important;
        padding: 6px 10px 0 10px !important;
        border-bottom: 1px solid rgba(180,190,210,0.12) !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #98a6bd !important;
        border-radius: 8px 8px 0 0 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        border: none !important;
        transition: all 0.2s !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #eef2ff !important; background: rgba(255,255,255,0.05) !important; }
    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,0.08) !important;
        color: #f8fafc !important;
        border-bottom: 2px solid #e2e8f0 !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(14, 20, 34, 0.52) !important;
        border: 1px solid rgba(180, 190, 210, 0.14) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: 20px !important;
        backdrop-filter: blur(16px) saturate(150%);
        -webkit-backdrop-filter: blur(16px) saturate(150%);
    }

    /* ─── INPUTS ─────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(180,190,210,0.16) !important;
        border-radius: 9px !important;
        color: #f0f4fb !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #e2e8f0 !important;
        box-shadow: 0 0 0 2px rgba(226,232,240,0.16) !important;
    }

    /* ─── DATAFRAME ──────────────────────────────────────── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    .stDataFrame thead tr th {
        background: rgba(255,255,255,0.04) !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .stDataFrame tbody tr:nth-child(even) td { background: #0c1624 !important; }
    .stDataFrame tbody tr:hover td { background: #111e30 !important; }

    /* ─── METRICS ────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: rgba(16, 22, 36, 0.56) !important;
        border: 1px solid rgba(180,190,210,0.14) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        backdrop-filter: blur(16px) saturate(150%);
        -webkit-backdrop-filter: blur(16px) saturate(150%);
    }
    [data-testid="stMetricLabel"] { color: #aab4c6 !important; font-size: 0.78rem !important; font-weight: 500 !important; }
    [data-testid="stMetricValue"] { color: #e8eaf0 !important; font-size: 1.4rem !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] { font-size: 0.82rem !important; font-weight: 600 !important; }

    /* ─── ALERTS/INFO/WARNINGS ───────────────────────────── */
    .stAlert { border-radius: 10px !important; border-left: 3px solid !important; }

    /* ─── DIVIDER ────────────────────────────────────────── */
    hr { border-color: #27314a !important; }

    /* ─── HEADERS ────────────────────────────────────────── */
    h1 { color: #f4f7ff !important; font-weight: 700 !important; }
    h2 { color: #e1e8f4 !important; font-weight: 600 !important; }
    h3 { color: #c6d0e0 !important; font-weight: 600 !important; }

    /* ─── PAGE HEADER BANNER ─────────────────────────────── */
    .page-header {
        background: rgba(18, 24, 40, 0.52);
        border: 1px solid rgba(180,190,210,0.14);
        border-radius: 16px;
        padding: 18px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 16px;
        backdrop-filter: blur(18px) saturate(160%);
        -webkit-backdrop-filter: blur(18px) saturate(160%);
    }
    .page-header .ph-icon { font-size: 2rem; }
    .page-header .ph-title { font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin: 0; line-height: 1.2; }
    .page-header .ph-sub { font-size: 0.85rem; color: #aab4c6; margin: 3px 0 0 0; }

    @media (max-width: 768px) {
        .page-header {
            padding: 14px 16px;
            gap: 12px;
            flex-direction: column;
            align-items: flex-start;
        }
        .page-header .ph-title { font-size: 1.15rem; }
        .page-header .ph-sub { font-size: 0.78rem; }
        .glass-panel, .metric-card, .index-card {
            border-radius: 14px;
        }
        .metric-card, .index-card {
            padding: 14px;
        }
        .index-card .ix-value { font-size: 1.65rem; }
        .stButton > button {
            min-height: 42px !important;
            font-size: 0.82rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 7px 10px !important;
            font-size: 0.76rem !important;
        }
        [data-testid="stMetric"] {
            padding: 12px !important;
        }
        section[data-testid="stSidebar"] {
            width: 100% !important;
        }
    }

    /* ─── RESPONSIVE ─────────────────────────────────────── */
    @media (max-width: 768px) {
        .stMarkdown h1 { font-size: 22px !important; }
        .stMarkdown h2 { font-size: 18px !important; }
        .stMarkdown h3 { font-size: 16px !important; }
        section[data-testid="stSidebar"] { display: none; }
        .stDataFrame { font-size: 11px !important; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'warning_shown' not in st.session_state:
        st.warning("⚠️ **Data Not Saved:** Portfolio and trades are temporary. Export to CSV before closing!")
        st.session_state.warning_shown = True

    if 'portfolio' not in st.session_state:
        st.session_state.portfolio =[]
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist =[]
    if 'trade_journal' not in st.session_state:
        st.session_state.trade_journal =[]
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'db' not in st.session_state:
        st.session_state.db = None
    if 'db_conn_string' not in st.session_state:
        st.session_state.db_conn_string = None
    if 'db_type' not in st.session_state:
        st.session_state.db_type = None
    if 'exchange' not in st.session_state:
        st.session_state.exchange = "NSE"

    st.markdown("""
    <div style="background:linear-gradient(135deg,#061020 0%,#0e1e38 50%,#061020 100%);
         border:1px solid #28314a;border-radius:18px;padding:20px 26px;margin-bottom:18px;
         display:flex;align-items:center;gap:16px;">
        <div style="font-size:2.8rem;line-height:1;">📈</div>
        <div>
            <div style="font-size:1.75rem;font-weight:800;color:#f4f7ff;letter-spacing:-0.02em;">
                Fundfolio <span style="color:#c084fc;">Pro</span>
            </div>
            <div style="font-size:0.82rem;color:#93a0b7;margin-top:2px;">
                Indian Stock Market Platform &nbsp;•&nbsp; NSE/BSE &nbsp;•&nbsp; Real-time Data
            </div>
        </div>
        <div style="margin-left:auto;text-align:right;">
            <div style="font-size:0.75rem;color:#8c98ad;">Market Status</div>
            <div style="font-size:0.95rem;font-weight:700;color:{'#22c55e' if is_market_open() else '#fb7185'};">
                {'🟢 OPEN' if is_market_open() else '🔴 CLOSED'}
            </div>
        </div>
    </div>
    """.replace("{'#22c55e' if is_market_open() else '#fb7185'}", "#22c55e" if is_market_open() else "#fb7185")
        .replace("{'🟢 OPEN' if is_market_open() else '🔴 CLOSED'}", "🟢 OPEN" if is_market_open() else "🔴 CLOSED"),
    unsafe_allow_html=True)

    # Initialize and run Synchronous Alert Check
    alert_manager = AlertManager()
    alert_manager.check_alerts()

    def get_db():
        if st.session_state.db is None and st.session_state.db_conn_string and st.session_state.db_type:
            try:
                st.session_state.db = get_database(st.session_state.db_type, st.session_state.db_conn_string)
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                return None
        return st.session_state.db

    def auto_save():
        db = get_db()
        if db and st.session_state.user_id:
            try:
                db.save_portfolio(st.session_state.user_id, st.session_state.portfolio)
                logger.info("Auto-save completed")
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:12px 0 8px 0;">
            <div style="font-size:1.5rem;font-weight:800;color:#c084fc;letter-spacing:-0.02em;">📈 Fundfolio</div>
            <div style="font-size:0.7rem;color:#8c98ad;margin-top:2px;">Indian Market Platform</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        st.markdown("<div style='font-size:0.7rem;color:#3a5070;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;'>🏠 Overview</div>", unsafe_allow_html=True)
        overview_pages = ["🏠 Dashboard"]

        st.markdown("<div style='font-size:0.7rem;color:#3a5070;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin:10px 0 4px 0;'>📅 Market Sessions</div>", unsafe_allow_html=True)
        session_pages = [
            "🌅 Pre-Market Prep",
            "⚡ Pre-Open Session (9–9:15 AM)",
            "🔴 Live Market (After 9:15 AM)",
        ]

        st.markdown("<div style='font-size:0.7rem;color:#3a5070;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin:10px 0 4px 0;'>🔬 Analysis</div>", unsafe_allow_html=True)
        analysis_pages = [
            "📊 Fundamental Analysis",
            "📈 Technical Analysis",
            "🧮 Options Analyzer",
            "🔍 Stock Screener",
            "🔀 Multi-Stock Comparison",
            "🎯 Stock Selection Engine",
        ]

        st.markdown("<div style='font-size:0.7rem;color:#3a5070;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin:10px 0 4px 0;'>💼 Portfolio & Trading</div>", unsafe_allow_html=True)
        trading_pages = [
            "💼 Portfolio Manager",
            "📝 Trade Journal",
            "🎯 Position Sizer",
            "📱 Quick Trade Setup",
            "📈 Strategy Backtester",
        ]

        st.markdown("<div style='font-size:0.7rem;color:#3a5070;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin:10px 0 4px 0;'>💰 Finance & Tax</div>", unsafe_allow_html=True)
        finance_pages = [
            "💰 Tax Calculator",
            "📉 Tax P&L Report",
            "📉 Risk Analytics",
        ]

        st.markdown("<div style='font-size:0.7rem;color:#3a5070;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin:10px 0 4px 0;'>🔧 Tools</div>", unsafe_allow_html=True)
        tools_pages = [
            "🔔 Alerts",
            "📅 Corporate Actions",
            "📈 Futures & Options",
            "🇮🇳 India Market Hub",
            "❓ FAQ",
            "📚 Education Center",
        ]

        all_pages = overview_pages + session_pages + analysis_pages + trading_pages + finance_pages + tools_pages
        if "nav_page" not in st.session_state or st.session_state.nav_page not in all_pages:
            st.session_state.nav_page = all_pages[0]
        page = st.radio("Navigate", all_pages, index=all_pages.index(st.session_state.nav_page), label_visibility="collapsed")
        st.session_state.nav_page = page
        st.divider()

        st.header("📈 Exchange")
        exchange_choice = st.radio("Select Exchange", ["NSE", "BSE"], index=0)
        st.session_state.exchange = exchange_choice

        st.header("💾 Database")
        db_type_choice = st.selectbox("Storage Type",["None (Session Only)", "PostgreSQL", "MongoDB"])
        user_id = st.text_input("User ID", value=st.session_state.user_id or "", help="Use different IDs to separate data")
        st.session_state.user_id = user_id if user_id else None

        if db_type_choice != "None (Session Only)":
            conn_str = st.text_input("Connection String", 
                                    value=st.session_state.db_conn_string or "",
                                    type="password")
            if st.button("Connect to Database"):
                try:
                    if db_type_choice == "PostgreSQL":
                        if not SQLALCHEMY_AVAILABLE:
                            st.error("SQLAlchemy not installed. Run: pip install sqlalchemy psycopg2-binary")
                        else:
                            st.session_state.db_conn_string = conn_str or None
                            st.session_state.db_type = db_type_choice.lower()
                            st.session_state.db = None
                            db = get_db()
                            if db:
                                st.success(f"Connected to {db_type_choice} as user {user_id}")
                                portfolio = db.load_portfolio(user_id)
                                if portfolio:
                                    st.session_state.portfolio = portfolio
                                    st.info(f"Loaded {len(portfolio)} positions")
                                trades = db.load_trades(user_id)
                                if trades:
                                    st.session_state.trade_journal = trades
                                    st.info(f"Loaded {len(trades)} trades")
                            else:
                                st.error("Failed to connect")
                    else:
                        if not MONGODB_AVAILABLE:
                            st.error("pymongo not installed. Run: pip install pymongo")
                        else:
                            st.session_state.db_conn_string = conn_str or None
                            st.session_state.db_type = db_type_choice.lower()
                            st.session_state.db = None
                            db = get_db()
                            if db:
                                st.success(f"Connected to {db_type_choice} as user {user_id}")
                                portfolio = db.load_portfolio(user_id)
                                if portfolio:
                                    st.session_state.portfolio = portfolio
                                    st.info(f"Loaded {len(portfolio)} positions")
                                trades = db.load_trades(user_id)
                                if trades:
                                    st.session_state.trade_journal = trades
                                    st.info(f"Loaded {len(trades)} trades")
                            else:
                                st.error("Failed to connect")
                except Exception as e:
                    st.error(f"Database connection failed: {e}")
            if st.session_state.db_conn_string:
                if st.button("Sync to Database", use_container_width=True):
                    db = get_db()
                    if db:
                        if db.save_portfolio(user_id, st.session_state.portfolio):
                            st.success("Portfolio saved")
                        else:
                            st.error("Failed to save portfolio")
                if st.button("Load from Database", use_container_width=True):
                    db = get_db()
                    if db:
                        portfolio = db.load_portfolio(user_id)
                        if portfolio:
                            st.session_state.portfolio = portfolio
                            st.success(f"Loaded {len(portfolio)} positions")
                        trades = db.load_trades(user_id)
                        if trades:
                            st.session_state.trade_journal = trades
                            st.success(f"Loaded {len(trades)} trades")
        else:
            st.session_state.db_conn_string = None
            st.session_state.db_type = None
            st.session_state.db = None

        st.markdown("### 📡 Market Clock")
        now_ist = get_ist_time()
        ist_time_str = now_ist.strftime("%I:%M %p IST")
        ist_date_str = now_ist.strftime("%a, %d %b %Y")
        market_color = "#00d084" if is_market_open() else "#ff6060"
        market_label = "🟢 Market Open" if is_market_open() else "🔴 Market Closed"
        st.markdown(f"""
        <div style="background:#0a1520;border:1px solid #1a2a40;border-radius:10px;padding:12px;text-align:center;">
            <div style="color:#4a6080;font-size:0.72rem;">{ist_date_str}</div>
            <div style="color:#00d4ff;font-size:1.3rem;font-weight:700;">{ist_time_str}</div>
            <div style="color:{market_color};font-size:0.82rem;font-weight:600;margin-top:4px;">{market_label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📊 Session Stats")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Portfolio", len(st.session_state.portfolio), label_visibility="visible")
        with c2:
            st.metric("Trades", len(st.session_state.trade_journal), label_visibility="visible")
        st.metric("Watchlist", len(st.session_state.watchlist), label_visibility="visible")
        st.divider()
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.portfolio = []
            st.session_state.watchlist =[]
            st.session_state.trade_journal =[]
            auto_save()
            st.success("All data cleared!")

    # ----------------------------------------------------------------------
    # PAGE FUNCTIONS
    # ----------------------------------------------------------------------
    def render_glass_header(icon: str, title: str, subtitle: str):
        st.markdown(
            f"""
            <div class="page-header glass-panel">
                <div class="ph-icon">{icon}</div>
                <div>
                    <div class="ph-title">{title}</div>
                    <div class="ph-sub">{subtitle}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def navigate_to(page_name: str):
        st.session_state.nav_page = page_name
        st.rerun()

    def show_dashboard():
        render_glass_header("🏠", "Market Dashboard", "Overview of market indices, movers, and quick actions")
        now_ist = get_ist_time()
        market_open = is_market_open()
        day_name = now_ist.strftime("%A")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""
                <div class="glass-panel" style="padding:16px;min-height:92px;">
                    <div style="font-size:0.8rem;color:#aab4c6;">Today</div>
                    <div style="font-size:1.2rem;font-weight:700;color:#f8fafc;">{day_name}</div>
                    <div style="font-size:0.84rem;color:#aab4c6;">{now_ist.strftime('%d %b %Y')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="glass-panel" style="padding:16px;min-height:92px;">
                    <div style="font-size:0.8rem;color:#aab4c6;">Market Status</div>
                    <div style="font-size:1.2rem;font-weight:700;color:{'#22c55e' if market_open else '#fb7185'};">{'Open' if market_open else 'Closed'}</div>
                    <div style="font-size:0.84rem;color:#aab4c6;">IST {now_ist.strftime('%I:%M %p')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.success("✅ Data refreshed!")
                st.rerun()
        st.divider()
        indices = {'NIFTY 50': '^NSEI', 'SENSEX': '^BSESN', 'NIFTY BANK': '^NSEBANK'}
        cols = st.columns(3)
        for i, (name, ticker) in enumerate(indices.items()):
            current, change, pct = get_cached_index_data(ticker)
            if current is not None:
                if change is not None and change > 0:
                    arrow, color = "▲", "#22c55e"
                elif change is not None and change < 0:
                    arrow, color = "▼", "#fb7185"
                else:
                    arrow, color = "•", "#888"
                change_display = f"{arrow} {change:+.0f} ({pct:+.2f}%)" if change is not None else "No change"
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: #aab4c6;">{name}</div>
                        <div style="color: #f8fafc; font-size: 2rem; font-weight: 600;">{current:,.0f}</div>
                        <div style="color: {color};">{change_display}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                with cols[i]:
                    st.warning(f"{name} data unavailable")
        st.divider()
        gainers, losers = get_cached_top_movers(limit=5)
        live_cols = st.columns([3,1])
        with live_cols[0]:
            st.subheader("📊 Live Movers Panel")
        with live_cols[1]:
            if st.button("🔄 Refresh Movers", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔥 Top Gainers")
            if gainers:
                df = pd.DataFrame(gainers)
                df['Price'] = format_indian_series(df['Price'], is_share_price=True)
                st.dataframe(df, width="stretch", hide_index=True)
            else:
                st.info("No gainers data")
        with col2:
            st.subheader("📉 Top Losers")
            if losers:
                df = pd.DataFrame(losers)
                df['Price'] = format_indian_series(df['Price'], is_share_price=True)
                st.dataframe(df, width="stretch", hide_index=True)
            else:
                st.info("No losers data")
        st.divider()
        st.subheader("⚡ One-Click Trading Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("📈 Analyze Stock", use_container_width=True):
                navigate_to("📈 Technical Analysis")
        with c2:
            if st.button("💼 Add to Portfolio", use_container_width=True):
                navigate_to("💼 Portfolio Manager")
        with c3:
            if st.button("🎯 Calculate Position", use_container_width=True):
                navigate_to("🎯 Position Sizer")
        with c4:
            if st.button("📝 Log Trade", use_container_width=True):
                navigate_to("📝 Trade Journal")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            if st.button("🧮 Options", use_container_width=True):
                navigate_to("🧮 Options Analyzer")
        with c6:
            if st.button("🇮🇳 Market Hub", use_container_width=True):
                navigate_to("🇮🇳 India Market Hub")
        with c7:
            if st.button("🔔 Alerts", use_container_width=True):
                navigate_to("🔔 Alerts")
        with c8:
            if st.button("📚 Learn", use_container_width=True):
                navigate_to("📚 Education Center")

        st.divider()
        st.subheader("🗓️ Today’s Usage")
        checklist = [
            ("Pre-market", "Check market status, news, and the pre-market prep page."),
            ("Open", "Use live market and technical analysis for entries."),
            ("Risk", "Run position sizing before every trade."),
            ("After trade", "Log execution and update the journal."),
        ]
        for title, desc in checklist:
            st.markdown(f"- **{title}:** {desc}")

        st.divider()
        st.subheader("✨ Feature Hub")
        feature_cards = [
            ("📊 Fundamental Analysis", "Annual reports, live fundamentals, ratios"),
            ("📈 Technical Analysis", "Charts, indicators, signals, patterns"),
            ("🧮 Options Analyzer", "Black-Scholes, Greeks, payoff charts"),
            ("📈 Futures & Options", "Derivatives basics and quick compare"),
            ("🇮🇳 India Market Hub", "NSE/SEBI shortcuts and market usage"),
            ("🔔 Alerts", "Price alerts and trigger checks"),
            ("📚 Education Center", "Glossary, tax guide, psychology"),
        ]
        for row in [feature_cards[i:i+3] for i in range(0, len(feature_cards), 3)]:
            cols = st.columns(3)
            for idx, (title, desc) in enumerate(row):
                with cols[idx]:
                    st.markdown(
                        f"""
                        <div class="glass-panel" style="padding:16px 16px 14px 16px;min-height:108px;">
                            <div style="font-size:1rem;font-weight:700;color:#f4f7ff;margin-bottom:6px;">{title}</div>
                            <div style="font-size:0.84rem;line-height:1.45;color:#aab4c6;">{desc}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    def show_fundamental_analysis():
        render_glass_header("📊", "Fundamental Analysis", "Annual reports, live fundamentals, and ratios")
        tabs = st.tabs(["📄 Annual Report Analysis", "📈 Live Fundamentals", "🔍 Financial Ratios"])
        with tabs[0]:
            st.subheader("📄 Upload Annual Reports")
            uploaded = st.file_uploader("Upload PDF Annual Reports", type=['pdf'], accept_multiple_files=True)
            if uploaded:
                MAX_SIZE = CONFIG.max_pdf_size_mb * 1024 * 1024
                oversized =[f for f in uploaded if f.size > MAX_SIZE]
                if oversized:
                    st.error(f"❌ These files are too large (max {CONFIG.max_pdf_size_mb}MB): {', '.join([f.name for f in oversized])}")
                    uploaded =[f for f in uploaded if f.size <= MAX_SIZE]
                    if not uploaded:
                        return
                st.success(f"✅ {len(uploaded)} file(s) uploaded")
                col1, col2 = st.columns(2)
                with col1:
                    max_pages = st.slider("Max Pages to Scan", 50, 300, 120)
                with col2:
                    stock_price_input = st.number_input("Current Stock Price (₹)", 0.0, step=10.0)
                if st.button("🔍 Analyze Reports", type="primary"):
                    temp_dir = tempfile.mkdtemp()
                    paths, companies = [],[]
                    for f in uploaded:
                        path = os.path.join(temp_dir, f.name)
                        with open(path, 'wb') as out:
                            out.write(f.read())
                        paths.append(path)
                        comp, _ = extract_company_year_from_pdf(path)
                        companies.append(comp)
                    price_map = {c: stock_price_input for c in set(companies)}
                    prog = st.progress(0)
                    status = st.empty()
                    def cb(cur, total, msg):
                        prog.progress((cur+1)/total, text=msg)
                        status.text(msg)
                    result, latest, raw = generate_analysis_from_pdfs(paths, price_map, max_pages=max_pages, progress_callback=cb)
                    prog.progress(1.0, text="Complete!")
                    status.empty()
                    try:
                        shutil.rmtree(temp_dir)
                    except:
                        pass
                    gc.collect()
                    if result.empty:
                        st.error("❌ No financial data could be extracted.")
                        st.info("Ensure PDFs contain text‑based financial statements.")
                    else:
                        st.success(f"✅ Analysis complete!")
                        display = result.copy()
                        for col in['Revenue','Net Profit','Equity','Total Assets','Current Assets',
                                    'Current Liabilities','Total Debt','Operating Cash Flow','Capex','Cash']:
                            if col in display.columns:
                                display[col] = format_indian_series(display[col], is_share_price=False)
                        st.markdown("### 📊 Extracted Results")
                        st.dataframe(display, width="stretch")
                        csv = result.to_csv().encode('utf-8')
                        st.download_button("📥 Download Raw CSV", csv, "analysis_raw.csv")
            else:
                st.info("👆 Upload annual report PDFs to begin analysis")
        with tabs[1]:
            st.subheader("📈 Live Fundamental Data")
            ticker_raw = st.text_input("Enter Stock Symbol", placeholder="RELIANCE.NS")
            ticker = normalize_ticker(ticker_raw, st.session_state.exchange)
            if ticker:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        cp = info.get('currentPrice', 0)
                        st.metric("Current Price", format_indian_number(cp, is_share_price=True))
                        pe = info.get('trailingPE')
                        st.metric("P/E Ratio", f"{pe:.2f}" if pe is not None else "N/A")
                        pb = info.get('priceToBook')
                        st.metric("P/B Ratio", f"{pb:.2f}" if pb is not None else "N/A")
                    with col2:
                        mc = info.get('marketCap')
                        st.metric("Market Cap", format_indian_number(mc, is_share_price=False) if mc is not None else "N/A")
                        div = info.get('dividendYield')
                        st.metric("Dividend Yield", f"{div*100:.2f}%" if div is not None else "N/A")
                        eps = info.get('trailingEps')
                        st.metric("EPS", format_indian_number(eps, is_share_price=True) if eps is not None else "N/A")
                    with col3:
                        roe_raw = info.get('returnOnEquity')
                        st.metric("ROE", f"{roe_raw*100:.2f}%" if roe_raw is not None else "N/A")
                        de = info.get('debtToEquity')
                        de_val = de/100 if de is not None else None
                        st.metric("Debt/Equity", f"{de_val:.2f}" if de_val is not None else "N/A")
                        high52 = info.get('fiftyTwoWeekHigh')
                        st.metric("52W High", format_indian_number(high52, is_share_price=True) if high52 is not None else "N/A")
                except Exception as e:
                    st.error(f"Error: {e}")
        with tabs[2]:
            st.subheader("🔍 Financial Ratio Calculator")
            with st.form("ratio_form"):
                col1, col2 = st.columns(2)
                with col1:
                    revenue = st.number_input("Revenue (₹ Cr)", 0.0, step=100.0)
                    net_profit = st.number_input("Net Profit (₹ Cr)", 0.0, step=10.0)
                    equity = st.number_input("Equity (₹ Cr)", 0.0, step=10.0)
                    total_assets = st.number_input("Total Assets (₹ Cr)", 0.0, step=100.0)
                with col2:
                    total_debt = st.number_input("Total Debt (₹ Cr)", 0.0, step=10.0)
                    current_assets = st.number_input("Current Assets (₹ Cr)", 0.0, step=10.0)
                    current_liabilities = st.number_input("Current Liabilities (₹ Cr)", 0.0, step=10.0)
                    stock_price = st.number_input("Stock Price (₹)", 0.0, step=10.0)
                submitted = st.form_submit_button("📊 Calculate Ratios")
                if submitted:
                    rev_val = revenue * 1e7
                    net_profit_val = net_profit * 1e7
                    eq_val = equity * 1e7
                    ta_val = total_assets * 1e7
                    td_val = total_debt * 1e7
                    ca_val = current_assets * 1e7
                    cl_val = current_liabilities * 1e7
                    ratios = {}
                    if rev_val > 0 and net_profit_val > 0:
                        ratios['Net Profit Margin'] = (net_profit_val/rev_val)*100
                    if eq_val > 0 and net_profit_val > 0:
                        ratios['ROE'] = (net_profit_val/eq_val)*100
                    if ta_val > 0 and net_profit_val > 0:
                        ratios['ROA'] = (net_profit_val/ta_val)*100
                    if eq_val > 0 and td_val >= 0:
                        ratios['Debt to Equity'] = td_val/eq_val
                    if cl_val > 0 and ca_val > 0:
                        ratios['Current Ratio'] = ca_val/cl_val
                    st.markdown("### 📊 Calculated Ratios")
                    cols = st.columns(3)
                    for idx, (name, val) in enumerate(ratios.items()):
                        with cols[idx % 3]:
                            st.metric(name, f"{val:.2f}{'%' if 'Margin' in name or 'RO' in name else ''}")

    def show_options_analyzer():
        render_glass_header("🧮", "Options Analyzer", "Black-Scholes pricing, Greeks, and strategy payoffs")
        st.caption("Black-Scholes pricing, Greeks, expiry payoff, and strategy templates for NSE-style analysis.")

        strategy_notes = {
            "Long Call": "Directional bullish bet with limited downside.",
            "Long Put": "Directional bearish bet with defined risk.",
            "Short Call": "Income trade with open-ended upside risk.",
            "Short Put": "Bullish income trade with downside exposure.",
            "Long Straddle": "Volatility bet using both call and put.",
            "Short Straddle": "Range-bound view with large tail risk.",
            "Covered Call": "Stock holding plus short call for premium income.",
            "Protective Put": "Stock holding plus put for downside protection.",
        }
        strategy_presets = list(strategy_notes.keys())

        with st.form("options_analyzer_form"):
            c1, c2 = st.columns(2)
            with c1:
                spot = st.number_input("Underlying Price (S)", min_value=0.0, value=22500.0, step=50.0)
                strike = st.number_input("Strike Price (K)", min_value=0.0, value=22600.0, step=50.0)
                stock_entry = st.number_input("Stock Entry Price", min_value=0.0, value=22500.0, step=50.0)
                strategy = st.selectbox("Strategy Preset", strategy_presets, index=0)
            with c2:
                days = st.number_input("Days to Expiry", min_value=0.0, value=30.0, step=1.0)
                risk_free = st.number_input("Risk-free Rate (%)", min_value=0.0, value=6.5, step=0.1)
                volatility = st.number_input("Annual Volatility (%)", min_value=0.0, value=18.0, step=0.5)
                observed_call = st.number_input("Observed Call Premium (optional)", min_value=0.0, value=0.0, step=1.0)
                observed_put = st.number_input("Observed Put Premium (optional)", min_value=0.0, value=0.0, step=1.0)
            submitted = st.form_submit_button("Calculate Options")

        if submitted:
            T_years = days / 365.0 if days > 0 else 0.0
            r = risk_free / 100.0
            sigma = volatility / 100.0
            call_price = black_scholes(spot, strike, T_years, r, sigma, 'call')
            put_price = black_scholes(spot, strike, T_years, r, sigma, 'put')
            call_used = observed_call if observed_call > 0 else call_price
            put_used = observed_put if observed_put > 0 else put_price

            call_greeks = black_scholes_greeks(spot, strike, T_years, r, sigma, 'call')
            put_greeks = black_scholes_greeks(spot, strike, T_years, r, sigma, 'put')

            st.subheader("Fair Value")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Call Premium", f"₹{call_price:,.2f}", delta=f"Market {call_used:,.2f}" if observed_call > 0 else None)
            with c2:
                st.metric("Put Premium", f"₹{put_price:,.2f}", delta=f"Market {put_used:,.2f}" if observed_put > 0 else None)

            g1, g2, g3, g4 = st.columns(4)
            with g1:
                st.metric("Call Delta", f"{call_greeks['delta']:.4f}")
            with g2:
                st.metric("Call Gamma", f"{call_greeks['gamma']:.6f}")
            with g3:
                st.metric("Call Theta/day", f"{call_greeks['theta']:.4f}")
            with g4:
                st.metric("Call Vega", f"{call_greeks['vega']:.4f}")

            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric("Put Delta", f"{put_greeks['delta']:.4f}")
            with p2:
                st.metric("Put Gamma", f"{put_greeks['gamma']:.6f}")
            with p3:
                st.metric("Put Theta/day", f"{put_greeks['theta']:.4f}")
            with p4:
                st.metric("Put Vega", f"{put_greeks['vega']:.4f}")

            st.subheader("Option Breakdown")
            tab_call, tab_put, tab_strategy = st.tabs(["Call", "Put", "Strategy"])
            with tab_call:
                summary = analyze_option(spot, strike, call_used, T_years, 'call')
                st.write(f"Intrinsic Value: ₹{summary['intrinsic_value']:,.2f}")
                st.write(f"Break-even: ₹{summary['break_even']:,.2f}")
                st.write(f"Status: {summary['status']}")
                st.write(f"Buyer: max loss = ₹{summary['buyer_max_loss']:,.2f}; {summary['buyer_profit_note']}")
                st.write(f"Seller: max profit = ₹{summary['seller_max_profit']:,.2f}; {summary['seller_risk_note']}")
            with tab_put:
                summary = analyze_option(spot, strike, put_used, T_years, 'put')
                st.write(f"Intrinsic Value: ₹{summary['intrinsic_value']:,.2f}")
                st.write(f"Break-even: ₹{summary['break_even']:,.2f}")
                st.write(f"Status: {summary['status']}")
                st.write(f"Buyer: max loss = ₹{summary['buyer_max_loss']:,.2f}; {summary['buyer_profit_note']}")
                st.write(f"Seller: max profit = ₹{summary['seller_max_profit']:,.2f}; {summary['seller_risk_note']}")
            with tab_strategy:
                st.info(strategy_notes[strategy])

                def strategy_payoff(px: float) -> float:
                    legs = {
                        "Long Call": option_payoff_at_expiry(px, strike, call_used, 'call', 'long'),
                        "Long Put": option_payoff_at_expiry(px, strike, put_used, 'put', 'long'),
                        "Short Call": option_payoff_at_expiry(px, strike, call_used, 'call', 'short'),
                        "Short Put": option_payoff_at_expiry(px, strike, put_used, 'put', 'short'),
                        "Long Straddle": option_payoff_at_expiry(px, strike, call_used, 'call', 'long') + option_payoff_at_expiry(px, strike, put_used, 'put', 'long'),
                        "Short Straddle": option_payoff_at_expiry(px, strike, call_used, 'call', 'short') + option_payoff_at_expiry(px, strike, put_used, 'put', 'short'),
                        "Covered Call": (px - stock_entry) + option_payoff_at_expiry(px, strike, call_used, 'call', 'short'),
                        "Protective Put": (px - stock_entry) + option_payoff_at_expiry(px, strike, put_used, 'put', 'long'),
                    }
                    return legs[strategy]

                payoff_df = build_payoff_frame(spot, strike, call_used, put_used, 'call')
                payoff_df['Strategy'] = payoff_df['Spot'].map(strategy_payoff)
                be = strategy_breakeven(strategy, strike, call_used, put_used, stock_entry)
                if isinstance(be, tuple):
                    st.metric("Breakeven Range", f"₹{be[0]:,.2f} to ₹{be[1]:,.2f}")
                else:
                    st.metric("Breakeven", f"₹{be:,.2f}")

                if PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=payoff_df['Spot'], y=payoff_df['Strategy'], mode='lines', name=strategy))
                    fig.add_hline(y=0, line_dash='dash', line_color='#888')
                    fig.update_layout(title=f"{strategy} Payoff at Expiry", xaxis_title='Spot Price', yaxis_title='P&L', template='plotly_dark', height=420)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart(payoff_df.set_index('Spot')['Strategy'])

                st.markdown("#### Payoff Table")
                table = payoff_df[['Spot', 'Strategy']].copy()
                table['Spot'] = table['Spot'].round(2)
                table['Strategy'] = table['Strategy'].round(2)
                st.dataframe(table, use_container_width=True, hide_index=True)

            with st.expander("Key Concepts"):
                st.markdown(
                    """
                    - Strike price: the exercise price.
                    - Premium: the option price.
                    - Expiry: when the contract ends.
                    - Intrinsic value: immediate exercise value.
                    - Greeks: sensitivity measures for price, time, and volatility.
                    - Strategy presets: quick templates for common expiry payoffs.
                    """
                )

    def show_technical_analysis():
        render_glass_header("📈", "Technical Analysis", "Price charts, indicators, patterns, and signals")
        col1, col2 = st.columns([2,1])
        with col1:
            ticker_raw = st.text_input("Enter Stock Symbol", value="RELIANCE.NS", placeholder="TCS.NS, INFY.NS")
            ticker = normalize_ticker(ticker_raw, st.session_state.exchange)
        with col2:
            period = st.selectbox("Time Period",['1mo','3mo','6mo','1y','2y','5y'])
        if ticker:
            with st.spinner("Fetching data..."):
                df = compute_technical_indicators_cached(ticker, period)
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) >= 2 else latest
                    cols = st.columns(4)
                    with cols[0]:
                        change = latest['Close'] - prev['Close']
                        pct = (change/prev['Close'])*100 if prev['Close']!=0 else 0
                        color = "#00d084" if change>0 else "#ff4444" if change<0 else "#888"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="color:#888;">Current Price</div>
                            <div style="color:{color}; font-size:2rem; font-weight:600;">{format_indian_number(latest['Close'], is_share_price=True)}</div>
                            <div style="color:{color};">{change:+.2f} ({pct:+.2f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        st.metric("Volume", f"{latest['Volume']:,.0f}")
                        if 'Volume_SMA_20' in df.columns:
                            st.metric("Avg Volume", f"{df['Volume_SMA_20'].iloc[-1]:,.0f}")
                    with cols[2]:
                        if 'RSI' in df.columns:
                            st.metric("RSI", f"{latest['RSI']:.2f}")
                            rsi_sig = "Oversold" if latest['RSI']<30 else "Overbought" if latest['RSI']>70 else "Neutral"
                            st.info(f"Signal: {rsi_sig}")
                        else:
                            st.metric("RSI", "N/A")
                    with cols[3]:
                        if 'ATR' in df.columns:
                            st.metric("ATR", format_indian_number(latest['ATR'], is_share_price=True))
                            st.metric("Volatility", f"{(latest['ATR']/latest['Close']*100):.2f}%")
                    tabs = st.tabs(["📊 Price Chart","📈 Indicators","🎯 Signals","📋 Data Table","📈 Candlestick Patterns"])
                    with tabs[0]:
                        if PLOTLY_AVAILABLE:
                            fig = go.Figure()
                            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
                            for ma in ['SMA_20','SMA_50','SMA_200']:
                                if ma in df.columns:
                                    fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(width=1)))
                            fig.update_layout(title=f"{ticker} Price Chart", yaxis_title='Price (₹)', template='plotly_dark', height=600)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.line_chart(df['Close'])
                    with tabs[1]:
                        if PLOTLY_AVAILABLE and 'RSI' in df.columns and 'MACD' in df.columns:
                            fig = make_subplots(rows=3, cols=1, subplot_titles=('RSI','MACD','Stochastic'))
                            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=1, col=1)
                            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
                            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=2, col=1)
                            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'), row=2, col=1)
                            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram'), row=2, col=1)
                            if 'Stochastic_%K' in df.columns:
                                fig.add_trace(go.Scatter(x=df.index, y=df['Stochastic_%K'], name='%K'), row=3, col=1)
                                fig.add_trace(go.Scatter(x=df.index, y=df['Stochastic_%D'], name='%D'), row=3, col=1)
                            fig.update_layout(height=900, template='plotly_dark', showlegend=True)
                            st.plotly_chart(fig, use_container_width=True)
                    with tabs[2]:
                        st.subheader("🎯 Trading Signals")
                        signals = []
                        if 'SMA_50' in df.columns:
                            if latest['Close'] > latest['SMA_50']:
                                signals.append(("🟢 BULLISH", "Price above 50 SMA", "Uptrend"))
                            else:
                                signals.append(("🔴 BEARISH", "Price below 50 SMA", "Downtrend"))
                        if 'RSI' in df.columns:
                            if latest['RSI'] < 30:
                                signals.append(("🟢 BUY", "RSI Oversold", "Strong"))
                            elif latest['RSI'] > 70:
                                signals.append(("🔴 SELL", "RSI Overbought", "Strong"))
                            else:
                                signals.append(("🟡 NEUTRAL", "RSI Neutral", "Weak"))
                        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                            if latest['MACD'] > latest['MACD_Signal']:
                                signals.append(("🟢 BULLISH", "MACD above Signal", "Medium"))
                            else:
                                signals.append(("🔴 BEARISH", "MACD below Signal", "Medium"))
                        for sig, desc, strength in signals:
                            c1, c2, c3 = st.columns([1,3,1])
                            with c1:
                                st.markdown(f"**{sig}**")
                            with c2:
                                st.text(desc)
                            with c3:
                                st.text(strength)
                        if 'BB_Lower' in df.columns and 'BB_Upper' in df.columns:
                            st.divider()
                            st.subheader("📊 Support & Resistance")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.metric("Support 1", format_indian_number(latest['BB_Lower'], is_share_price=True))
                                st.metric("Support 2", format_indian_number(df['Low'].tail(20).min(), is_share_price=True))
                            with c2:
                                st.metric("Resistance 1", format_indian_number(latest['BB_Upper'], is_share_price=True))
                                st.metric("Resistance 2", format_indian_number(df['High'].tail(20).max(), is_share_price=True))
                    with tabs[3]:
                        disp = df.tail(50).copy()
                        for c in ['Open','High','Low','Close']:
                            if c in disp.columns:
                                disp[c] = format_indian_series(disp[c], is_share_price=True)
                        st.dataframe(disp, width="stretch")
                    with tabs[4]:
                        st.subheader("🕯️ Candlestick Pattern Detection")
                        if len(df) >= 3:
                            patterns = PatternScanner.scan(df, last_n=20)
                            if patterns:
                                st.markdown("### Detected Patterns")
                                for p in patterns:
                                    date = df.index[p.index].strftime('%Y-%m-%d')
                                    if p.signal == PatternSignal.BULLISH:
                                        with st.success(f"**{p.name}** ({p.strength}) - {p.description}"):
                                            st.write(f"📅 {date} | Signal: {p.signal.value}")
                                    elif p.signal == PatternSignal.BEARISH:
                                        with st.error(f"**{p.name}** ({p.strength}) - {p.description}"):
                                            st.write(f"📅 {date} | Signal: {p.signal.value}")
                                    else:
                                        with st.info(f"**{p.name}** ({p.strength}) - {p.description}"):
                                            st.write(f"📅 {date} | Signal: {p.signal.value}")
                                bullish =[p for p in patterns if p.signal == PatternSignal.BULLISH]
                                bearish =[p for p in patterns if p.signal == PatternSignal.BEARISH]
                                neutral =[p for p in patterns if p.signal == PatternSignal.NEUTRAL]
                                st.info(f"Summary: {len(bullish)} bullish, {len(bearish)} bearish, {len(neutral)} neutral")
                            else:
                                st.info("No candlestick patterns detected in the last 20 candles.")
                        else:
                            st.warning("Need at least 3 candles for pattern detection.")
                else:
                    st.error("Could not fetch data. Check ticker symbol.")

    def show_portfolio_manager():
        st.header("💼 Portfolio Manager")
        tabs = st.tabs(["📊 Current Holdings", "➕ Add Position", "📈 Performance", "📉 P&L Analysis"])
        with tabs[0]:
            st.subheader("📊 Your Portfolio")
            if st.session_state.portfolio:
                df = pd.DataFrame(st.session_state.portfolio)
                for col in ['Total Cost','Current Value','P&L','P&L %']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                total_inv = df['Total Cost'].sum()
                total_cur = df['Current Value'].sum()
                total_pnl = total_cur - total_inv
                total_pnl_pct = (total_pnl/total_inv)*100 if total_inv>0 else 0
                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    st.metric("Total Investment", format_indian_number(total_inv))
                with c2:
                    st.metric("Current Value", format_indian_number(total_cur))
                with c3:
                    st.metric("Total P&L", format_indian_number(total_pnl), f"{total_pnl_pct:+.2f}%")
                with c4:
                    st.metric("Holdings", len(df))
                disp = df.copy()
                for col in ['Buy Price','Current Price','Total Cost','Current Value','P&L']:
                    if col in disp.columns:
                        disp[col] = format_indian_series(disp[col], is_share_price=True)
                if 'P&L %' in disp.columns:
                    disp['P&L %'] = disp['P&L %'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
                st.dataframe(disp, width="stretch", hide_index=True)
                if PLOTLY_AVAILABLE:
                    fig = px.pie(df, values='Current Value', names='Stock', title='Portfolio Allocation')
                    fig.update_layout(template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Export to Excel", use_container_width=True):
                        excel_file = export_portfolio_to_excel(st.session_state.portfolio)
                        st.download_button(
                            label="Download Excel",
                            data=excel_file,
                            file_name=f"portfolio_{get_ist_time().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                with col2:
                    csv = pd.DataFrame(st.session_state.portfolio).to_csv(index=False)
                    st.download_button("📥 Export to CSV", csv, f"portfolio_{get_ist_time().strftime('%Y%m%d')}.csv")
            else:
                st.info("No holdings yet. Add your first position!")
        with tabs[1]:
            st.subheader("➕ Add New Position")
            with st.form("add_position_form"):
                col1, col2 = st.columns(2)
                with col1:
                    stock_name = st.text_input("Stock Name")
                    ticker_raw = st.text_input("Ticker Symbol", placeholder="RELIANCE.NS")
                    ticker = normalize_ticker(ticker_raw, st.session_state.exchange)
                    shares = st.number_input("Number of Shares", min_value=1, value=1)
                    buy_price = st.number_input("Buy Price (₹)", min_value=0.0, step=0.01)
                with col2:
                    buy_date = st.date_input("Purchase Date")
                    current_price = st.number_input("Current Price (₹)", min_value=0.0, step=0.01)
                    sector = st.selectbox("Sector", INDIAN_SECTORS)
                submitted = st.form_submit_button("➕ Add to Portfolio")
                if submitted:
                    if stock_name and shares>0 and buy_price>0:
                        existing_tickers =[p.get('Ticker', '') for p in st.session_state.portfolio]
                        if ticker in existing_tickers:
                            st.warning(f"⚠️ {ticker} already exists. Adding duplicate position.")
                        total_cost = shares*buy_price
                        cur_val = shares*current_price
                        pnl = cur_val - total_cost
                        pnl_pct = (pnl/total_cost)*100 if total_cost>0 else 0
                        pos = {
                            'Stock': stock_name, 'Ticker': ticker, 'Shares': shares,
                            'Buy Price': buy_price, 'Current Price': current_price,
                            'Total Cost': total_cost, 'Current Value': cur_val,
                            'P&L': pnl, 'P&L %': pnl_pct, 'Sector': sector,
                            'Date': buy_date.strftime('%Y-%m-%d')
                        }
                        st.session_state.portfolio.append(pos)
                        auto_save()
                        st.success(f"✅ Added {stock_name}")
                    else:
                        st.error("Please fill required fields")
        with tabs[2]:
            st.subheader("📈 Portfolio Performance")
            if st.session_state.portfolio:
                df = pd.DataFrame(st.session_state.portfolio)
                df['P&L %'] = pd.to_numeric(df['P&L %'], errors='coerce')
                top = df.nlargest(5, 'P&L %')[['Stock','P&L %']]
                bottom = df.nsmallest(5, 'P&L %')[['Stock','P&L %']]
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🏆 Top Performers")
                    for _,row in top.iterrows():
                        st.success(f"{row['Stock']}: +{row['P&L %']:.2f}%")
                with c2:
                    st.markdown("#### 📉 Worst Performers")
                    for _,row in bottom.iterrows():
                        st.error(f"{row['Stock']}: {row['P&L %']:.2f}%")
                st.divider()
                st.subheader("📉 Maximum Drawdown")
                st.info("To compute maximum drawdown, portfolio history is needed. This will be available when historical portfolio tracking is added.")
            else:
                st.info("No portfolio data")
        with tabs[3]:
            st.subheader("📉 Profit & Loss Analysis")
            if st.session_state.portfolio:
                df = pd.DataFrame(st.session_state.portfolio)
                df['P&L'] = pd.to_numeric(df['P&L'], errors='coerce')
                df['Total Cost'] = pd.to_numeric(df['Total Cost'], errors='coerce')
                sector_pnl = df.groupby('Sector').agg({'P&L':'sum','Total Cost':'sum'}).reset_index()
                sector_pnl['P&L %'] = (sector_pnl['P&L']/sector_pnl['Total Cost'])*100
                disp = sector_pnl.copy()
                disp['P&L'] = format_indian_series(disp['P&L'], is_share_price=False)
                st.dataframe(disp, width="stretch", hide_index=True)
                if PLOTLY_AVAILABLE:
                    fig = px.bar(sector_pnl, x='Sector', y='P&L', title='Sector-wise P&L',
                                color='P&L', color_continuous_scale=['red','yellow','green'])
                    fig.update_layout(template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)

    def show_trade_journal():
        st.header("📝 Trade Journal")
        tabs = st.tabs(["📋 All Trades", "➕ Log New Trade", "📊 Trade Statistics"])
        with tabs[0]:
            if st.session_state.trade_journal:
                df = pd.DataFrame(st.session_state.trade_journal)
                disp = df.copy()
                if 'P&L' in disp.columns:
                    disp['P&L'] = format_indian_series(disp['P&L'], is_share_price=False)
                if 'P&L %' in disp.columns:
                    disp['P&L %'] = disp['P&L %'].apply(lambda x: format_percent(x) if pd.notna(x) else "-")
                for c in ['Entry','Exit','Stop Loss','Target']:
                    if c in disp.columns:
                        disp[c] = format_indian_series(disp[c], is_share_price=True)
                st.dataframe(disp, width="stretch", hide_index=True)

                col1, col2 = st.columns(2)
                with col1:
                    csv = export_trades_to_csv(st.session_state.trade_journal)
                    st.download_button("📥 Export to CSV", csv, f"trades_{get_ist_time().strftime('%Y%m%d')}.csv")
                with col2:
                    if st.button("📥 Export to Excel", use_container_width=True):
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, sheet_name='Trades', index=False)
                        output.seek(0)
                        st.download_button(
                            label="Download Excel",
                            data=output,
                            file_name=f"trades_{get_ist_time().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            else:
                st.info("No trades logged yet.")
        with tabs[1]:
            st.subheader("➕ Log New Trade")
            with st.form("log_trade_form"):
                col1, col2 = st.columns(2)
                with col1:
                    trade_type = st.selectbox("Trade Type", ["Buy","Sell"])
                    stock = st.text_input("Stock Name")
                    qty = st.number_input("Quantity", min_value=1, value=1)
                    entry = st.number_input("Entry Price (₹)", min_value=0.0, step=0.01)
                    exit_price = st.number_input("Exit Price (₹)", min_value=0.0, step=0.01) if trade_type=="Sell" else 0
                with col2:
                    trade_date = st.date_input("Trade Date")
                    strategy = st.selectbox("Strategy",["Intraday","Swing","Positional","Long Term"])
                    stop = st.number_input("Stop Loss (₹)", min_value=0.0, step=0.01)
                    target = st.number_input("Target (₹)", min_value=0.0, step=0.01)
                notes = st.text_area("Trade Notes")
                submitted = st.form_submit_button("📝 Log Trade")
                if submitted:
                    if stock and qty>0 and entry>0:
                        trade = {
                            'Date': trade_date.strftime('%Y-%m-%d'),
                            'Type': trade_type, 'Stock': stock, 'Quantity': qty,
                            'Entry': entry, 'Stop Loss': stop, 'Target': target,
                            'Strategy': strategy, 'Notes': notes
                        }
                        if exit_price>0:
                            pnl = (exit_price-entry)*qty
                            pnl_pct = ((exit_price-entry)/entry)*100
                            trade['P&L'] = pnl
                            trade['P&L %'] = pnl_pct
                            trade['Exit'] = exit_price
                        else:
                            trade['P&L'] = trade['P&L %'] = trade['Exit'] = None
                        st.session_state.trade_journal.append(trade)
                        db = get_db()
                        if db and st.session_state.user_id:
                            db.save_trade(st.session_state.user_id, trade)
                        st.success(f"✅ Logged {trade_type} {qty} shares of {stock}")
                    else:
                        st.error("Please fill all required fields")
        with tabs[2]:
            st.subheader("📊 Trading Statistics")
            if st.session_state.trade_journal:
                df = pd.DataFrame(st.session_state.trade_journal)
                closed = df[df['P&L'].notna()].copy()
                if not closed.empty:
                    closed['P&L'] = pd.to_numeric(closed['P&L'], errors='coerce')
                    total = len(closed)
                    wins = len(closed[closed['P&L']>0])
                    losses = len(closed[closed['P&L']<0])
                    win_rate = (wins/total)*100 if total>0 else 0
                    total_pnl = closed['P&L'].sum()
                    avg_win = closed[closed['P&L']>0]['P&L'].mean() if wins>0 else 0
                    avg_loss = closed[closed['P&L']<0]['P&L'].mean() if losses>0 else 0
                    profit_factor = abs(avg_win/avg_loss) if avg_loss!=0 else float('inf')
                    c1,c2,c3,c4 = st.columns(4)
                    with c1:
                        st.metric("Total Closed Trades", total)
                        st.metric("Total P&L", format_indian_number(total_pnl))
                    with c2:
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                        st.metric("Profit Factor", f"{profit_factor:.2f}")
                    with c3:
                        st.metric("Avg Win", format_indian_number(avg_win))
                        st.metric("Winning Trades", wins)
                    with c4:
                        st.metric("Avg Loss", format_indian_number(avg_loss))
                        st.metric("Losing Trades", losses)
                else:
                    st.info("No closed trades yet")
            else:
                st.info("No trading statistics available")

    def show_position_sizer():
        st.header("🎯 Position Size Calculator")
        st.markdown("### 💡 Risk Management")
        with st.form("position_sizer_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Portfolio Details")
                portfolio = st.number_input("Portfolio Value (₹)", min_value=0.0, value=1000000.0, step=10000.0)
                risk_pct = st.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.25)
                st.info(f"💰 Risk Amount: {format_indian_number(portfolio * risk_pct / 100)}")
            with col2:
                st.subheader("🎯 Trade Details")
                trade_type = st.radio("Trade Type", ["Long", "Short"], horizontal=True)
                entry = st.number_input("Entry Price (₹)", min_value=0.0, step=0.01)
                stop = st.number_input("Stop Loss (₹)", min_value=0.0, step=0.01)
                target = st.number_input("Target Price (₹)", min_value=0.0, step=0.01)
            submitted = st.form_submit_button("🧮 Calculate Position Size")
            if submitted:
                if not validate_number_input(portfolio, min_val=1000):
                    st.error("Portfolio value must be at least ₹1,000")
                elif not validate_number_input(entry, min_val=0.01):
                    st.error("Entry price must be positive")
                elif not validate_number_input(stop, min_val=0.01):
                    st.error("Stop loss must be positive")
                elif trade_type == "Long" and entry <= stop:
                    st.error("For Long trades, Entry price must be greater than Stop Loss")
                elif trade_type == "Short" and entry >= stop:
                    st.error("For Short trades, Entry price must be less than Stop Loss")
                else:
                    res = calculate_position_size(portfolio, risk_pct, entry, stop, trade_type)
                    if res:
                        st.success("✅ Position Size Calculated")
                        c1,c2,c3 = st.columns(3)
                        with c1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="color:#888;">{trade_type.upper()} QUANTITY</div>
                                <div style="color:#00d4ff; font-size:2.5rem;">{res['shares']}</div>
                                <div style="color:#888;">shares</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="color:#888;">POSITION VALUE</div>
                                <div style="color:#00d4ff; font-size:2.5rem;">{format_indian_number(res['position_value'])}</div>
                                <div style="color:#888;">{res['position_pct']:.2f}% of portfolio</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with c3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="color:#888;">RISK AMOUNT</div>
                                <div style="color:#ff4444; font-size:2.5rem;">{format_indian_number(res['risk_amount'])}</div>
                                <div style="color:#888;">if SL hit</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        pot_profit = 0
                        rr = 0
                        if trade_type == "Long" and target > entry:
                            pot_profit = (target - entry) * res['shares']
                            rr = pot_profit / res['risk_amount']
                        elif trade_type == "Short" and target < entry:
                            pot_profit = (entry - target) * res['shares']
                            rr = pot_profit / res['risk_amount']
                            
                        st.divider()
                        st.markdown("### 📊 Risk-Reward Analysis")
                        c1,c2,c3 = st.columns(3)
                        with c1:
                            st.metric("Potential Profit", format_indian_number(pot_profit) if pot_profit > 0 else "N/A")
                        with c2:
                            st.metric("Risk-Reward Ratio", f"1:{rr:.2f}" if rr > 0 else "N/A")
                        with c3:
                            if rr >= 2:
                                st.success("✅ Good R:R Ratio")
                            elif rr >= 1.5:
                                st.warning("⚠️ Acceptable R:R")
                            elif rr > 0:
                                st.error("❌ Poor R:R - Reconsider")
                            else:
                                st.info("Target not set properly.")
                        st.divider()
                        st.markdown("### 📋 Trade Plan")
                        st.markdown(f"""
                        <div class="trade-entry">
                            <h4>{'🟢' if trade_type == 'Long' else '🔴'} ENTRY PLAN</h4>
                            <p><b>{trade_type}:</b> {res['shares']} shares @ {format_indian_number(entry, is_share_price=True)}</p>
                            <p><b>Total Value:</b> {format_indian_number(res['position_value'])}</p>
                        </div>
                        <div class="trade-exit">
                            <h4>🏁 EXIT PLAN</h4>
                            <p><b>Stop Loss:</b> {format_indian_number(stop, is_share_price=True)} (Loss: {format_indian_number(res['risk_amount'])})</p>
                            <p><b>Target:</b> {format_indian_number(target, is_share_price=True)} (Profit: {format_indian_number(pot_profit)})</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Position too small – risk amount less than price risk.")

    def show_tax_calculator():
        st.header("💰 Tax Calculator (India)")
        st.markdown("### 📊 Calculate Tax Liability (Delivery Trades)")
        with st.form("tax_form"):
            col1, col2 = st.columns(2)
            with col1:
                buy = st.number_input("Buy Price (₹)", min_value=0.0, step=0.01)
                sell = st.number_input("Sell Price (₹)", min_value=0.0, step=0.01)
                shares = st.number_input("Number of Shares", min_value=1, value=1)
            with col2:
                buy_date = st.date_input("Purchase Date")
                sell_date = st.date_input("Sale Date")
            submitted = st.form_submit_button("💰 Calculate Tax")
            if submitted:
                if buy > 0 and sell > 0 and shares > 0:
                    days = (sell_date - buy_date).days
                    res = calculate_tax(buy, sell, shares, days)
                    st.success(f"✅ Tax Calculated ({res['tax_type']})")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Gross Profit", format_indian_number(res['gross_profit']))
                    with c2:
                        st.metric("STT (0.1% each side)", format_indian_number(res['stt']))
                    with c3:
                        st.metric("Capital Gains Tax", format_indian_number(res['capital_gains_tax']))
                    with c4:
                        st.metric("Net Profit", format_indian_number(res['net_profit']))
                    st.divider()
                    st.markdown("### 📋 Tax Breakdown")
                    breakdown = pd.DataFrame({
                        'Item':['Sale Value', 'Purchase Cost', 'Gross Profit', 'STT', 'Capital Gains Tax', 'Total Tax', 'Net Profit'],
                        'Amount (₹)':[sell * shares, buy * shares, res['gross_profit'], res['stt'], res['capital_gains_tax'], res['total_tax'], res['net_profit']]
                    })
                    breakdown['Amount (₹)'] = format_indian_series(breakdown['Amount (₹)'], is_share_price=False)
                    st.dataframe(breakdown, width="stretch", hide_index=True)
                    st.info(f"📅 Holding Period: {days} days ({res['tax_type']})")
                else:
                    st.error("Please enter valid numbers.")

    def show_tax_pnl_report():
        st.header("📉 Tax P&L Report (Financial Year)")
        st.markdown("Aggregate realized gains/losses for the selected financial year.")
        if not st.session_state.trade_journal:
            st.info("No trades logged yet.")
            return
        df = pd.DataFrame(st.session_state.trade_journal)
        closed = df[df['P&L'].notna()].copy()
        if closed.empty:
            st.info("No closed trades found.")
            return
        closed['P&L'] = pd.to_numeric(closed['P&L'], errors='coerce')
        closed['Date'] = pd.to_datetime(closed['Date'])
        financial_years = sorted(closed['Date'].dt.year.unique())
        if not financial_years:
            st.info("No date information.")
            return
        selected_year = st.selectbox("Select Financial Year (Start Year)", financial_years)
        start_date = pd.Timestamp(f"{selected_year}-04-01")
        end_date = pd.Timestamp(f"{selected_year+1}-03-31")
        filtered = closed[(closed['Date'] >= start_date) & (closed['Date'] <= end_date)]
        if filtered.empty:
            st.info(f"No trades in FY {selected_year}-{selected_year+1}")
            return
        st.subheader(f"📊 Trades in FY {selected_year}-{selected_year+1}")
        def compute_trade_tax(row):
            buy_price = row['Entry']
            sell_price = row['Exit']
            qty = row['Quantity']
            profit = (sell_price - buy_price) * qty if sell_price else 0
            if profit <= 0:
                tax = 0
                tax_type = 'Loss'
            else:
                tax = profit * (CONFIG.stcg_equity_rate / 100)
                tax_type = 'STCG'
            return tax, tax_type
        filtered['Tax'] = filtered.apply(compute_trade_tax, axis=1, result_type='expand')[0]
        filtered['Tax Type'] = filtered.apply(compute_trade_tax, axis=1, result_type='expand')[1]
        st.dataframe(filtered[['Date','Stock','Type','Quantity','Entry','Exit','P&L','Tax','Tax Type']])
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            total_profit = filtered[filtered['P&L']>0]['P&L'].sum()
            total_loss = filtered[filtered['P&L']<0]['P&L'].sum()
            st.metric("Total Realized Profit", format_indian_number(total_profit))
            st.metric("Total Realized Loss", format_indian_number(abs(total_loss)))
        with col2:
            net_pnl = total_profit + total_loss
            st.metric("Net P&L", format_indian_number(net_pnl))
        with col3:
            total_tax = filtered['Tax'].sum()
            st.metric("Estimated Tax Liability", format_indian_number(total_tax))
        st.info("Note: This is a simplified calculation assuming all trades are STCG (held < 1 year). For accurate LTCG, purchase date must be stored.")

    def show_stock_screener():
        st.header("🔍 Smart Stock Screener")
        st.markdown("Screen Nifty 50 + Next 50 stocks using live fundamental & technical filters.")

        with st.form("screener_form"):
            st.markdown("#### 📊 Fundamental Filters")
            col1, col2, col3 = st.columns(3)
            with col1:
                pe_max = st.slider("Max P/E Ratio", 0, 150, 50)
                pb_max = st.slider("Max P/B Ratio", 0.0, 20.0, 5.0)
            with col2:
                roe_min = st.slider("Min ROE (%)", 0, 50, 12)
                de_max = st.slider("Max Debt/Equity", 0.0, 5.0, 2.0)
            with col3:
                div_min = st.slider("Min Dividend Yield (%)", 0.0, 10.0, 0.0)
                mc_min = st.selectbox("Min Market Cap", ["Any", "Small Cap (>500Cr)", "Mid Cap (>5000Cr)", "Large Cap (>20000Cr)"])

            st.markdown("#### 📈 Technical Filters")
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_min, rsi_max = st.slider("RSI Range", 0, 100, (30, 70))
            with col2:
                above_sma50 = st.checkbox("Price > SMA 50", value=True)
                above_sma200 = st.checkbox("Price > SMA 200", value=False)
            with col3:
                sectors_filter = st.multiselect("Sectors", INDIAN_SECTORS, default=[])

            universe_choice = st.selectbox("Universe", ["Nifty 50", "Nifty 50 + Next 50"])
            submitted = st.form_submit_button("🔍 Run Screener", type="primary", use_container_width=True)

        if submitted:
            universe = NIFTY_50 if universe_choice == "Nifty 50" else {**NIFTY_50, **NIFTY_NEXT50}
            results = []
            progress = st.progress(0)
            status = st.empty()
            tickers = list(universe.keys())

            for idx, ticker in enumerate(tickers):
                progress.progress((idx+1)/len(tickers))
                name, sector = universe[ticker]
                if sectors_filter and sector not in sectors_filter:
                    continue
                status.text(f"Screening {name}…")
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    pe = info.get('trailingPE')
                    pb = info.get('priceToBook')
                    roe = info.get('returnOnEquity')
                    de = info.get('debtToEquity')
                    div = info.get('dividendYield', 0) or 0
                    mc = info.get('marketCap', 0) or 0
                    cp = info.get('currentPrice') or info.get('regularMarketPrice')

                    if pe and pe > pe_max: continue
                    if pb and pb > pb_max: continue
                    if roe and roe * 100 < roe_min: continue
                    if de and de / 100 > de_max: continue
                    if div * 100 < div_min: continue

                    mc_cr = mc / 1e7
                    if mc_min == "Small Cap (>500Cr)" and mc_cr < 500: continue
                    elif mc_min == "Mid Cap (>5000Cr)" and mc_cr < 5000: continue
                    elif mc_min == "Large Cap (>20000Cr)" and mc_cr < 20000: continue

                    # Technical
                    if above_sma50 or above_sma200 or rsi_min > 0 or rsi_max < 100:
                        df_t = compute_technical_indicators_cached(ticker, '6mo')
                        if df_t is not None and not df_t.empty:
                            last = df_t.iloc[-1]
                            if above_sma50 and 'SMA_50' in df_t.columns:
                                if last['Close'] <= last['SMA_50']: continue
                            if above_sma200 and 'SMA_200' in df_t.columns:
                                if last['Close'] <= last['SMA_200']: continue
                            if 'RSI' in df_t.columns:
                                rsi_v = last['RSI']
                                if not (rsi_min <= rsi_v <= rsi_max): continue

                    results.append({
                        'Stock': name,
                        'Ticker': ticker,
                        'Sector': sector,
                        'Price (₹)': format_indian_number(cp, True) if cp else '–',
                        'P/E': f"{pe:.1f}" if pe else '–',
                        'P/B': f"{pb:.2f}" if pb else '–',
                        'ROE %': f"{roe*100:.1f}" if roe else '–',
                        'D/E': f"{de/100:.2f}" if de else '–',
                        'Div Yield %': f"{div*100:.2f}" if div else '–',
                        'Mkt Cap': format_indian_number(mc),
                    })
                except Exception as e:
                    logger.warning(f"Screener skip {ticker}: {e}")

            progress.empty()
            status.empty()

            if results:
                st.success(f"✅ Found {len(results)} stocks matching your criteria")
                res_df = pd.DataFrame(results)
                st.dataframe(res_df, use_container_width=True, hide_index=True)
                csv = res_df.to_csv(index=False)
                st.download_button("⬇️ Export Results CSV", csv,
                    file_name=f"screener_{get_ist_time().strftime('%Y%m%d')}.csv",
                    mime="text/csv")
            else:
                st.warning("No stocks matched your criteria. Try relaxing the filters.")

    # -------------------------------------------------------
    # FULLY IMPLEMENTED MODULES
    # -------------------------------------------------------

    # ── STRATEGY BACKTESTER ────────────────────────────────
    def show_backtester():
        st.header("📈 Strategy Backtester")

        col1, col2 = st.columns([2, 1])
        with col1:
            ticker_raw = st.text_input("Stock Symbol", value="RELIANCE.NS", key="bt_ticker")
            ticker = normalize_ticker(ticker_raw, st.session_state.exchange)
        with col2:
            period = st.selectbox("Period",["1y", "2y", "3y", "5y"], key="bt_period")

        st.markdown("### ⚙️ Strategy Settings")
        strategy = st.selectbox("Strategy",[
            "SMA Crossover (20/50)",
            "SMA Crossover (50/200) – Golden/Death Cross",
            "RSI Mean Reversion",
            "MACD Signal Cross",
            "Bollinger Band Breakout",
        ], key="bt_strategy")

        col1, col2, col3 = st.columns(3)
        with col1:
            initial_capital = st.number_input("Initial Capital (₹)", value=100000.0, step=10000.0, min_value=10000.0)
        with col2:
            position_pct = st.slider("Position Size (% of capital)", 10, 100, 100, step=10)
        with col3:
            commission_pct = st.number_input("Commission (%)", value=0.05, step=0.01, min_value=0.0, max_value=2.0)

        if st.button("▶️ Run Backtest", type="primary"):
            with st.spinner("Running backtest…"):
                df = fetch_stock_data(ticker, period)
                if df is None or df.empty:
                    st.error("Could not fetch data. Check ticker symbol.")
                    return

                df = calculate_technical_indicators(df.copy())
                df = df.dropna()

                # Generate signals
                signals = pd.Series(0, index=df.index)

                if strategy == "SMA Crossover (20/50)":
                    signals[df['SMA_20'] > df['SMA_50']] = 1
                    signals[df['SMA_20'] < df['SMA_50']] = -1
                elif strategy == "SMA Crossover (50/200) – Golden/Death Cross":
                    signals[df['SMA_50'] > df['SMA_200']] = 1
                    signals[df['SMA_50'] < df['SMA_200']] = -1
                elif strategy == "RSI Mean Reversion":
                    signals[df['RSI'] < 30] = 1
                    signals[df['RSI'] > 70] = -1
                elif strategy == "MACD Signal Cross":
                    signals[df['MACD'] > df['MACD_Signal']] = 1
                    signals[df['MACD'] < df['MACD_Signal']] = -1
                elif strategy == "Bollinger Band Breakout":
                    signals[df['Close'] > df['BB_Upper']] = 1
                    signals[df['Close'] < df['BB_Lower']] = -1

                # Simple vectorised backtest
                position = signals.shift(1).fillna(0)
                daily_ret = df['Close'].pct_change().fillna(0)
                commission = commission_pct / 100
                trades_count = position.diff().abs().sum() / 2
                total_commission = trades_count * commission * initial_capital * position_pct / 100

                strategy_ret = position * daily_ret * (position_pct / 100)
                capital_curve = (1 + strategy_ret).cumprod() * initial_capital
                bh_curve = (1 + daily_ret).cumprod() * initial_capital

                final_val = capital_curve.iloc[-1] - total_commission
                total_ret = (final_val - initial_capital) / initial_capital * 100
                bh_ret = (bh_curve.iloc[-1] - initial_capital) / initial_capital * 100

                # Drawdown
                roll_max = capital_curve.cummax()
                drawdown = (capital_curve - roll_max) / roll_max * 100
                max_dd = drawdown.min()

                # Sharpe (annualised, risk-free 7%)
                excess = strategy_ret - CONFIG.risk_free_rate / 100 / 252
                sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() != 0 else 0

                # Win/loss
                trade_rets = strategy_ret[position.diff() != 0].dropna()
                wins = (trade_rets > 0).sum()
                losses = (trade_rets < 0).sum()
                win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

                st.divider()
                st.markdown("### 📊 Backtest Results")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    color = "#00d084" if total_ret > 0 else "#ff4444"
                    st.markdown(f"""<div class="metric-card">
                        <div style="color:#888;">Strategy Return</div>
                        <div style="color:{color};font-size:2rem;font-weight:600;">{total_ret:+.2f}%</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    color2 = "#00d084" if bh_ret > 0 else "#ff4444"
                    st.markdown(f"""<div class="metric-card">
                        <div style="color:#888;">Buy & Hold Return</div>
                        <div style="color:{color2};font-size:2rem;font-weight:600;">{bh_ret:+.2f}%</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""<div class="metric-card">
                        <div style="color:#888;">Max Drawdown</div>
                        <div style="color:#ff4444;font-size:2rem;font-weight:600;">{max_dd:.2f}%</div>
                    </div>""", unsafe_allow_html=True)
                with c4:
                    sharpe_color = "#00d084" if sharpe > 1 else "#ffa500" if sharpe > 0 else "#ff4444"
                    st.markdown(f"""<div class="metric-card">
                        <div style="color:#888;">Sharpe Ratio</div>
                        <div style="color:{sharpe_color};font-size:2rem;font-weight:600;">{sharpe:.2f}</div>
                    </div>""", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Final Portfolio Value", format_indian_number(final_val))
                    st.metric("Total Commission Paid", format_indian_number(total_commission))
                with c2:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                    st.metric("Total Trades", int(trades_count))
                with c3:
                    alpha = total_ret - bh_ret
                    st.metric("Alpha vs B&H", f"{alpha:+.2f}%")

                if PLOTLY_AVAILABLE:
                    st.markdown("### 📈 Equity Curve")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=capital_curve.index, y=capital_curve.values,
                                             name="Strategy", line=dict(color="#00d4ff")))
                    fig.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve.values,
                                             name="Buy & Hold", line=dict(color="#ffa500", dash="dash")))
                    fig.update_layout(title=f"{ticker} – {strategy}", yaxis_title="Portfolio Value (₹)",
                                      template="plotly_dark", height=450)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### 📉 Drawdown")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values,
                                              fill="tozeroy", name="Drawdown %",
                                              line=dict(color="#ff4444")))
                    fig2.update_layout(yaxis_title="Drawdown (%)", template="plotly_dark", height=300)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.line_chart(pd.DataFrame({"Strategy": capital_curve, "Buy & Hold": bh_curve}))

    # ── RISK ANALYTICS ─────────────────────────────────────
    def show_risk_analytics():
        st.header("📉 Risk Analytics")

        if not st.session_state.portfolio:
            st.info("💡 Add holdings in Portfolio Manager first to unlock risk analytics.")
            return

        tickers =[p.get("Ticker", "") for p in st.session_state.portfolio if p.get("Ticker")]
        if not tickers:
            st.warning("No tickers found in portfolio. Make sure each position has a Ticker field.")
            return

        period = st.selectbox("Analysis Period",["1y", "2y", "3y"], key="ra_period")

        with st.spinner("Fetching portfolio price data…"):
            price_data = {}
            for t in tickers:
                df = fetch_stock_data(t, period)
                if df is not None and not df.empty:
                    price_data[t] = df["Close"]

        if not price_data:
            st.error("Could not fetch price data for any portfolio ticker.")
            return

        prices = pd.DataFrame(price_data).dropna()
        returns = prices.pct_change().dropna()

        # Weights from portfolio values
        weights_raw = {}
        for p in st.session_state.portfolio:
            t = p.get("Ticker", "")
            if t in prices.columns:
                weights_raw[t] = p.get("Current Value") or p.get("Total Cost") or 1.0
        total_w = sum(weights_raw.values()) or 1
        weights = {t: v / total_w for t, v in weights_raw.items()}
        w_arr = np.array([weights.get(c, 0) for c in prices.columns])

        # Portfolio returns
        port_ret = (returns * w_arr).sum(axis=1)

        # Metrics
        ann_ret = port_ret.mean() * 252 * 100
        ann_vol = port_ret.std() * np.sqrt(252) * 100
        excess = port_ret - CONFIG.risk_free_rate / 100 / 252
        sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() != 0 else 0

        # VaR (Historical 95% & 99%)
        var_95 = np.percentile(port_ret, 5) * 100
        var_99 = np.percentile(port_ret, 1) * 100

        # CVaR (Expected Shortfall)
        cvar_95 = port_ret[port_ret <= np.percentile(port_ret, 5)].mean() * 100

        # Max Drawdown
        cum = (1 + port_ret).cumprod()
        roll_max = cum.cummax()
        dd = (cum - roll_max) / roll_max
        max_dd = dd.min() * 100

        # Beta vs NIFTY 50
        try:
            nifty = fetch_stock_data("^NSEI", period)
            if nifty is not None and not nifty.empty:
                nifty_ret = nifty["Close"].pct_change().dropna()
                aligned = pd.concat([port_ret, nifty_ret], axis=1).dropna()
                aligned.columns =["port", "nifty"]
                cov = np.cov(aligned["port"], aligned["nifty"])
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else None
            else:
                beta = None
        except Exception:
            beta = None

        st.markdown("### 📊 Portfolio Risk Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            col_ann = "#00d084" if ann_ret > 0 else "#ff4444"
            st.markdown(f"""<div class="metric-card">
                <div style="color:#888;">Annualised Return</div>
                <div style="color:{col_ann};font-size:1.8rem;font-weight:600;">{ann_ret:+.2f}%</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div style="color:#888;">Annualised Volatility</div>
                <div style="color:#ffa500;font-size:1.8rem;font-weight:600;">{ann_vol:.2f}%</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            sc = "#00d084" if sharpe > 1 else "#ffa500" if sharpe > 0 else "#ff4444"
            st.markdown(f"""<div class="metric-card">
                <div style="color:#888;">Sharpe Ratio</div>
                <div style="color:{sc};font-size:1.8rem;font-weight:600;">{sharpe:.2f}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div style="color:#888;">Max Drawdown</div>
                <div style="color:#ff4444;font-size:1.8rem;font-weight:600;">{max_dd:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("### ⚠️ Value at Risk (1-day, Historical Simulation)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("VaR 95%", f"{var_95:.2f}%", help="On a bad day (1 in 20), expect this loss or worse")
        with c2:
            st.metric("VaR 99%", f"{var_99:.2f}%", help="On a very bad day (1 in 100)")
        with c3:
            st.metric("CVaR 95% (Expected Shortfall)", f"{cvar_95:.2f}%",
                      help="Average loss on days beyond the 95% VaR threshold")
        if beta is not None:
            st.metric("Beta vs NIFTY 50", f"{beta:.2f}",
                      help="<1 = less volatile than market, >1 = more volatile")

        if PLOTLY_AVAILABLE:
            tabs = st.tabs(["📈 Cumulative Return", "📊 Return Distribution", "🔗 Correlation Matrix", "📉 Drawdown"])
            with tabs[0]:
                fig = go.Figure()
                for col in prices.columns:
                    norm = (prices[col] / prices[col].iloc[0] - 1) * 100
                    fig.add_trace(go.Scatter(x=norm.index, y=norm.values, name=col))
                port_cum = (cum - 1) * 100
                fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values,
                                         name="Portfolio", line=dict(width=3, color="#00d4ff")))
                fig.update_layout(yaxis_title="Return (%)", template="plotly_dark", height=450)
                st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=port_ret * 100, nbinsx=50, name="Daily Returns",
                                           marker_color="#00d4ff", opacity=0.75))
                fig.add_vline(x=var_95, line_color="orange", line_dash="dash",
                              annotation_text=f"VaR 95%: {var_95:.2f}%")
                fig.add_vline(x=var_99, line_color="red", line_dash="dash",
                              annotation_text=f"VaR 99%: {var_99:.2f}%")
                fig.update_layout(xaxis_title="Daily Return (%)", template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

            with tabs[2]:
                corr = returns.corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                    colorscale="RdBu", zmid=0, text=corr.round(2).values,
                    texttemplate="%{text}", colorbar=dict(title="Correlation")))
                fig.update_layout(title="Stock Return Correlation Matrix",
                                  template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

            with tabs[3]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dd.index, y=dd.values * 100,
                                         fill="tozeroy", line=dict(color="#ff4444"), name="Drawdown %"))
                fig.update_layout(yaxis_title="Drawdown (%)", template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart((prices / prices.iloc[0] - 1) * 100)

    # ── QUICK TRADE SETUP ──────────────────────────────────
    def show_quick_trade_setup():
        st.header("📱 Quick Trade Setup")
        st.markdown("Instant trade plan with ATR-based stops, targets and position sizing.")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            ticker_raw = st.text_input("Stock Symbol", value="RELIANCE.NS", key="qts_ticker")
            ticker = normalize_ticker(ticker_raw, st.session_state.exchange)
        with col2:
            trade_type = st.radio("Direction",["Long (Buy)", "Short (Sell)"], key="qts_dir")
        with col3:
            atr_mult_sl = st.number_input("ATR × for Stop Loss", value=1.5, step=0.25, min_value=0.5)
            atr_mult_tgt = st.number_input("ATR × for Target", value=3.0, step=0.25, min_value=0.5)

        capital = st.number_input("Available Capital (₹)", value=100000.0, step=10000.0, min_value=1000.0)
        risk_pct = st.slider("Max Risk per Trade (%)", 0.5, 5.0, 2.0, 0.25)

        if st.button("⚡ Generate Trade Plan", type="primary"):
            with st.spinner("Fetching live data…"):
                df = compute_technical_indicators_cached(ticker, "3mo")

            if df is None or df.empty:
                st.error("Could not fetch data. Check ticker.")
                return

            latest = df.iloc[-1]
            price = latest["Close"]
            atr = latest.get("ATR", price * 0.02)

            is_long = "Long" in trade_type
            sl = price - atr_mult_sl * atr if is_long else price + atr_mult_sl * atr
            tgt = price + atr_mult_tgt * atr if is_long else price - atr_mult_tgt * atr

            risk_per_share = abs(price - sl)
            risk_amount = capital * risk_pct / 100
            shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            position_value = shares * price
            reward = abs(tgt - price) * shares
            rr = abs(tgt - price) / risk_per_share if risk_per_share > 0 else 0

            # Signal summary
            rsi = latest.get("RSI", 50)
            macd = latest.get("MACD", 0)
            macd_sig = latest.get("MACD_Signal", 0)
            above_sma50 = latest["Close"] > latest.get("SMA_50", latest["Close"])

            signals_ok = 0
            signal_rows =[]
            if is_long:
                ok = rsi < 65; signals_ok += ok
                signal_rows.append(("RSI", f"{rsi:.1f}", "✅" if ok else "⚠️", "Below 65 for long"))
                ok = macd > macd_sig; signals_ok += ok
                signal_rows.append(("MACD", f"{macd:.2f} vs {macd_sig:.2f}", "✅" if ok else "⚠️", "MACD above signal"))
                ok = above_sma50; signals_ok += ok
                signal_rows.append(("50 SMA", f"{latest.get('SMA_50', 0):.2f}", "✅" if ok else "⚠️", "Price above SMA50"))
            else:
                ok = rsi > 35; signals_ok += ok
                signal_rows.append(("RSI", f"{rsi:.1f}", "✅" if ok else "⚠️", "Above 35 for short"))
                ok = macd < macd_sig; signals_ok += ok
                signal_rows.append(("MACD", f"{macd:.2f} vs {macd_sig:.2f}", "✅" if ok else "⚠️", "MACD below signal"))
                ok = not above_sma50; signals_ok += ok
                signal_rows.append(("50 SMA", f"{latest.get('SMA_50', 0):.2f}", "✅" if ok else "⚠️", "Price below SMA50"))

            st.divider()
            overall_color = "#00d084" if signals_ok == 3 else "#ffa500" if signals_ok == 2 else "#ff4444"
            st.markdown(f"""<div class="metric-card">
                <div style="color:#888;">Signal Confluence</div>
                <div style="color:{overall_color};font-size:2rem;font-weight:700;">{signals_ok}/3 signals confirmed</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("### 🔍 Signal Checklist")
            for ind, val, status, note in signal_rows:
                c1, c2, c3, c4 = st.columns([1, 2, 1, 3])
                with c1: st.write(f"**{ind}**")
                with c2: st.write(val)
                with c3: st.write(status)
                with c4: st.write(note)

            st.divider()
            st.markdown("### 📋 Trade Plan")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div style="color:#888;">Entry Price</div>
                    <div style="color:#00d4ff;font-size:1.8rem;">{format_indian_number(price, True)}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                    <div style="color:#888;">Stop Loss ({atr_mult_sl}× ATR)</div>
                    <div style="color:#ff4444;font-size:1.8rem;">{format_indian_number(sl, True)}</div>
                    <div style="color:#888;">{format_indian_number(abs(price-sl), True)} risk/share</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                    <div style="color:#888;">Target ({atr_mult_tgt}× ATR)</div>
                    <div style="color:#00d084;font-size:1.8rem;">{format_indian_number(tgt, True)}</div>
                    <div style="color:#888;">{format_indian_number(abs(tgt-price), True)} reward/share</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                rr_color = "#00d084" if rr >= 2 else "#ffa500" if rr >= 1.5 else "#ff4444"
                st.markdown(f"""<div class="metric-card">
                    <div style="color:#888;">Risk:Reward</div>
                    <div style="color:{rr_color};font-size:1.8rem;">1:{rr:.1f}</div>
                </div>""", unsafe_allow_html=True)

            st.divider()
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Shares to Buy", f"{shares:,}")
            with c2:
                st.metric("Position Value", format_indian_number(position_value))
            with c3:
                st.metric("Max Profit Potential", format_indian_number(reward))

            st.metric("ATR (14-day)", format_indian_number(atr, True))

            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                window = df.tail(60)
                fig.add_trace(go.Candlestick(x=window.index, open=window["Open"],
                                             high=window["High"], low=window["Low"],
                                             close=window["Close"], name="Price"))
                fig.add_hline(y=price, line_color="#00d4ff", line_dash="dot",
                              annotation_text=f"Entry {format_indian_number(price, True)}")
                fig.add_hline(y=sl, line_color="#ff4444", line_dash="dash",
                              annotation_text=f"SL {format_indian_number(sl, True)}")
                fig.add_hline(y=tgt, line_color="#00d084", line_dash="dash",
                              annotation_text=f"Target {format_indian_number(tgt, True)}")
                fig.update_layout(title=f"{ticker} – Quick Trade Setup",
                                  template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)

    # ── ALERTS ─────────────────────────────────────────────
    def show_alerts():
        st.header("🔔 Price Alerts")
        alert_manager.display_alerts_ui()

        st.divider()
        st.markdown("### 📊 Alert History")
        all_alerts = st.session_state.get("alerts", [])
        triggered = [a for a in all_alerts if a.get("triggered")]
        pending = [a for a in all_alerts if not a.get("triggered")]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Pending Alerts", len(pending))
        with c2:
            st.metric("Triggered Alerts", len(triggered))

        if triggered:
            st.markdown("#### ✅ Triggered Alerts")
            for a in reversed(triggered):
                st.success(
                    f"**{a['ticker']}** – {a['type'].replace('_', ' ').title()} "
                    f"{format_indian_number(a['threshold'], True)} "
                    f"→ triggered @ {format_indian_number(a.get('triggered_price', 0), True)} "
                    f"on {a.get('triggered_at', 'N/A')}"
                )

        if pending:
            st.markdown("#### ⏳ Pending Alerts")
            for a in pending:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.info(
                        f"**{a['ticker']}** – {a['type'].replace('_', ' ').title()} "
                        f"{format_indian_number(a['threshold'], True)}"
                    )
                with col2:
                    if st.button("🗑️ Delete", key=f"del_alert_{a['id']}"):
                        st.session_state.alerts =[x for x in st.session_state.alerts if x["id"] != a["id"]]
                        st.rerun()

        st.divider()
        st.markdown("### ℹ️ How Alerts Work")
        st.info(
            "Alerts are checked synchronously as you interact with the app, with a debounce of 60 seconds "
            "to avoid spamming API calls. When the price crosses your threshold, the alert fires and appears "
            "at the top of this page. Alerts do **not** persist between browser sessions unless connected to a database."
        )

    # ── EDUCATION CENTER ───────────────────────────────────
    def show_education_center():
        render_glass_header("📚", "Education Center", "Market glossary, indicators, tax, and psychology")
        tabs = st.tabs([
            "📖 Glossary",
            "📈 Technical Indicators",
            "🕯️ Candlestick Patterns",
            "💰 Tax Guide (India)",
            "🎯 Trading Psychology",
            "📊 Fundamental Analysis 101",
        ])

        with tabs[0]:
            st.markdown("### 📖 Stock Market Glossary")
            glossary = {
                "Equity / Share": "A unit of ownership in a company. Shareholders are entitled to a portion of the company's profits (dividends) and assets.",
                "Market Capitalisation": "Total market value of a company's outstanding shares = Current Price × Total Shares.",
                "Face Value (FV)": "The nominal value of a share as stated in the company's charter. Often ₹1, ₹2, or ₹10.",
                "Book Value": "Net asset value of the company divided by the number of shares (Assets − Liabilities) / Shares.",
                "EPS (Earnings Per Share)": "Net Profit ÷ Total Shares. Higher EPS generally indicates better profitability.",
                "P/E Ratio": "Price ÷ EPS. How much investors pay per rupee of earnings. Lower may indicate undervaluation.",
                "P/B Ratio": "Price ÷ Book Value Per Share. <1 may suggest the stock is trading below its net assets.",
                "Dividend Yield": "Annual Dividend ÷ Share Price × 100. Income return from holding a stock.",
                "ROCE": "Return on Capital Employed – measures how efficiently a company uses capital to generate profit.",
                "Debt-to-Equity": "Total Debt ÷ Equity. Higher values indicate more leverage and financial risk.",
                "Circuit Breaker": "An automatic trading halt triggered when a stock moves ±5%, ±10%, or ±20% in a session.",
                "Intraday Trading": "Buying and selling shares within the same trading session (9:15 AM – 3:30 PM IST).",
                "Delivery Trading": "Taking physical ownership of shares, held overnight or longer.",
                "F&O (Futures & Options)": "Derivative instruments that derive value from an underlying asset like stocks or indices.",
                "SEBI": "Securities and Exchange Board of India – the market regulator.",
                "NSE / BSE": "National Stock Exchange and Bombay Stock Exchange – India's two main stock exchanges.",
                "NIFTY 50": "Index tracking the 50 largest companies by market cap on NSE.",
                "SENSEX": "Index of 30 major companies listed on BSE.",
                "STT": "Securities Transaction Tax – levied on the value of securities traded.",
                "LTCG": "Long Term Capital Gains – gains from equity held >12 months; taxed at 10% above ₹1 lakh.",
                "STCG": "Short Term Capital Gains – gains from equity held ≤12 months; taxed at 15%.",
            }
            search = st.text_input("🔍 Search glossary…", key="edu_gloss_search")
            for term, defn in glossary.items():
                if not search or search.lower() in term.lower() or search.lower() in defn.lower():
                    with st.expander(f"**{term}**"):
                        st.write(defn)

        with tabs[1]:
            st.markdown("### 📈 Technical Indicators Guide")
            indicators = {
                "SMA – Simple Moving Average": {
                    "formula": "Average of closing prices over N periods",
                    "interpretation": "Price above SMA = uptrend. Golden Cross (50 SMA crosses above 200 SMA) is bullish. Death Cross is bearish.",
                    "settings": "Common: 20, 50, 200 days",
                    "limitations": "Lagging indicator – reacts after the move has started.",
                },
                "EMA – Exponential Moving Average": {
                    "formula": "Weighted average giving more weight to recent prices",
                    "interpretation": "Faster than SMA. Used in MACD calculation. 12 & 26 EMA crossing is a trade signal.",
                    "settings": "Common: 9, 12, 26 days",
                    "limitations": "More sensitive – can generate false signals in sideways markets.",
                },
                "RSI – Relative Strength Index": {
                    "formula": "100 − 100 / (1 + Average Gain / Average Loss) over 14 periods",
                    "interpretation": ">70 = Overbought (consider selling). <30 = Oversold (consider buying). Divergence with price is powerful.",
                    "settings": "14-period standard",
                    "limitations": "In strong trends, RSI can stay overbought/oversold for extended periods.",
                },
                "MACD – Moving Average Convergence Divergence": {
                    "formula": "MACD Line = 12 EMA − 26 EMA. Signal = 9 EMA of MACD Line. Histogram = MACD − Signal",
                    "interpretation": "MACD crossing above signal line = bullish. Below = bearish. Histogram shows momentum strength.",
                    "settings": "12, 26, 9 (standard)",
                    "limitations": "Lagging. Poor in ranging/choppy markets.",
                },
                "Bollinger Bands": {
                    "formula": "Middle = 20 SMA. Upper/Lower = Middle ± 2× Standard Deviation",
                    "interpretation": "Price touching upper band = overbought zone. Lower band = oversold. Squeeze (narrow bands) precedes a breakout.",
                    "settings": "20-period, 2 SD",
                    "limitations": "Does not predict direction of breakout.",
                },
                "ATR – Average True Range": {
                    "formula": "14-period average of True Range (max of: H−L, |H−Prev Close|, |L−Prev Close|)",
                    "interpretation": "Measures volatility. High ATR = high volatility. Used to set dynamic stop losses (e.g., 1.5× ATR below entry).",
                    "settings": "14-period standard",
                    "limitations": "Not directional – only measures range/volatility.",
                },
                "Stochastic Oscillator": {
                    "formula": "%K = (Close − Lowest Low) / (Highest High − Lowest Low) × 100 over 14 periods. %D = 3-period SMA of %K.",
                    "interpretation": "%K > 80 = overbought. %K < 20 = oversold. %K crossing above %D = bullish signal.",
                    "settings": "14, 3, 3 (standard)",
                    "limitations": "Can stay extreme in trending markets.",
                },
            }
            for name, info in indicators.items():
                with st.expander(f"📊 {name}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Formula:** {info['formula']}")
                        st.markdown(f"**Settings:** {info['settings']}")
                    with c2:
                        st.markdown(f"**Interpretation:** {info['interpretation']}")
                        st.warning(f"⚠️ Limitation: {info['limitations']}")

        with tabs[2]:
            st.markdown("### 🕯️ Candlestick Patterns Guide")
            patterns_guide = {
                "Doji": {
                    "signal": "NEUTRAL / Reversal Warning",
                    "description": "Open ≈ Close. Very small body with wicks. Represents indecision between buyers and sellers.",
                    "action": "Wait for confirmation from the next candle before acting.",
                },
                "Hammer": {
                    "signal": "BULLISH Reversal",
                    "description": "Small body at top, long lower wick (≥2× body), minimal upper wick. Found at bottom of downtrend.",
                    "action": "Buy above the hammer's high with stop below the low.",
                },
                "Shooting Star": {
                    "signal": "BEARISH Reversal",
                    "description": "Small body at bottom, long upper wick, minimal lower wick. Found at top of uptrend. Mirror image of hammer.",
                    "action": "Sell/short below the shooting star's low with stop above the high.",
                },
                "Bullish Engulfing": {
                    "signal": "BULLISH Reversal (Strong)",
                    "description": "A large green candle completely engulfs the previous red candle. Shows buyers have overwhelmed sellers.",
                    "action": "Buy on confirmation; strong signal especially at support levels.",
                },
                "Bearish Engulfing": {
                    "signal": "BEARISH Reversal (Strong)",
                    "description": "A large red candle completely engulfs the previous green candle. Shows sellers have overwhelmed buyers.",
                    "action": "Sell/short on confirmation; strong signal at resistance levels.",
                },
                "Marubozu": {
                    "signal": "BULLISH (Green) / BEARISH (Red) Continuation",
                    "description": "Full body with no or minimal wicks. Shows overwhelming one-sided momentum.",
                    "action": "Trade in the direction of the Marubozu; tight stops as momentum is strong.",
                },
                "Harami": {
                    "signal": "Potential Reversal (Weak)",
                    "description": "Small candle contained within the range of the previous large candle. Suggests slowing momentum.",
                    "action": "Use as an early warning; wait for a third candle to confirm the reversal.",
                },
                "Piercing Line": {
                    "signal": "BULLISH Reversal (Medium)",
                    "description": "After a red candle, a green candle opens below the low and closes above the midpoint of the red candle.",
                    "action": "Bullish if it occurs after a downtrend; confirm with volume.",
                },
            }
            for name, info in patterns_guide.items():
                sig_color = "success" if "BULLISH" in info["signal"] else "error" if "BEARISH" in info["signal"] else "info"
                with st.expander(f"🕯️ {name} – {info['signal']}"):
                    if sig_color == "success":
                        st.success(f"Signal: {info['signal']}")
                    elif sig_color == "error":
                        st.error(f"Signal: {info['signal']}")
                    else:
                        st.info(f"Signal: {info['signal']}")
                    st.write(f"**What it looks like:** {info['description']}")
                    st.write(f"**Trading action:** {info['action']}")

        with tabs[3]:
            st.markdown("### 💰 Indian Stock Market Tax Guide (FY 2024-25)")

            st.markdown("#### 📅 Classification by Holding Period")
            tax_data = pd.DataFrame({
                "Asset Type": ["Listed Equity Shares", "Listed Equity Shares", "Equity Mutual Funds", "Equity Mutual Funds"],
                "Holding Period":["≤ 12 months", "> 12 months", "≤ 12 months", "> 12 months"],
                "Gain Type":["STCG", "LTCG", "STCG", "LTCG"],
                "Tax Rate":["15%", "10% (above ₹1 lakh)", "15%", "10% (above ₹1 lakh)"],
                "Indexation":["No", "No", "No", "No"],
            })
            st.dataframe(tax_data, width="stretch", hide_index=True)

            st.markdown("#### 💸 Transaction Taxes")
            charges = pd.DataFrame({
                "Charge":["STT (Delivery Buy)", "STT (Delivery Sell)", "STT (Intraday Sell)", "SEBI Charges", "Exchange Transaction Charges (NSE)", "GST on Brokerage"],
                "Rate":["0.1%", "0.1%", "0.025%", "₹10 per crore", "~0.00322%", "18% on brokerage"],
            })
            st.dataframe(charges, width="stretch", hide_index=True)

            st.info("💡 **LTCG Exemption:** The first ₹1,00,000 of long-term capital gains from equity is tax-free each financial year.")
            st.warning("⚠️ This guide is for educational purposes only. Consult a CA or tax advisor for personalised advice.")

        with tabs[4]:
            st.markdown("### 🎯 Trading Psychology")

            principles =[
                ("🧠 Cut Losses, Let Profits Run",
                 "The hardest rule to follow. Most traders do the opposite – holding losers hoping they recover and selling winners too early. Use a stop loss and stick to it."),
                ("📏 Always Define Risk Before Entry",
                 "Know your stop loss and position size before you enter a trade. Risk management is more important than being right about the direction."),
                ("🚫 Avoid Revenge Trading",
                 "After a loss, the urge to immediately recover it leads to larger, impulsive trades. Take a break. The market will always provide another opportunity."),
                ("📔 Keep a Trade Journal",
                 "Tracking your trades, reasoning, and emotions is the fastest way to improve. Patterns in your mistakes become obvious over time."),
                ("🎲 Probability Thinking",
                 "No strategy wins 100% of the time. Focus on having a positive expected value (Win Rate × Avg Win > Loss Rate × Avg Loss), not on any single trade."),
                ("💤 FOMO – Fear of Missing Out",
                 "Chasing stocks that have already moved significantly is a common trap. There is always another trade. Missing one trade is far less costly than entering a bad one."),
                ("⚖️ Position Sizing is Everything",
                 "A trader who risks 2% per trade can survive a 10-loss streak. One who risks 20% per trade cannot. Kelly Criterion and fixed-percent risk keep you in the game."),
                ("🔄 Process Over Outcome",
                 "A good process can produce a loss on any individual trade (variance). Judge yourself on following your rules, not on individual trade P&L."),
            ]
            for title, content in principles:
                with st.expander(title):
                    st.write(content)

        with tabs[5]:
            st.markdown("### 📊 Fundamental Analysis 101")

            st.markdown("#### 🔢 Key Ratios & What They Tell You")
            ratios_guide =[
                ("P/E Ratio", "Price / EPS", "Is the stock cheap or expensive relative to earnings?",
                 "Compare with sector P/E. High P/E may mean growth expectations or overvaluation."),
                ("P/B Ratio", "Price / Book Value per Share", "Am I paying more than the company's net assets are worth?",
                 "<1 may indicate undervaluation. Banking stocks are often valued on P/B."),
                ("ROE", "Net Profit / Shareholder Equity × 100", "How efficiently is management using shareholder money?",
                 ">15% is generally considered good. Compare across multiple years for consistency."),
                ("ROCE", "EBIT / Capital Employed × 100", "How efficiently is the company using all its capital?",
                 "Should be greater than the cost of debt. Stable or improving ROCE is positive."),
                ("Debt/Equity", "Total Debt / Total Equity", "How leveraged is the company?",
                 "<1 is generally safer. High D/E in cyclical industries is a red flag."),
                ("Current Ratio", "Current Assets / Current Liabilities", "Can the company pay short-term obligations?",
                 ">1.5 is comfortable. <1 means the company may struggle to meet near-term obligations."),
                ("Free Cash Flow", "Operating Cash Flow − Capex", "Is the company generating real cash after investments?",
                 "Consistently positive FCF is a strong quality indicator. Look for FCF > Net Profit."),
                ("Operating Margin", "EBIT / Revenue × 100", "How much profit from every rupee of revenue after operating costs?",
                 "Higher and stable margins indicate pricing power and efficient operations."),
            ]
            for ratio, formula, question, guide in ratios_guide:
                with st.expander(f"📌 {ratio}"):
                    st.markdown(f"**Formula:** `{formula}`")
                    st.markdown(f"**Question it answers:** {question}")
                    st.success(f"📊 How to use: {guide}")

            st.divider()
            st.markdown("#### 🔍 The 5-Step Fundamental Checklist")
            checklist =[
                "**1. Business Quality** – Does the company have a durable competitive advantage (moat)? Consistent revenue growth over 5+ years?",
                "**2. Management** – Promoter holding >50%? Low pledging? Clean corporate governance? Track record of capital allocation?",
                "**3. Financial Health** – Low or reducing debt? ROE & ROCE consistently >15%? Free cash flow positive?",
                "**4. Valuation** – P/E, P/B below sector averages or historical averages? PEG ratio < 1 (P/E / earnings growth rate)?",
                "**5. Future Catalysts** – Industry tailwinds? New product launches? Capacity expansion? Regulatory benefits?",
            ]
            for item in checklist:
                st.markdown(item)

    def show_multi_stock_comparison():
        st.header("🔀 Multi-Stock Comparison")
        tickers_raw = st.text_input("Enter comma-separated tickers (e.g. RELIANCE.NS, TCS.NS, INFY.NS)")
        if tickers_raw:
            tickers =[t.strip() for t in tickers_raw.split(',') if t.strip()]
            if st.button("Compare"):
                results =[]
                for t in tickers:
                    df = fetch_stock_data(t, '1y')
                    if df is not None and not df.empty:
                        ret = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                        results.append({'Ticker': t, 'Current Price': df['Close'].iloc[-1], '1Y Return %': round(ret, 2)})
                if results:
                    st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)
                else:
                    st.error("Could not fetch data for any of the tickers.")

    def show_corporate_actions():
        st.header("📅 Corporate Actions")
        ticker_raw = st.text_input("Enter Stock Symbol", placeholder="RELIANCE.NS")
        ticker = normalize_ticker(ticker_raw, st.session_state.exchange)
        if ticker and st.button("Fetch Corporate Actions"):
            try:
                stock = yf.Ticker(ticker)
                divs = stock.dividends
                splits = stock.splits
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("💰 Dividends")
                    if not divs.empty:
                        st.dataframe(divs.tail(20).reset_index(), width="stretch", hide_index=True)
                    else:
                        st.info("No dividend history found.")
                with col2:
                    st.subheader("✂️ Stock Splits")
                    if not splits.empty:
                        st.dataframe(splits.reset_index(), width="stretch", hide_index=True)
                    else:
                        st.info("No stock splits found.")
            except Exception as e:
                st.error(f"Error fetching corporate actions: {e}")

    def show_india_market_hub():
        render_glass_header("🇮🇳", "India Market Hub", "Official NSE/SEBI shortcuts and the core tools for daily market usage")
        st.caption("A single place for the links and workflows most Indian market users need every day.")
        now_ist = get_ist_time()

        st.subheader("Quick Usage")
        usage_steps = [
            "1. Check market status, holidays, and timings before the session.",
            "2. Use market data pages for gainers, losers, most active, and 52-week levels.",
            "3. Open the derivatives tools for option chain, OI, and payoff analysis.",
            "4. Review corporate filings, IPOs, and circulars for event-driven trades.",
            "5. Use SEBI/SMART ODR links for complaints and investor protection.",
        ]
        for step in usage_steps:
            st.markdown(f"- {step}")

        st.subheader("Market Operations")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### NSE")
            nse_links = {
                "Live Equity Market": "https://www.nseindia.com/market-data/live-equity-market",
                "Option Chain": "https://www.nseindia.com/option-chain",
                "Most Active Equities": "https://www.nseindia.com/market-data/most-active-equities",
                "Gainers/Losers": "https://www.nseindia.com/market-data/top-gainers-losers",
                "Price Band Hitters": "https://www.nseindia.com/market-data/upper-band-hitters",
                "52W High/Low": "https://www.nseindia.com/market-data/52-week-high-equity-market",
                "Daily Reports": "https://www.nseindia.com/all-reports",
                "Market Timings": "https://www.nseindia.com/market-data/market-timings",
                "Holidays": "https://www.nseindia.com/resources/exchange-communication-holidays",
            }
            for label, url in nse_links.items():
                st.markdown(f"- [{label}]({url})")
        with col2:
            st.markdown("#### SEBI / Investor Safety")
            sebi_links = {
                "SEBI Home": "https://www.sebi.gov.in/",
                "SCORES Complaint Registration": "https://scores.sebi.gov.in/en/investor-complaint",
                "SCORES Complaint Status": "https://scores.sebi.gov.in/complaint-status",
                "SMART ODR": "https://smartodr.in/login",
                "Investor Charter": "https://investor.sebi.gov.in/Campaign.html",
                "Investor Services Centres": "https://investor.sebi.gov.in/iscs_contacts.html",
            }
            for label, url in sebi_links.items():
                st.markdown(f"- [{label}]({url})")

        st.subheader("Core In-App Tools")
        tool_cols = st.columns(3)
        tools = [
            ("📊 Fundamental Analysis", "Reports, live ratios, valuation"),
            ("📈 Technical Analysis", "Indicators, charts, patterns"),
            ("🧮 Options Analyzer", "Fair value, Greeks, payoff"),
            ("📈 Futures & Options", "Derivatives basics and compare"),
            ("🔔 Alerts", "Price thresholds and triggers"),
            ("📚 Education Center", "Glossary, tax, and psychology"),
        ]
        for i, (title, desc) in enumerate(tools):
            with tool_cols[i % 3]:
                st.markdown(
                    f"""
                    <div class="glass-panel" style="padding:16px;min-height:96px;margin-bottom:10px;">
                        <div style="font-weight:700;color:#f8fafc;margin-bottom:4px;">{title}</div>
                        <div style="font-size:0.84rem;color:#aab4c6;line-height:1.4;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.subheader("Research Shortcuts")
        research_cols = st.columns(2)
        with research_cols[0]:
            st.markdown("- [NSE Research](https://www.nseindia.com/research/research-overview)")
            st.markdown("- [NSE Market Snapshot](https://www.nseindia.com/market-data/analysis-and-tools-capital-market-snapshot)")
            st.markdown("- [Corporate Filings](https://www.nseindia.com/companies-listing/corporate-filings-application)")
        with research_cols[1]:
            st.markdown("- [SEBI Departments](https://www.sebi.gov.in/)")
            st.markdown("- [Exchange Circulars](https://www.nseindia.com/resources/exchange-communication-circulars)")
            st.markdown("- [Market Holidays](https://www.nseindia.com/resources/exchange-communication-holidays)")
            st.markdown("- [Financial Education](https://www.nseindia.com/invest/investors-home)")

        st.subheader("Session Calendar")
        session_cols = st.columns(3)
        next_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        next_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        if now_ist >= next_close or now_ist.weekday() >= 5:
            next_session = next_open + timedelta(days=1 if now_ist.weekday() < 4 else (7 - now_ist.weekday()))
        elif now_ist < next_open:
            next_session = next_open
        else:
            next_session = next_close
        with session_cols[0]:
            st.markdown(f"<div class='glass-panel' style='padding:16px;'><div style='color:#aab4c6;font-size:0.8rem;'>Status</div><div style='font-size:1.05rem;font-weight:700;color:#f8fafc;'>{'Market Open' if is_market_open() else 'Market Closed'}</div></div>", unsafe_allow_html=True)
        with session_cols[1]:
            st.markdown(f"<div class='glass-panel' style='padding:16px;'><div style='color:#aab4c6;font-size:0.8rem;'>Next Session Event</div><div style='font-size:1.05rem;font-weight:700;color:#f8fafc;'>{'Close' if is_market_open() else 'Open'}</div></div>", unsafe_allow_html=True)
        with session_cols[2]:
            st.markdown(f"<div class='glass-panel' style='padding:16px;'><div style='color:#aab4c6;font-size:0.8rem;'>Next Time Point</div><div style='font-size:1.05rem;font-weight:700;color:#f8fafc;'>{next_session.strftime('%I:%M %p')}</div></div>", unsafe_allow_html=True)

        st.subheader("Pre-Market Shortcuts")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            if st.button("🌅 Pre-Market Prep", use_container_width=True):
                navigate_to("🌅 Pre-Market Prep")
        with p2:
            if st.button("⚡ Pre-Open", use_container_width=True):
                navigate_to("⚡ Pre-Open Session (9–9:15 AM)")
        with p3:
            if st.button("🔴 Live Market", use_container_width=True):
                navigate_to("🔴 Live Market (After 9:15 AM)")
        with p4:
            if st.button("📈 Technical", use_container_width=True):
                navigate_to("📈 Technical Analysis")

        st.subheader("Intraday Scanner Shortcuts")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            if st.button("🔍 Market Scanner", use_container_width=True):
                navigate_to("🔍 Stock Screener")
        with s2:
            if st.button("📱 Quick Trade Setup", use_container_width=True):
                navigate_to("📱 Quick Trade Setup")
        with s3:
            if st.button("🎯 Selection Engine", use_container_width=True):
                navigate_to("🎯 Stock Selection Engine")
        with s4:
            if st.button("📈 Backtester", use_container_width=True):
                navigate_to("📈 Strategy Backtester")

        st.subheader("Live Option Chain")
        oc_symbol = st.selectbox("Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0)
        if st.button("Fetch Live Option Chain", type="primary"):
            oc_df, oc_meta = fetch_nse_option_chain(oc_symbol)
            if oc_df is None or oc_df.empty:
                st.warning(f"Could not load the live option chain for {oc_symbol}.")
                if oc_meta.get("error"):
                    st.caption(oc_meta["error"])
            else:
                atm = oc_meta.get("underlying")
                st.metric("Underlying", format_indian_number(atm, True) if atm else "N/A")
                st.caption(f"Last updated: {oc_meta.get('timestamp', 'N/A')} | Expiries: {', '.join(oc_meta.get('expiry_dates', [])[:3])}")
                view = oc_df.copy()
                if atm:
                    view["Distance"] = (view["Strike"] - atm).abs()
                    view = view.sort_values("Distance").head(15).sort_values("Strike")
                st.dataframe(view, use_container_width=True, hide_index=True)

    def show_futures_options():
        render_glass_header("📈", "Futures & Options", "Derivatives basics and quick compare")
        st.caption("Quick reference for derivatives basics and the built-in pricing tools.")

        tabs = st.tabs(["Futures", "Options", "Quick Compare"])
        with tabs[0]:
            st.info("A futures contract is a binding agreement to buy or sell at a fixed price on a future date.")
            futures_df = pd.DataFrame({
                "Topic": ["Payoff", "Margin", "Risk", "Use case"],
                "Futures": ["Linear one-to-one exposure", "Margin-based", "High leverage", "Hedging or directional trading"],
            })
            st.dataframe(futures_df, use_container_width=True, hide_index=True)

        with tabs[1]:
            st.info("Options give the right, not the obligation, to buy or sell at the strike price before expiry.")
            options_df = pd.DataFrame({
                "Topic": ["Call", "Put", "Buyer risk", "Seller risk"],
                "Options": ["Bullish right to buy", "Bearish right to sell", "Limited to premium", "Can be large or unlimited"],
            })
            st.dataframe(options_df, use_container_width=True, hide_index=True)

        with tabs[2]:
            compare = pd.DataFrame({
                "Feature": ["Obligation", "Upfront cost", "Risk profile", "Ideal use"],
                "Futures": ["Yes", "Margin", "Symmetric", "Hedge / leverage"],
                "Options": ["No for buyer", "Premium", "Asymmetric", "Defined-risk bets"],
            })
            st.dataframe(compare, use_container_width=True, hide_index=True)
            st.success("Use the 🧮 Options Analyzer page for pricing, Greeks, and payoff charts.")

    def show_faq():
        render_glass_header("❓", "FAQ", "Fast answers to common platform questions")
        st.caption("Short answers to common platform and trading questions.")

        faq_items = [
            ("How do I use this app?", "Pick a tool from the sidebar, enter a ticker or values, and run the analysis."),
            ("Where is the options tool?", "Go to Tools → Futures & Options or the 🧮 Options Analyzer under Analysis."),
            ("Why does the theme look different?", "I updated it to a cleaner slate layout with softer colors and spacing."),
            ("Is this data live?", "Most market data is fetched live from yfinance; availability depends on the source."),
            ("Can I save my data?", "Use the database section in the sidebar or export CSV files before closing."),
        ]
        for q, a in faq_items:
            with st.expander(q):
                st.write(a)

    # ── PRE-MARKET PREP ─────────────────────────────────────
    def show_pre_market_prep():
        st.header("🌅 Pre-Market Preparation (Before 9:00 AM)")
        st.markdown("**Your complete pre-market checklist for Indian markets — complete this before 9:15 AM IST**")

        def reset_premarket_checklist(keys):
            for key in keys:
                st.session_state[key] = False

        now_ist = get_ist_time()
        ist_time_str = now_ist.strftime("%I:%M %p IST")
        if now_ist.hour < 9:
            st.success(f"🕐 {ist_time_str} — ✅ Pre-market window. Market opens in {9*60 - now_ist.hour*60 - now_ist.minute} minutes.")
        elif now_ist.hour == 9 and now_ist.minute < 15:
            st.warning(f"🕐 {ist_time_str} — ⚡ Pre-open session active! Market opens in {15 - now_ist.minute} minutes.")
        else:
            st.info(f"🕐 {ist_time_str} — Market is {'OPEN' if is_market_open() else 'CLOSED'}. Use this tab for tomorrow's prep.")

        tabs = st.tabs([
            "🔍 Step 1: Find Momentum Stocks",
            "📊 Step 2: Confirm Trend with TA",
            "🌏 Step 3: Market Trend & Indices",
            "📋 Step 4: Watchlist Builder",
            "✅ Prep Checklist",
        ])

        # ── STEP 1: FIND MOMENTUM STOCKS ──
        with tabs[0]:
            st.markdown("### 🔍 Step 1: Identify Trending Stocks with Momentum")
            st.info("Use this section to scan for stocks showing strong momentum using Open Interest data and price action from NSE.")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("#### 📡 Momentum Scanner — Nifty 50 Universe")
                momentum_period = st.selectbox("Scan Period", ["5d", "1mo", "3mo"], index=1, key="mom_period")
                min_return = st.slider("Min Return % threshold", 1.0, 20.0, 5.0, 0.5, key="mom_min_ret")
                min_volume_mult = st.slider("Min Volume vs 20-day avg (×)", 1.0, 5.0, 1.5, 0.1, key="mom_vol")
                if st.button("🚀 Run Momentum Scan", use_container_width=True, key="run_mom_scan"):
                    results = []
                    ticker_list = list(NIFTY_50_DATA.items())
                    progress = st.progress(0)
                    status_text = st.empty()
                    for i, (ticker, (name, sector)) in enumerate(ticker_list):
                        progress.progress((i + 1) / len(ticker_list))
                        status_text.text(f"Scanning {name}…")
                        try:
                            df = fetch_stock_data(ticker, momentum_period)
                            if df is None or len(df) < 5:
                                continue
                            ret = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100
                            avg_vol = df["Volume"].mean()
                            last_vol = df["Volume"].iloc[-1]
                            vol_mult = last_vol / avg_vol if avg_vol > 0 else 0
                            latest = df.iloc[-1]
                            rsi = latest.get("RSI", 50)
                            above_sma20 = latest["Close"] > latest.get("SMA_20", latest["Close"])
                            above_sma50 = latest["Close"] > latest.get("SMA_50", latest["Close"])
                            if ret >= min_return and vol_mult >= min_volume_mult:
                                results.append({
                                    "Ticker": ticker,
                                    "Name": name,
                                    "Sector": sector,
                                    "Return %": round(ret, 2),
                                    "Volume (×avg)": round(vol_mult, 2),
                                    "RSI": round(rsi, 1) if not pd.isna(rsi) else "-",
                                    "Price": format_indian_number(latest["Close"], True),
                                    "Above SMA20": "✅" if above_sma20 else "❌",
                                    "Above SMA50": "✅" if above_sma50 else "❌",
                                })
                        except Exception:
                            continue
                    progress.empty()
                    status_text.empty()
                    if results:
                        st.success(f"✅ Found {len(results)} momentum stocks")
                        results_df = pd.DataFrame(results).sort_values("Return %", ascending=False)
                        st.dataframe(results_df, use_container_width=True, hide_index=True)
                        # Add to watchlist
                        if st.button("📋 Add all to Watchlist", key="mom_to_watchlist"):
                            for r in results:
                                ticker = r["Ticker"]
                                if ticker not in st.session_state.watchlist:
                                    st.session_state.watchlist.append(ticker)
                            st.success(f"Added {len(results)} stocks to watchlist!")
                    else:
                        st.warning("No stocks met the momentum criteria. Try lowering thresholds.")

                st.divider()
                st.markdown("#### ✅ Buy Signal Checker")
                buy_mode = st.selectbox("Mode", ["Conservative", "Balanced", "Aggressive"], index=1, key="premarket_buy_mode")
                default_min = 7 if buy_mode == "Conservative" else 6 if buy_mode == "Balanced" else 4
                min_buy_score = st.slider("Minimum Score", 1, 10, default_min, key="premarket_buy_min_score")

                if st.button("🟢 Check for Buys", use_container_width=True, key="premarket_buy_check"):
                    buy_rows = []
                    ticker_list = list(NIFTY_50_DATA.items())
                    progress2 = st.progress(0)
                    status2 = st.empty()
                    for i, (ticker, (name, sector)) in enumerate(ticker_list):
                        progress2.progress((i + 1) / len(ticker_list))
                        status2.text(f"Scoring {name}…")
                        try:
                            dfb = fetch_stock_data(ticker, "6mo")
                            if dfb is None or dfb.empty or len(dfb) < 60:
                                continue
                            latest = dfb.iloc[-1]
                            close = latest.get("Close")
                            sma20 = latest.get("SMA_20")
                            sma50 = latest.get("SMA_50")
                            rsi = latest.get("RSI")
                            macd = latest.get("MACD")
                            macd_sig = latest.get("MACD_Signal")
                            vol_ratio = (dfb["Volume"].iloc[-1] / (dfb["Volume"].rolling(20).mean().iloc[-1] or 1))
                            ret20 = ((dfb["Close"].iloc[-1] / dfb["Close"].iloc[-21]) - 1) * 100 if len(dfb) > 21 else 0

                            score = 0
                            reasons = []
                            if pd.notna(close) and pd.notna(sma20) and pd.notna(sma50) and close > sma20 > sma50:
                                score += 4 if buy_mode == "Conservative" else 3
                                reasons.append("Price > SMA20 > SMA50")
                            if pd.notna(rsi):
                                if buy_mode == "Conservative" and 55 <= rsi <= 68:
                                    score += 2; reasons.append(f"RSI {rsi:.1f}")
                                elif buy_mode == "Balanced" and 52 <= rsi <= 72:
                                    score += 2; reasons.append(f"RSI {rsi:.1f}")
                                elif buy_mode == "Aggressive" and 48 <= rsi <= 78:
                                    score += 1; reasons.append(f"RSI {rsi:.1f}")
                            if pd.notna(macd) and pd.notna(macd_sig) and macd > macd_sig:
                                score += 2 if buy_mode != "Aggressive" else 1
                                reasons.append("MACD bullish")
                            vol_cutoff = 1.5 if buy_mode == "Conservative" else 1.3 if buy_mode == "Balanced" else 1.1
                            if vol_ratio >= vol_cutoff:
                                score += 2 if buy_mode != "Aggressive" else 1
                                reasons.append(f"Vol {vol_ratio:.2f}x")
                            mom_cutoff = 6 if buy_mode == "Conservative" else 4 if buy_mode == "Balanced" else 2
                            if ret20 >= mom_cutoff:
                                score += 1
                                reasons.append(f"20D {ret20:.2f}%")

                            if score >= min_buy_score:
                                buy_rows.append({
                                    "Ticker": ticker,
                                    "Name": name,
                                    "Sector": sector,
                                    "Price": format_indian_number(close, True),
                                    "Score": score,
                                    "RSI": round(float(rsi), 1) if pd.notna(rsi) else "-",
                                    "20D Return %": round(float(ret20), 2),
                                    "Volume x": round(float(vol_ratio), 2),
                                    "Signal": "BUY",
                                    "Reasons": ", ".join(reasons),
                                })
                        except Exception:
                            continue

                    progress2.empty()
                    status2.empty()
                    if buy_rows:
                        out = pd.DataFrame(buy_rows).sort_values(["Score", "20D Return %"], ascending=[False, False])
                        st.success(f"✅ Found {len(out)} BUY candidates")
                        st.dataframe(out, use_container_width=True, hide_index=True)
                        if st.button("📋 Add BUY candidates to Watchlist", key="premarket_buy_to_watch"):
                            for t in out["Ticker"].tolist():
                                if t not in st.session_state.watchlist:
                                    st.session_state.watchlist.append(t)
                            st.success("Added BUY candidates to watchlist.")
                    else:
                        st.warning("No BUY candidates found for the selected mode and minimum score.")

            with col2:
                st.markdown("#### 🔗 External Scanner Links")
                st.markdown("""
                | Tool | Purpose | Link |
                |------|---------|------|
                | **Streak** | Algo scanner, backtester | [streak.tech](https://streak.tech) |
                | **ChartInk** | Free NSE screener | [chartink.com](https://chartink.com) |
                | **NSE India** | OI data, FII/DII, bulk deals | [nseindia.com](https://www.nseindia.com) |
                | **Trendlyne** | Momentum + fundamentals | [trendlyne.com](https://trendlyne.com) |
                | **Investing.com India** | Global + Indian indices | [investing.com](https://in.investing.com) |
                """, unsafe_allow_html=False)

                st.markdown("#### 📈 Open Interest Interpretation")
                st.markdown("""
                | Price | OI | Interpretation |
                |-------|-----|---------------|
                | ⬆️ Rising | ⬆️ Rising | **Long build-up** — Bullish |
                | ⬆️ Rising | ⬇️ Falling | **Short covering** — Bullish |
                | ⬇️ Falling | ⬆️ Rising | **Short build-up** — Bearish |
                | ⬇️ Falling | ⬇️ Falling | **Long unwinding** — Bearish |
                """)

                st.markdown("#### 📰 NSE Resources")
                nse_links = {
                    "Bulk Deals": "https://www.nseindia.com/market-data/bulk-block-deals",
                    "FII/DII Data": "https://www.nseindia.com/reports-indices-data/fii-dii-activity",
                    "OI Data (F&O)": "https://www.nseindia.com/option-chain",
                    "Most Active Stocks": "https://www.nseindia.com/market-data/most-active-securities",
                    "Circuit Breakers Today": "https://www.nseindia.com/market-data/live-market-indices",
                }
                for label, url in nse_links.items():
                    st.markdown(f"• [{label}]({url})")

        # ── STEP 2: CONFIRM TREND WITH TA ──
        with tabs[1]:
            st.markdown("### 📊 Step 2: Confirm Trend with Technical Analysis & Draw Trend Lines")
            st.markdown("Enter up to 5 stocks from your momentum scan and confirm their technical trend before market open.")

            pre_tickers_raw = st.text_input("Enter comma-separated tickers to analyse", 
                                             value=", ".join(st.session_state.watchlist[:5]) if st.session_state.watchlist else "",
                                             placeholder="RELIANCE.NS, TCS.NS, INFY.NS",
                                             key="pre_ta_tickers")
            confirm_period = st.selectbox("Chart Period", ["1mo", "3mo", "6mo", "1y"], index=1, key="pre_ta_period")

            if pre_tickers_raw and st.button("🔎 Analyse Trends", use_container_width=True, key="pre_ta_run"):
                tickers_to_check = [t.strip() for t in pre_tickers_raw.split(",") if t.strip()][:5]
                for t in tickers_to_check:
                    ticker = normalize_ticker(t, st.session_state.exchange)
                    df = fetch_stock_data(ticker, confirm_period)
                    if df is None or df.empty:
                        st.error(f"Could not fetch data for {ticker}")
                        continue

                    latest = df.iloc[-1]
                    price = latest["Close"]
                    sma20 = latest.get("SMA_20", price)
                    sma50 = latest.get("SMA_50", price)
                    sma200_col = [c for c in df.columns if "200" in c]
                    sma200 = latest[sma200_col[0]] if sma200_col else None
                    rsi = latest.get("RSI", 50)
                    macd = latest.get("MACD", 0)
                    macd_sig = latest.get("MACD_Signal", 0)

                    # Trend determination
                    signals = []
                    bullish_count = 0
                    if price > sma20:
                        signals.append(("Above SMA 20", "✅ Bullish")); bullish_count += 1
                    else:
                        signals.append(("Below SMA 20", "❌ Bearish"))
                    if price > sma50:
                        signals.append(("Above SMA 50", "✅ Bullish")); bullish_count += 1
                    else:
                        signals.append(("Below SMA 50", "❌ Bearish"))
                    if sma200 is not None and not pd.isna(sma200):
                        if price > sma200:
                            signals.append(("Above SMA 200", "✅ Bullish")); bullish_count += 1
                        else:
                            signals.append(("Below SMA 200", "❌ Bearish"))
                    if not pd.isna(rsi):
                        if rsi > 50:
                            signals.append((f"RSI {rsi:.1f} > 50", "✅ Bullish")); bullish_count += 1
                        else:
                            signals.append((f"RSI {rsi:.1f} < 50", "❌ Bearish"))
                    if not pd.isna(macd):
                        if macd > macd_sig:
                            signals.append(("MACD above Signal", "✅ Bullish")); bullish_count += 1
                        else:
                            signals.append(("MACD below Signal", "❌ Bearish"))

                    trend_pct = bullish_count / len(signals) * 100
                    trend_color = "#00d084" if trend_pct >= 60 else "#ffa500" if trend_pct >= 40 else "#ff4444"
                    trend_label = "UPTREND" if trend_pct >= 60 else "SIDEWAYS" if trend_pct >= 40 else "DOWNTREND"

                    with st.expander(f"📊 {ticker} — {trend_label} ({bullish_count}/{len(signals)} signals bullish)", expanded=True):
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.markdown(f"""<div class="metric-card">
                                <div style="color:#888;">Trend</div>
                                <div style="color:{trend_color};font-size:1.6rem;font-weight:700;">{trend_label}</div>
                                <div style="color:#888;">Current Price: {format_indian_number(price, True)}</div>
                            </div>""", unsafe_allow_html=True)
                            for sig_label, sig_val in signals:
                                icon = "✅" if "✅" in sig_val else "❌"
                                st.write(f"{icon} {sig_label}")

                            # Trend line support/resistance
                            recent = df.tail(20)
                            support = recent["Low"].min()
                            resistance = recent["High"].max()
                            st.markdown("---")
                            st.metric("📉 20-day Support", format_indian_number(support, True))
                            st.metric("📈 20-day Resistance", format_indian_number(resistance, True))
                            pct_to_res = ((resistance - price) / price * 100)
                            pct_to_sup = ((price - support) / price * 100)
                            st.caption(f"Room to resistance: {pct_to_res:.1f}%  |  Buffer to support: {pct_to_sup:.1f}%")

                        with c2:
                            if PLOTLY_AVAILABLE:
                                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                                    row_heights=[0.7, 0.3], vertical_spacing=0.05)
                                fig.add_trace(go.Candlestick(x=df.index, open=df["Open"],
                                    high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
                                if "SMA_20" in df.columns:
                                    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA20",
                                        line=dict(color="#ffa500", width=1)), row=1, col=1)
                                if "SMA_50" in df.columns:
                                    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA50",
                                        line=dict(color="#00d4ff", width=1)), row=1, col=1)
                                if sma200_col:
                                    fig.add_trace(go.Scatter(x=df.index, y=df[sma200_col[0]], name="SMA200",
                                        line=dict(color="#ff4444", width=1)), row=1, col=1)
                                fig.add_hline(y=support, line_color="#ff4444", line_dash="dot",
                                    annotation_text="Support", row=1, col=1)
                                fig.add_hline(y=resistance, line_color="#00d084", line_dash="dot",
                                    annotation_text="Resistance", row=1, col=1)
                                if "RSI" in df.columns:
                                    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                                        line=dict(color="#a78bfa")), row=2, col=1)
                                    fig.add_hline(y=70, line_color="#ff4444", line_dash="dash", row=2, col=1)
                                    fig.add_hline(y=30, line_color="#00d084", line_dash="dash", row=2, col=1)
                                fig.update_layout(template="plotly_dark", height=400, showlegend=True,
                                    margin=dict(l=0, r=0, t=20, b=0))
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Install plotly for charts: pip install plotly")

        # ── STEP 3: MARKET TREND & GLOBAL INDICES ──
        with tabs[2]:
            st.markdown("### 🌏 Step 3: Identify Market Trend")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📰 Market Trend — Follow News")
                st.markdown("Quick links to check pre-market news:")
                news_links = {
                    "Economic Times Markets": "https://economictimes.indiatimes.com/markets",
                    "MoneyControl": "https://www.moneycontrol.com/news/business/markets/",
                    "NSE Circulars": "https://www.nseindia.com/regulations/listing-compliance/nse-market-updates",
                    "RBI Announcements": "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
                    "SEBI Orders": "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=2&ssid=15&smid=0",
                    "SGX Nifty / Gift Nifty": "https://www.nseindia.com/market-data/giftCity",
                }
                for label, url in news_links.items():
                    st.markdown(f"• [{label}]({url})")

                st.divider()
                st.markdown("#### 🎯 Key Events to Watch")
                st.markdown("""
                - **RBI MPC decisions** — affect all banking + rate-sensitive stocks
                - **Inflation data (CPI/WPI)** — released monthly by MOSPI
                - **US Fed meetings** — impacts FII flows into India
                - **Quarterly earnings season** — Apr–May and Oct–Nov
                - **Budget session** — February each year, major sector moves
                - **Global crude oil price** — impacts OMCs, Aviation, Paints, Tyres
                - **INR/USD exchange rate** — affects IT exporters and importers
                """)

            with col2:
                st.markdown("#### 📊 Index Charts — Market Trend at a Glance")
                index_tickers = {
                    "NIFTY 50": "^NSEI",
                    "SENSEX": "^BSESN",
                    "NIFTY Bank": "^NSEBANK",
                    "NIFTY IT": "^CNXIT",
                    "NIFTY Midcap 100": "^CNXMID",
                }
                index_period = st.selectbox("Period", ["5d", "1mo", "3mo"], index=1, key="idx_period")
                selected_index = st.selectbox("Select Index", list(index_tickers.keys()), key="idx_sel")
                if st.button("📈 Load Index Chart", use_container_width=True, key="idx_load"):
                    idx_ticker = index_tickers[selected_index]
                    try:
                        idx_df = yf.download(idx_ticker, period=index_period, progress=False)
                        idx_df = flatten_yf_columns(idx_df)
                        if idx_df is not None and not idx_df.empty:
                            if PLOTLY_AVAILABLE:
                                fig = go.Figure()
                                fig.add_trace(go.Candlestick(
                                    x=idx_df.index, open=idx_df["Open"], high=idx_df["High"],
                                    low=idx_df["Low"], close=idx_df["Close"], name=selected_index
                                ))
                                ret_pct = ((idx_df["Close"].iloc[-1] - idx_df["Close"].iloc[0]) / idx_df["Close"].iloc[0]) * 100
                                fig.update_layout(template="plotly_dark", height=380,
                                    title=f"{selected_index} — {ret_pct:+.2f}% over {index_period}")
                                st.plotly_chart(fig, use_container_width=True)
                                # Quick stats
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.metric("Current", f"{idx_df['Close'].iloc[-1]:,.0f}")
                                with c2:
                                    st.metric("Period High", f"{idx_df['High'].max():,.0f}")
                                with c3:
                                    st.metric("Period Low", f"{idx_df['Low'].min():,.0f}")
                            else:
                                st.line_chart(idx_df["Close"])
                        else:
                            st.error("Could not load index data.")
                    except Exception as e:
                        st.error(f"Error loading index: {e}")

            st.divider()
            st.markdown("#### 🌐 Global Indices — Follow Global Markets")
            if st.button("🌐 Fetch Global Indices Snapshot", use_container_width=True, key="global_idx"):
                global_indices = {
                    "S&P 500 (US)": "^GSPC",
                    "NASDAQ (US)": "^IXIC",
                    "Dow Jones (US)": "^DJI",
                    "FTSE 100 (UK)": "^FTSE",
                    "Nikkei 225 (Japan)": "^N225",
                    "Hang Seng (HK)": "^HSI",
                    "Shanghai Composite": "000001.SS",
                    "DAX (Germany)": "^GDAXI",
                    "SGX Nifty": "^NSEI",
                    "Crude Oil (WTI)": "CL=F",
                    "Gold": "GC=F",
                    "USD/INR": "INR=X",
                }
                rows = []
                progress2 = st.progress(0)
                items = list(global_indices.items())
                for i, (name, sym) in enumerate(items):
                    progress2.progress((i + 1) / len(items))
                    try:
                        data = yf.download(sym, period="2d", progress=False)
                        data = flatten_yf_columns(data)
                        if data is not None and len(data) >= 2:
                            prev_close = data["Close"].iloc[-2]
                            last_close = data["Close"].iloc[-1]
                            chg = last_close - prev_close
                            chg_pct = (chg / prev_close) * 100
                            rows.append({
                                "Index": name,
                                "Last": f"{last_close:.2f}",
                                "Change": f"{chg:+.2f}",
                                "Change %": f"{chg_pct:+.2f}%",
                                "Sentiment": "🟢 Positive" if chg_pct > 0.3 else "🔴 Negative" if chg_pct < -0.3 else "🟡 Neutral",
                            })
                    except Exception:
                        continue
                progress2.empty()
                if rows:
                    global_df = pd.DataFrame(rows)
                    positive = sum(1 for r in rows if "Positive" in r["Sentiment"])
                    negative = sum(1 for r in rows if "Negative" in r["Sentiment"])
                    total = len(rows)
                    global_sentiment = "🟢 Globally Positive" if positive > total * 0.6 else \
                                       "🔴 Globally Negative" if negative > total * 0.6 else "🟡 Mixed Global Sentiment"
                    st.markdown(f"### Overall Global Sentiment: {global_sentiment}")
                    st.dataframe(global_df, use_container_width=True, hide_index=True)

                    if "Crude Oil (WTI)" in [r["Index"] for r in rows]:
                        crude_row = next((r for r in rows if r["Index"] == "Crude Oil (WTI)"), None)
                        if crude_row:
                            crude_chg = float(crude_row["Change %"].replace("%", ""))
                            if crude_chg > 2:
                                st.warning("🛢️ Crude oil rising >2% — Watch: OMCs (IOC, BPCL), Aviation (IndiGo), Paints (Asian Paints)")
                            elif crude_chg < -2:
                                st.success("🛢️ Crude oil falling >2% — Positive for: OMCs, Aviation, Tyres, Paints")
                else:
                    st.error("Could not load global indices.")

        # ── STEP 4: WATCHLIST BUILDER ──
        with tabs[3]:
            st.markdown("### 📋 Step 4: Build Your Watchlist & Note Breakout Points")
            st.info("Add as many companies as you can. Note the key price levels — breakout, support, and stop loss — before market opens.")

            with st.expander("➕ Add Stock to Pre-Market Watchlist", expanded=True):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    new_ticker = st.text_input("Stock Ticker", placeholder="RELIANCE.NS", key="wl_ticker")
                with col2:
                    breakout_price = st.number_input("Breakout Price (₹)", min_value=0.0, step=0.5, key="wl_breakout")
                with col3:
                    stop_loss_price = st.number_input("Stop Loss (₹)", min_value=0.0, step=0.5, key="wl_sl")
                
                col4, col5, col6 = st.columns([2, 1, 1])
                with col4:
                    trade_bias = st.radio("Bias", ["Long", "Short", "Watch Only"], horizontal=True, key="wl_bias")
                with col5:
                    target_price = st.number_input("Target (₹)", min_value=0.0, step=0.5, key="wl_target")
                with col6:
                    notes_text = st.text_input("Notes", placeholder="Pattern, reason…", key="wl_notes")

                if st.button("📌 Add to Watchlist", use_container_width=True, key="wl_add"):
                    if new_ticker:
                        ticker_norm = normalize_ticker(new_ticker, st.session_state.exchange)
                        entry = {
                            "ticker": ticker_norm,
                            "breakout": breakout_price,
                            "stop_loss": stop_loss_price,
                            "target": target_price,
                            "bias": trade_bias,
                            "notes": notes_text,
                            "added_at": get_ist_time().strftime("%Y-%m-%d %H:%M"),
                        }
                        if "pre_watchlist" not in st.session_state:
                            st.session_state.pre_watchlist = []
                        # Avoid duplicates
                        existing = [w["ticker"] for w in st.session_state.pre_watchlist]
                        if ticker_norm not in existing:
                            st.session_state.pre_watchlist.append(entry)
                            if ticker_norm not in st.session_state.watchlist:
                                st.session_state.watchlist.append(ticker_norm)
                            st.success(f"✅ {ticker_norm} added to pre-market watchlist!")
                        else:
                            st.warning(f"{ticker_norm} already in watchlist.")
                    else:
                        st.error("Please enter a ticker.")

            if "pre_watchlist" not in st.session_state:
                st.session_state.pre_watchlist = []

            if st.session_state.pre_watchlist:
                st.markdown("### 📊 Today's Pre-Market Watchlist")
                if st.button("🔄 Refresh Live Prices", use_container_width=True, key="wl_refresh"):
                    for item in st.session_state.pre_watchlist:
                        try:
                            df = fetch_stock_data(item["ticker"], "5d")
                            if df is not None and not df.empty:
                                item["live_price"] = df["Close"].iloc[-1]
                                item["day_chg_pct"] = ((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100) if len(df) >= 2 else 0
                        except Exception:
                            pass
                    st.rerun()

                for i, item in enumerate(st.session_state.pre_watchlist):
                    live_price = item.get("live_price", None)
                    day_chg = item.get("day_chg_pct", None)
                    bias_color = "#00d084" if item["bias"] == "Long" else "#ff4444" if item["bias"] == "Short" else "#a78bfa"
                    price_str = format_indian_number(live_price, True) if live_price else "–"
                    chg_str = f"{day_chg:+.2f}%" if day_chg is not None else "–"

                    rr = 0
                    if item["bias"] == "Long" and item["target"] > 0 and item["stop_loss"] > 0 and item["breakout"] > 0:
                        reward = item["target"] - item["breakout"]
                        risk = item["breakout"] - item["stop_loss"]
                        rr = reward / risk if risk > 0 else 0
                    elif item["bias"] == "Short" and item["target"] > 0 and item["stop_loss"] > 0 and item["breakout"] > 0:
                        reward = item["breakout"] - item["target"]
                        risk = item["stop_loss"] - item["breakout"]
                        rr = reward / risk if risk > 0 else 0

                    # Check if price is near breakout
                    alert_text = ""
                    if live_price and item["breakout"] > 0:
                        pct_from_bo = (item["breakout"] - live_price) / live_price * 100
                        if abs(pct_from_bo) < 0.5:
                            alert_text = "⚡ AT BREAKOUT"
                        elif 0 < pct_from_bo < 2:
                            alert_text = f"🔔 {pct_from_bo:.1f}% to breakout"

                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"""<div class="metric-card" style="border-left: 4px solid {bias_color};">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <span style="font-size:1.1rem;font-weight:700;color:{bias_color};">{item['ticker']}</span>
                                    <span style="margin-left:12px;background:{bias_color}22;color:{bias_color};padding:2px 8px;border-radius:12px;font-size:0.8rem;">{item['bias']}</span>
                                    {f'<span style="margin-left:8px;color:#ffd700;font-size:0.85rem;">{alert_text}</span>' if alert_text else ''}
                                </div>
                                <div style="text-align:right;">
                                    <span style="font-size:1.2rem;color:#fff;">{price_str}</span>
                                    <span style="margin-left:8px;color:{'#00d084' if day_chg and day_chg > 0 else '#ff4444'};">{chg_str}</span>
                                </div>
                            </div>
                            <div style="display:flex;gap:24px;margin-top:8px;font-size:0.85rem;color:#aaa;">
                                <span>🎯 Breakout: <b style="color:#fff;">{format_indian_number(item['breakout'], True) if item['breakout'] else '–'}</b></span>
                                <span>🛑 SL: <b style="color:#ff4444;">{format_indian_number(item['stop_loss'], True) if item['stop_loss'] else '–'}</b></span>
                                <span>🏁 Target: <b style="color:#00d084;">{format_indian_number(item['target'], True) if item['target'] else '–'}</b></span>
                                <span>R:R: <b style="color:{'#00d084' if rr >= 2 else '#ffa500' if rr >= 1.5 else '#888'};">{'1:'+str(round(rr,1)) if rr > 0 else '–'}</b></span>
                            </div>
                            {f'<div style="margin-top:6px;color:#888;font-size:0.8rem;">📝 {item["notes"]}</div>' if item.get('notes') else ''}
                        </div>""", unsafe_allow_html=True)
                    with col2:
                        if st.button("🗑️ Remove", key=f"rem_wl_{i}"):
                            st.session_state.pre_watchlist.pop(i)
                            st.rerun()

                st.divider()
                # Export watchlist
                wl_df = pd.DataFrame([{
                    "Ticker": w["ticker"],
                    "Bias": w["bias"],
                    "Breakout (₹)": w["breakout"],
                    "Stop Loss (₹)": w["stop_loss"],
                    "Target (₹)": w["target"],
                    "R:R": round((w["target"] - w["breakout"]) / (w["breakout"] - w["stop_loss"]), 2)
                             if w["bias"] == "Long" and w["breakout"] > w["stop_loss"] > 0 and w["target"] > 0 else "-",
                    "Notes": w["notes"],
                    "Added": w["added_at"],
                } for w in st.session_state.pre_watchlist])
                csv = wl_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Export Watchlist CSV",
                    data=csv,
                    file_name=f"pre_market_watchlist_{get_ist_time().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="wl_export"
                )
            else:
                st.info("Your watchlist is empty. Add stocks above or run the Momentum Scanner in Step 1.")

        # ── STEP 5: CHECKLIST ──
        with tabs[4]:
            st.markdown("### ✅ Pre-Market Preparation Checklist")
            st.markdown("Track your daily preparation. Aim to complete this by **9:00 AM IST**.")

            checklist_items = [
                ("🔍 Stock Scanning", [
                    "Run momentum scanner on Nifty 50 universe",
                    "Check Streak/ChartInk for sector breakouts",
                    "Review NSE bulk deals and block deals",
                    "Check FII/DII activity from previous session",
                ]),
                ("📊 Open Interest Analysis", [
                    "Check Nifty & Bank Nifty option chain on NSE",
                    "Identify max pain level for Nifty",
                    "Note PCR (Put-Call Ratio) — >1.2 bullish, <0.8 bearish",
                    "Check OI change — short build-up or long build-up?",
                ]),
                ("📈 Technical Trend Confirmation", [
                    "Confirm trend with TA for top 3-5 watchlist stocks",
                    "Draw support and resistance levels on charts",
                    "Check SMA 20, 50, 200 alignment",
                    "Note RSI and MACD signals for each stock",
                ]),
                ("🌏 Market & Global Trend", [
                    "Check Gift Nifty / SGX Nifty for gap-up/gap-down indication",
                    "Review US market close (S&P 500, Nasdaq, Dow)",
                    "Check Asian markets opening (Nikkei, Hang Seng)",
                    "Review INR/USD and crude oil price",
                    "Read today's top market news (ET, MoneyControl)",
                ]),
                ("📋 Watchlist & Trade Plans", [
                    "Finalise watchlist with breakout levels noted",
                    "Define entry, stop loss and target for each trade",
                    "Calculate R:R ratio — only trade if R:R ≥ 1.5",
                    "Set position size using the Position Sizer tool",
                    "Set price alerts for breakout levels",
                ]),
            ]

            all_keys = []
            for section_title, items in checklist_items:
                st.markdown(f"#### {section_title}")
                for item in items:
                    key = f"premarket_check_{item[:30].replace(' ', '_')}"
                    all_keys.append(key)
                    if key not in st.session_state:
                        st.session_state[key] = False
                    st.checkbox(item, key=key)
                st.markdown("")

            total_checks = len(all_keys)
            completed = sum(1 for k in all_keys if st.session_state.get(k, False))
            completion_pct = completed / total_checks * 100

            st.divider()
            if completion_pct == 100:
                st.success(f"🎯 **{completed}/{total_checks} complete — You're fully prepared! Trade with confidence.**")
            elif completion_pct >= 60:
                st.warning(f"⚡ **{completed}/{total_checks} complete ({completion_pct:.0f}%) — Good progress. Complete remaining items.**")
            else:
                st.error(f"⚠️ **{completed}/{total_checks} complete ({completion_pct:.0f}%) — Preparation incomplete. Don't rush into trades.**")

            st.progress(completion_pct / 100)

            col1, col2 = st.columns(2)
            with col1:
                st.button(
                    "🔄 Reset Checklist",
                    use_container_width=True,
                    key="reset_checklist",
                    on_click=reset_premarket_checklist,
                    args=(all_keys,),
                )
            with col2:
                watchlist_count = len(st.session_state.get("pre_watchlist", []))
                st.metric("Watchlist Stocks Ready", watchlist_count)

            st.divider()
            st.markdown("#### 📅 Pre-Market Timeline")
            timeline = [
                ("7:00 AM", "Check global markets, US close, Gift Nifty indication"),
                ("7:30 AM", "Read market news — ET Markets, MoneyControl"),
                ("8:00 AM", "Run momentum scan, review OI data from NSE website"),
                ("8:30 AM", "Confirm trend with TA, draw levels on charts"),
                ("8:45 AM", "Finalise watchlist, note all breakout and SL levels"),
                ("9:00 AM", "✅ All preparation complete — ready for pre-open session"),
                ("9:00–9:08 AM", "Pre-open order collection on NSE"),
                ("9:08–9:15 AM", "Pre-open order matching — do not chase"),
                ("9:15 AM", "🔔 Market opens — execute only pre-planned trades"),
            ]
            for time_str, action in timeline:
                icon = "✅" if "complete" in action.lower() or "ready" in action.lower() else "🕐"
                st.markdown(f"**{time_str}** — {action}")

    # ── PRE-OPEN SESSION ─────────────────────────────────────
    def show_pre_open_session():
        st.header("⚡ Pre-Open Session (9:00 AM – 9:15 AM)")
        st.markdown("**Monitor the pre-open market, identify gap ups/downs, and prioritise your watchlist before trading begins.**")

        now_ist = get_ist_time()
        hour, minute = now_ist.hour, now_ist.minute
        in_pre_open = (hour == 9 and minute < 15)
        pre_open_over = (hour > 9) or (hour == 9 and minute >= 15)

        if in_pre_open:
            mins_left = 15 - minute
            st.success(f"🟢 **PRE-OPEN SESSION ACTIVE** — {mins_left} minute(s) until market opens at 9:15 AM IST")
        elif pre_open_over:
            st.info(f"🔵 Pre-open session has ended. Market is {'OPEN' if is_market_open() else 'CLOSED'}. Use this tab for review or next session prep.")
        else:
            mins_to_start = (9 * 60) - (hour * 60 + minute)
            st.warning(f"🕐 Pre-open session starts in {mins_to_start} minutes (at 9:00 AM IST).")

        st.divider()
        tabs = st.tabs([
            "📡 NSE Pre-Open Data",
            "📈 Gap Up / Gap Down Scanner",
            "🎯 Watchlist Prioritiser",
            "📋 Pre-Open Log",
        ])

        # ── TAB 1: NSE PRE-OPEN DATA ──
        with tabs[0]:
            st.markdown("### 📡 Pre-Open Market Data from NSE")
            st.info("During 9:00–9:08 AM, NSE collects orders. During 9:08–9:15 AM, orders are matched and opening prices are determined.")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("#### 🔗 Live NSE Pre-Open Links")
                nse_preopen_links = {
                    "NSE Pre-Open Market (Equity)": "https://www.nseindia.com/market-data/pre-open-market-cm-and-emerge-platform",
                    "NSE Option Chain (Nifty)": "https://www.nseindia.com/option-chain",
                    "NSE Most Active by Volume": "https://www.nseindia.com/market-data/most-active-securities",
                    "NSE Most Active by Value": "https://www.nseindia.com/market-data/most-active-securities",
                    "NSE Market Watch": "https://www.nseindia.com/market-data/live-equity-market",
                    "Gift Nifty (Real-time)": "https://www.nseindia.com/market-data/giftCity",
                    "BSE Pre-Open Data": "https://www.bseindia.com/markets/equity/EQReports/PreOpenMarket.aspx",
                }
                for label, url in nse_preopen_links.items():
                    st.markdown(f"• [{label}]({url})")

                st.divider()
                st.markdown("#### 🕐 Pre-Open Session Timeline")
                timeline_data = [
                    ("9:00 – 9:08 AM", "Order Collection", "Place buy/sell orders at your desired price. Orders can be modified or cancelled.", "#00d4ff"),
                    ("9:08 – 9:15 AM", "Order Matching", "Orders are matched at a single equilibrium price. No modifications allowed.", "#ffa500"),
                    ("9:15 AM", "Market Opens", "Regular trading begins. Your pre-open orders may get executed at the equilibrium price.", "#00d084"),
                ]
                for time_str, phase, desc, color in timeline_data:
                    st.markdown(f"""<div style="border-left:4px solid {color};padding:10px 14px;margin:8px 0;background:#1a1d2e;border-radius:4px;">
                        <div style="color:{color};font-weight:700;">{time_str} — {phase}</div>
                        <div style="color:#aaa;font-size:0.85rem;margin-top:4px;">{desc}</div>
                    </div>""", unsafe_allow_html=True)

            with col2:
                st.markdown("#### 📊 What to Look For in Pre-Open Data")
                st.markdown("""
                | Signal | What It Means |
                |--------|---------------|
                | **High Volume** in pre-open | Institutional interest — strong conviction |
                | **Gap Up > 2%** | Positive overnight news; may fade or continue |
                | **Gap Down > 2%** | Negative news; wait for support before entering |
                | **Price > IEP** | More buyers than sellers at indicative price |
                | **Price < IEP** | More sellers — potential weakness |
                | **Nifty gap up** | Broad bullish bias — longs preferred |
                | **Nifty gap down** | Cautious — avoid chasing gap-down stocks |
                """)

                st.markdown("#### ⚠️ Pre-Open Trading Rules")
                st.markdown("""
                - **Do not chase gap-ups** blindly — wait for 9:15 AM confirmation
                - Pre-open orders execute at **equilibrium price** (IEP), not your limit price necessarily
                - **Avoid market orders** during pre-open — use limit orders only
                - High pre-open volume = genuine interest; low volume = may reverse
                - Check **Gift Nifty** for direction — it predicts Nifty opening
                """)

                st.markdown("#### 🧮 Gift Nifty Gap Calculator")
                col_a, col_b = st.columns(2)
                with col_a:
                    gift_nifty = st.number_input("Gift Nifty (current)", min_value=0.0, step=1.0, key="gift_val")
                with col_b:
                    prev_nifty_close = st.number_input("Nifty Previous Close", min_value=0.0, step=1.0, key="prev_nifty")
                if gift_nifty > 0 and prev_nifty_close > 0:
                    gap = gift_nifty - prev_nifty_close
                    gap_pct = (gap / prev_nifty_close) * 100
                    gap_color = "#00d084" if gap > 0 else "#ff4444"
                    gap_label = "GAP UP" if gap > 0 else "GAP DOWN"
                    st.markdown(f"""<div class="metric-card" style="text-align:center;">
                        <div style="color:#888;">Indicated Opening</div>
                        <div style="color:{gap_color};font-size:1.8rem;font-weight:700;">{gap_label}: {gap:+.0f} pts ({gap_pct:+.2f}%)</div>
                        <div style="color:#aaa;margin-top:6px;">{'⚡ Significant gap — exercise caution' if abs(gap_pct) > 1 else '✅ Small gap — relatively normal open'}</div>
                    </div>""", unsafe_allow_html=True)

        # ── TAB 2: GAP UP / GAP DOWN SCANNER ──
        with tabs[1]:
            st.markdown("### 📈 Gap Up / Gap Down Scanner")
            st.markdown("Scan Nifty 50 stocks for significant price gaps from previous close to today's open.")

            col1, col2 = st.columns([1, 2])
            with col1:
                gap_threshold = st.slider("Min Gap % to flag", 0.5, 5.0, 1.0, 0.25, key="gap_thresh")
                gap_universe = st.multiselect(
                    "Select Stocks to Scan",
                    options=list(NIFTY_50_DATA.keys()),
                    default=list(NIFTY_50_DATA.keys())[:20],
                    key="gap_universe"
                )
                if "pre_watchlist" in st.session_state and st.session_state.pre_watchlist:
                    if st.checkbox("Also include Pre-Market Watchlist stocks", value=True, key="gap_incl_wl"):
                        wl_tickers = [w["ticker"] for w in st.session_state.pre_watchlist]
                        gap_universe = list(set(gap_universe + wl_tickers))

                run_gap = st.button("🔍 Scan for Gaps", use_container_width=True, key="run_gap_scan")

            with col2:
                if run_gap:
                    gap_results = []
                    progress = st.progress(0)
                    status = st.empty()
                    for i, ticker in enumerate(gap_universe):
                        progress.progress((i + 1) / len(gap_universe))
                        name = NIFTY_50_DATA.get(ticker, (ticker, "Unknown"))[0]
                        status.text(f"Checking {name}…")
                        try:
                            df = yf.download(ticker, period="5d", interval="1d", progress=False)
                            df = flatten_yf_columns(df)
                            if df is None or len(df) < 2:
                                continue
                            prev_close = float(df["Close"].iloc[-2])
                            today_open = float(df["Open"].iloc[-1])
                            today_close = float(df["Close"].iloc[-1])
                            gap_pct = ((today_open - prev_close) / prev_close) * 100
                            close_change = ((today_close - today_open) / today_open) * 100
                            volume_today = float(df["Volume"].iloc[-1])
                            volume_prev = float(df["Volume"].iloc[-2])
                            vol_ratio = volume_today / volume_prev if volume_prev > 0 else 1

                            if abs(gap_pct) >= gap_threshold:
                                # Check if in pre-market watchlist
                                in_watchlist = ticker in [w["ticker"] for w in st.session_state.get("pre_watchlist", [])]
                                gap_results.append({
                                    "Ticker": ticker,
                                    "Name": name,
                                    "Prev Close": format_indian_number(prev_close, True),
                                    "Open": format_indian_number(today_open, True),
                                    "Gap %": round(gap_pct, 2),
                                    "Intraday Move": f"{close_change:+.2f}%",
                                    "Volume (×prev)": round(vol_ratio, 2),
                                    "Type": "⬆️ Gap Up" if gap_pct > 0 else "⬇️ Gap Down",
                                    "In Watchlist": "⭐ Yes" if in_watchlist else "—",
                                })
                        except Exception:
                            continue
                    progress.empty()
                    status.empty()

                    if gap_results:
                        gap_df = pd.DataFrame(gap_results).sort_values("Gap %", ascending=False)
                        gap_ups = gap_df[gap_df["Gap %"] > 0]
                        gap_downs = gap_df[gap_df["Gap %"] < 0]

                        g1, g2 = st.columns(2)
                        with g1:
                            st.metric("⬆️ Gap Ups", len(gap_ups))
                        with g2:
                            st.metric("⬇️ Gap Downs", len(gap_downs))

                        market_bias = "🟢 Bullish Bias" if len(gap_ups) > len(gap_downs) * 1.5 else \
                                      "🔴 Bearish Bias" if len(gap_downs) > len(gap_ups) * 1.5 else \
                                      "🟡 Mixed / Neutral"
                        st.markdown(f"### Market Opening Bias: {market_bias}")

                        st.markdown("#### ⬆️ Gap Ups")
                        if not gap_ups.empty:
                            st.dataframe(gap_ups.drop(columns=["Type"]), use_container_width=True, hide_index=True)
                        else:
                            st.info("No significant gap ups found.")

                        st.markdown("#### ⬇️ Gap Downs")
                        if not gap_downs.empty:
                            st.dataframe(gap_downs.drop(columns=["Type"]), use_container_width=True, hide_index=True)
                        else:
                            st.info("No significant gap downs found.")

                        # Watchlist stars
                        wl_in_gaps = gap_df[gap_df["In Watchlist"] == "⭐ Yes"]
                        if not wl_in_gaps.empty:
                            st.success(f"⭐ **{len(wl_in_gaps)} stock(s) from your Pre-Market Watchlist have significant gaps — prioritise these!**")
                            st.dataframe(wl_in_gaps, use_container_width=True, hide_index=True)

                        # Save to session for log
                        st.session_state["pre_open_gap_scan"] = {
                            "timestamp": get_ist_time().strftime("%H:%M IST"),
                            "results": gap_results,
                            "market_bias": market_bias,
                        }
                    else:
                        st.info(f"No stocks with gaps ≥ {gap_threshold}% found. Try lowering the threshold.")
                else:
                    st.info("Click 'Scan for Gaps' to identify gap up and gap down stocks.")

        # ── TAB 3: WATCHLIST PRIORITISER ──
        with tabs[2]:
            st.markdown("### 🎯 Watchlist Prioritiser")
            st.markdown("Cross-reference your Pre-Market Watchlist with current pre-open data to prioritise which stocks to focus on at 9:15 AM.")

            pre_wl = st.session_state.get("pre_watchlist", [])
            if not pre_wl:
                st.warning("Your Pre-Market Watchlist is empty. Go to **🌅 Pre-Market Prep → Step 4** to build your watchlist first.")
            else:
                if st.button("🔄 Fetch Live Data & Prioritise", use_container_width=True, key="prioritise_btn"):
                    enriched = []
                    progress = st.progress(0)
                    for i, item in enumerate(pre_wl):
                        progress.progress((i + 1) / len(pre_wl))
                        ticker = item["ticker"]
                        score = 0
                        reasons = []
                        try:
                            df = yf.download(ticker, period="5d", interval="1d", progress=False)
                            df = flatten_yf_columns(df)
                            if df is not None and len(df) >= 2:
                                prev_close = float(df["Close"].iloc[-2])
                                today_open = float(df["Open"].iloc[-1])
                                live_price = float(df["Close"].iloc[-1])
                                gap_pct = ((today_open - prev_close) / prev_close) * 100
                                vol_ratio = float(df["Volume"].iloc[-1]) / float(df["Volume"].iloc[-2]) \
                                            if float(df["Volume"].iloc[-2]) > 0 else 1

                                # Scoring logic
                                breakout = item.get("breakout", 0)
                                sl = item.get("stop_loss", 0)
                                target = item.get("target", 0)
                                bias = item.get("bias", "Watch Only")

                                if bias == "Long":
                                    if gap_pct > 0:
                                        score += 2; reasons.append(f"Gap up {gap_pct:+.2f}% ✅")
                                    else:
                                        score -= 1; reasons.append(f"Gap down {gap_pct:+.2f}% ⚠️")
                                    if breakout > 0 and abs(live_price - breakout) / breakout < 0.02:
                                        score += 3; reasons.append("Price near breakout ⚡")
                                    if live_price > breakout > 0:
                                        score += 2; reasons.append("Breakout already triggered 🚀")
                                elif bias == "Short":
                                    if gap_pct < 0:
                                        score += 2; reasons.append(f"Gap down {gap_pct:+.2f}% ✅")
                                    else:
                                        score -= 1; reasons.append(f"Gap up {gap_pct:+.2f}% ⚠️")
                                    if breakout > 0 and abs(live_price - breakout) / breakout < 0.02:
                                        score += 3; reasons.append("Price near breakdown ⚡")

                                if vol_ratio > 1.5:
                                    score += 1; reasons.append(f"High volume {vol_ratio:.1f}× ✅")

                                # R:R check
                                rr = 0
                                if bias == "Long" and breakout > sl > 0 and target > breakout:
                                    rr = (target - breakout) / (breakout - sl)
                                elif bias == "Short" and breakout < sl and target < breakout:
                                    rr = (breakout - target) / (sl - breakout)
                                if rr >= 2:
                                    score += 2; reasons.append(f"R:R 1:{rr:.1f} ✅")
                                elif rr >= 1.5:
                                    score += 1; reasons.append(f"R:R 1:{rr:.1f}")

                                enriched.append({
                                    "ticker": ticker,
                                    "bias": bias,
                                    "live_price": live_price,
                                    "gap_pct": gap_pct,
                                    "vol_ratio": vol_ratio,
                                    "breakout": breakout,
                                    "sl": sl,
                                    "target": target,
                                    "rr": rr,
                                    "score": score,
                                    "reasons": reasons,
                                    "notes": item.get("notes", ""),
                                })
                        except Exception:
                            enriched.append({
                                "ticker": ticker, "bias": item.get("bias",""),
                                "score": 0, "reasons": ["Could not fetch data"],
                                "live_price": 0, "gap_pct": 0, "vol_ratio": 0,
                                "breakout": item.get("breakout",0), "sl": item.get("stop_loss",0),
                                "target": item.get("target",0), "rr": 0, "notes": item.get("notes",""),
                            })
                    progress.empty()

                    enriched.sort(key=lambda x: x["score"], reverse=True)
                    st.session_state["pre_open_prioritised"] = enriched

                prioritised = st.session_state.get("pre_open_prioritised", [])
                if prioritised:
                    st.markdown("### 🏆 Prioritised Trade List (Highest Score First)")
                    for rank, item in enumerate(prioritised, 1):
                        score = item["score"]
                        score_color = "#00d084" if score >= 5 else "#ffa500" if score >= 2 else "#ff4444"
                        bias_color = "#00d084" if item["bias"] == "Long" else "#ff4444" if item["bias"] == "Short" else "#a78bfa"
                        gap_color = "#00d084" if item["gap_pct"] > 0 else "#ff4444"

                        priority_label = "🔥 HIGH PRIORITY" if score >= 5 else "⚡ MEDIUM" if score >= 2 else "👁️ WATCH ONLY"
                        st.markdown(f"""<div class="metric-card" style="border-left:5px solid {score_color};">
                            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                                <div>
                                    <span style="font-size:0.85rem;color:#888;">#{rank}</span>
                                    <span style="font-size:1.15rem;font-weight:700;color:#fff;margin:0 10px;">{item['ticker']}</span>
                                    <span style="background:{bias_color}22;color:{bias_color};padding:2px 8px;border-radius:12px;font-size:0.8rem;">{item['bias']}</span>
                                    <span style="margin-left:10px;color:{score_color};font-weight:600;">{priority_label}</span>
                                </div>
                                <div style="text-align:right;">
                                    <span style="font-size:1.1rem;color:#fff;">{format_indian_number(item['live_price'], True)}</span>
                                    <span style="margin-left:8px;color:{gap_color};">{item['gap_pct']:+.2f}% gap</span>
                                    <span style="margin-left:10px;background:{score_color}33;color:{score_color};padding:2px 8px;border-radius:8px;">Score: {score}</span>
                                </div>
                            </div>
                            <div style="display:flex;gap:20px;margin-top:8px;font-size:0.83rem;color:#aaa;flex-wrap:wrap;">
                                <span>🎯 Breakout: <b style="color:#fff;">{format_indian_number(item['breakout'], True)}</b></span>
                                <span>🛑 SL: <b style="color:#ff4444;">{format_indian_number(item['sl'], True)}</b></span>
                                <span>🏁 Target: <b style="color:#00d084;">{format_indian_number(item['target'], True)}</b></span>
                                <span>R:R: <b>{'1:'+str(round(item['rr'],1)) if item['rr'] > 0 else '–'}</b></span>
                                <span>Vol: <b>{item['vol_ratio']:.1f}×</b></span>
                            </div>
                            <div style="margin-top:8px;font-size:0.82rem;color:#aaa;">
                                {'  •  '.join(item['reasons']) if item['reasons'] else '—'}
                            </div>
                            {f'<div style="margin-top:4px;font-size:0.8rem;color:#888;">📝 {item["notes"]}</div>' if item.get("notes") else ''}
                        </div>""", unsafe_allow_html=True)

                    # Add top picks to session watchlist
                    high_priority = [x for x in prioritised if x["score"] >= 5]
                    if high_priority and st.button(f"📋 Add {len(high_priority)} High Priority to Main Watchlist", key="add_hp_wl"):
                        for x in high_priority:
                            if x["ticker"] not in st.session_state.watchlist:
                                st.session_state.watchlist.append(x["ticker"])
                        st.success(f"✅ Added {len(high_priority)} high-priority stocks to main watchlist!")

        # ── TAB 4: PRE-OPEN LOG ──
        with tabs[3]:
            st.markdown("### 📋 Pre-Open Session Log")
            st.markdown("Keep a manual or auto-populated log of key observations during the pre-open window.")

            with st.expander("➕ Add Manual Observation", expanded=True):
                log_col1, log_col2 = st.columns([2, 1])
                with log_col1:
                    log_note = st.text_area("Observation / Note", placeholder="e.g. Nifty gapping up 100pts, Gift Nifty at 24100, Reliance showing strong pre-open volume…", key="log_note", height=80)
                with log_col2:
                    log_type = st.selectbox("Type", ["📊 Market Observation", "⬆️ Gap Up Spotted", "⬇️ Gap Down Spotted", "⭐ Priority Stock", "⚠️ Risk Flag"], key="log_type")
                    log_ticker = st.text_input("Ticker (optional)", key="log_ticker")
                if st.button("📝 Log Observation", use_container_width=True, key="log_add"):
                    if log_note:
                        if "pre_open_log" not in st.session_state:
                            st.session_state.pre_open_log = []
                        st.session_state.pre_open_log.append({
                            "time": get_ist_time().strftime("%H:%M:%S IST"),
                            "type": log_type,
                            "ticker": log_ticker.upper() if log_ticker else "—",
                            "note": log_note,
                        })
                        st.success("Logged!")
                        st.rerun()

            # Auto-populate from gap scan
            gap_scan = st.session_state.get("pre_open_gap_scan")
            if gap_scan:
                st.markdown(f"""<div style="background:#1a2332;border:1px solid #2d3548;border-radius:8px;padding:12px;margin:8px 0;">
                    <span style="color:#00d4ff;font-weight:600;">🤖 Auto-logged from Gap Scanner ({gap_scan['timestamp']})</span><br>
                    <span style="color:#aaa;">Found {len(gap_scan['results'])} stocks with significant gaps. Market bias: {gap_scan['market_bias']}</span>
                </div>""", unsafe_allow_html=True)

            log_entries = st.session_state.get("pre_open_log", [])
            if log_entries:
                st.markdown(f"#### 📜 Log Entries ({len(log_entries)})")
                for entry in reversed(log_entries):
                    st.markdown(f"""<div style="border-left:3px solid #2d3548;padding:8px 12px;margin:4px 0;background:#1a1d2e;border-radius:4px;">
                        <span style="color:#888;font-size:0.8rem;">{entry['time']}</span>
                        <span style="margin-left:8px;font-size:0.85rem;">{entry['type']}</span>
                        {f'<span style="margin-left:8px;background:#2d3548;padding:1px 6px;border-radius:8px;font-size:0.8rem;color:#00d4ff;">{entry["ticker"]}</span>' if entry["ticker"] != "—" else ""}
                        <div style="color:#ddd;margin-top:4px;">{entry['note']}</div>
                    </div>""", unsafe_allow_html=True)

                if st.button("🗑️ Clear Log", key="clear_log"):
                    st.session_state.pre_open_log = []
                    st.rerun()

                # Export log
                log_df = pd.DataFrame(log_entries)
                csv = log_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Export Log CSV",
                    data=csv,
                    file_name=f"pre_open_log_{get_ist_time().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="log_export"
                )
            else:
                st.info("No log entries yet. Add observations above or run the Gap Scanner to auto-populate.")

    # ── LIVE MARKET ──────────────────────────────────────────
    def show_live_market():
        st.header("🔴 Live Market (After 9:15 AM)")
        st.markdown("**Real-time trade execution toolkit — breakout alerts, strategy scanners, price action & TA signals**")

        now_ist = get_ist_time()
        market_open = is_market_open()
        if market_open:
            st.success(f"🟢 Market is **LIVE** — {now_ist.strftime('%I:%M %p IST')}")
        elif now_ist.hour < 9 or (now_ist.hour == 9 and now_ist.minute < 15):
            st.warning(f"⏳ Market opens at 9:15 AM IST. Use Pre-Market Prep and Pre-Open tabs first.")
        else:
            st.info(f"🔵 Market closed for today. Reviewing live session data — {now_ist.strftime('%I:%M %p IST')}")

        tabs = st.tabs([
            "💥 Breakout Monitor",
            "📊 Strategy Scanners",
            "📉 Price Action Signals",
            "🔧 TA Live Dashboard",
            "🌊 Volume & Price Shockers",
            "📋 Trade Execution Log",
        ])

        # ─── TAB 1: BREAKOUT MONITOR ───────────────────────────
        with tabs[0]:
            st.markdown("### 💥 Breakout Monitor — Watch Prepared Levels in Real Time")
            pre_wl = st.session_state.get("pre_watchlist", [])

            if not pre_wl:
                st.warning("No pre-market watchlist found. Go to **🌅 Pre-Market Prep → Step 4** to set up breakout levels.")
                st.markdown("You can also add stocks directly below:")

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### ➕ Quick Add Breakout Level")
                with st.form("live_breakout_form"):
                    bo_ticker = st.text_input("Ticker", placeholder="RELIANCE.NS")
                    bo_price  = st.number_input("Breakout Price (₹)", min_value=0.0, step=0.5)
                    bo_sl     = st.number_input("Stop Loss (₹)",      min_value=0.0, step=0.5)
                    bo_target = st.number_input("Target (₹)",          min_value=0.0, step=0.5)
                    bo_bias   = st.radio("Bias", ["Long", "Short"], horizontal=True)
                    bo_submit = st.form_submit_button("📌 Add", use_container_width=True)
                if bo_submit and bo_ticker:
                    ticker_n = normalize_ticker(bo_ticker, st.session_state.exchange)
                    if "pre_watchlist" not in st.session_state:
                        st.session_state.pre_watchlist = []
                    existing = [w["ticker"] for w in st.session_state.pre_watchlist]
                    if ticker_n not in existing:
                        st.session_state.pre_watchlist.append({
                            "ticker": ticker_n, "breakout": bo_price,
                            "stop_loss": bo_sl, "target": bo_target,
                            "bias": bo_bias, "notes": "Added in Live Market",
                            "added_at": now_ist.strftime("%Y-%m-%d %H:%M"),
                        })
                        st.success(f"✅ {ticker_n} added!")
                        st.rerun()
                    else:
                        st.warning("Already in watchlist.")

            with col2:
                refresh_bo = st.button("🔄 Refresh Breakout Status", use_container_width=True, key="refresh_bo")
                wl = st.session_state.get("pre_watchlist", [])
                if not wl:
                    st.info("Watchlist empty — add stocks in the left panel or via Pre-Market Prep.")
                else:
                    triggered, near, not_yet = [], [], []
                    if refresh_bo:
                        prog = st.progress(0)
                        for i, item in enumerate(wl):
                            prog.progress((i+1)/len(wl))
                            try:
                                df = yf.download(item["ticker"], period="1d", interval="5m", progress=False)
                                df = flatten_yf_columns(df)
                                if df is not None and not df.empty:
                                    live = float(df["Close"].iloc[-1])
                                    item["live_price"] = live
                                    item["live_high"]  = float(df["High"].max())
                                    item["live_low"]   = float(df["Low"].min())
                                    item["live_vol"]   = int(df["Volume"].sum())
                            except Exception:
                                pass
                        prog.empty()
                        st.session_state.pre_watchlist = wl

                    for item in wl:
                        lp = item.get("live_price")
                        bo = item.get("breakout", 0)
                        sl = item.get("stop_loss", 0)
                        if lp is None or bo == 0:
                            not_yet.append(item)
                            continue
                        pct_away = (bo - lp) / lp * 100 if item["bias"] == "Long" else (lp - bo) / lp * 100
                        if item["bias"] == "Long":
                            if lp >= bo:
                                triggered.append((item, "breakout"))
                            elif sl > 0 and lp <= sl:
                                triggered.append((item, "stopped"))
                            elif abs(pct_away) <= 1.0:
                                near.append(item)
                            else:
                                not_yet.append(item)
                        else:
                            if lp <= bo:
                                triggered.append((item, "breakdown"))
                            elif sl > 0 and lp >= sl:
                                triggered.append((item, "stopped"))
                            elif abs(pct_away) <= 1.0:
                                near.append(item)
                            else:
                                not_yet.append(item)

                    # Summary metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("🔥 Triggered", len(triggered))
                    m2.metric("⚡ Near Breakout", len(near))
                    m3.metric("👁️ Watching", len(not_yet))

                    def _render_bo_card(item, tag="", tag_color="#888"):
                        lp = item.get("live_price")
                        bo = item.get("breakout", 0)
                        sl = item.get("stop_loss", 0)
                        tgt = item.get("target", 0)
                        bias_c = "#00d084" if item["bias"] == "Long" else "#ff4444"
                        lp_str = format_indian_number(lp, True) if lp else "–"
                        rr = 0
                        if item["bias"] == "Long" and bo > sl > 0 and tgt > bo:
                            rr = (tgt - bo) / (bo - sl)
                        elif item["bias"] == "Short" and bo < sl and 0 < tgt < bo:
                            rr = (bo - tgt) / (sl - bo)
                        st.markdown(f"""<div class="metric-card" style="border-left:4px solid {tag_color};margin-bottom:8px;">
                            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
                                <div>
                                    <span style="font-weight:700;font-size:1.05rem;">{item['ticker']}</span>
                                    <span style="margin-left:8px;background:{bias_c}22;color:{bias_c};padding:1px 7px;border-radius:10px;font-size:0.78rem;">{item['bias']}</span>
                                    {f'<span style="margin-left:8px;background:{tag_color}33;color:{tag_color};padding:1px 7px;border-radius:10px;font-size:0.78rem;">{tag}</span>' if tag else ''}
                                </div>
                                <span style="font-size:1.1rem;color:#fff;">{lp_str}</span>
                            </div>
                            <div style="display:flex;gap:18px;margin-top:6px;font-size:0.82rem;color:#aaa;flex-wrap:wrap;">
                                <span>🎯 BO: <b style="color:#fff;">{format_indian_number(bo, True) if bo else '–'}</b></span>
                                <span>🛑 SL: <b style="color:#ff4444;">{format_indian_number(sl, True) if sl else '–'}</b></span>
                                <span>🏁 Tgt: <b style="color:#00d084;">{format_indian_number(tgt, True) if tgt else '–'}</b></span>
                                <span>R:R: <b>{'1:'+str(round(rr,1)) if rr>0 else '–'}</b></span>
                                {'<span>📊 Vol: <b>'+f"{item.get('live_vol',0):,}"+'</b></span>' if item.get('live_vol') else ''}
                            </div>
                        </div>""", unsafe_allow_html=True)

                    if triggered:
                        st.markdown("#### 🔥 Triggered")
                        for item, kind in triggered:
                            color = "#00d084" if kind in ("breakout","breakdown") else "#ff4444"
                            label = "✅ BREAKOUT" if kind == "breakout" else "✅ BREAKDOWN" if kind == "breakdown" else "❌ STOPPED OUT"
                            _render_bo_card(item, label, color)
                    if near:
                        st.markdown("#### ⚡ Near Breakout (within 1%)")
                        for item in near:
                            _render_bo_card(item, "APPROACHING", "#ffd700")
                    if not_yet:
                        st.markdown("#### 👁️ Watching")
                        for item in not_yet:
                            _render_bo_card(item)

        # ─── TAB 2: STRATEGY SCANNERS ──────────────────────────
        with tabs[1]:
            st.markdown("### 📊 Intraday Strategy Scanners")
            st.info("These scanners use 5-minute candle data via yfinance to detect common intraday setups used in the Indian market.")

            strategy = st.selectbox("Select Strategy", [
                "Opening Range Breakout (ORB)",
                "Open = High (Bearish — Short Bias)",
                "Open = Low (Bullish — Long Bias)",
                "VWAP Reclaim (Bullish)",
                "VWAP Rejection (Bearish)",
                "Inside Bar Breakout",
            ], key="live_strategy")

            scan_universe_raw = st.multiselect(
                "Stocks to Scan",
                options=list(NIFTY_50_DATA.keys()),
                default=list(NIFTY_50_DATA.keys())[:15],
                key="live_scan_universe"
            )
            if "pre_watchlist" in st.session_state and st.session_state.pre_watchlist:
                if st.checkbox("Include Pre-Market Watchlist stocks", value=True, key="strat_incl_wl"):
                    wl_tickers = [w["ticker"] for w in st.session_state.pre_watchlist]
                    scan_universe_raw = list(set(scan_universe_raw + wl_tickers))

            orb_minutes = 15
            if "Opening Range Breakout" in strategy:
                orb_minutes = st.slider("ORB Window (minutes after open)", 5, 60, 15, 5, key="orb_min")

            if st.button(f"🚀 Run {strategy} Scan", use_container_width=True, key="run_strat_scan"):
                results = []
                prog = st.progress(0)
                status = st.empty()
                for i, ticker in enumerate(scan_universe_raw):
                    prog.progress((i+1)/len(scan_universe_raw))
                    name = NIFTY_50_DATA.get(ticker, (ticker,"Unknown"))[0]
                    status.text(f"Scanning {name}…")
                    try:
                        df5 = yf.download(ticker, period="1d", interval="5m", progress=False)
                        df5 = flatten_yf_columns(df5)
                        if df5 is None or len(df5) < 4:
                            continue
                        df5 = df5.dropna()

                        match = False
                        signal_detail = ""
                        live_price = float(df5["Close"].iloc[-1])

                        if "Opening Range Breakout" in strategy:
                            candles_in_window = orb_minutes // 5
                            orb_df = df5.iloc[:candles_in_window]
                            orb_high = float(orb_df["High"].max())
                            orb_low  = float(orb_df["Low"].min())
                            if live_price > orb_high * 1.001:
                                match = True
                                signal_detail = f"Price {format_indian_number(live_price,True)} broke ORB High {format_indian_number(orb_high,True)} — LONG"
                            elif live_price < orb_low * 0.999:
                                match = True
                                signal_detail = f"Price {format_indian_number(live_price,True)} broke ORB Low {format_indian_number(orb_low,True)} — SHORT"

                        elif "Open = High" in strategy:
                            first = df5.iloc[0]
                            open_p = float(first["Open"])
                            high_p = float(first["High"])
                            if abs(open_p - high_p) / open_p < 0.001:
                                match = True
                                signal_detail = f"Open ≈ High ({format_indian_number(open_p,True)}) — Bearish, Short bias"

                        elif "Open = Low" in strategy:
                            first = df5.iloc[0]
                            open_p = float(first["Open"])
                            low_p  = float(first["Low"])
                            if abs(open_p - low_p) / open_p < 0.001:
                                match = True
                                signal_detail = f"Open ≈ Low ({format_indian_number(open_p,True)}) — Bullish, Long bias"

                        elif "VWAP Reclaim" in strategy:
                            typical = (df5["High"] + df5["Low"] + df5["Close"]) / 3
                            cum_vol = df5["Volume"].cumsum()
                            vwap = (typical * df5["Volume"]).cumsum() / cum_vol
                            vwap_now = float(vwap.iloc[-1])
                            prev_close_5m = float(df5["Close"].iloc[-2])
                            if prev_close_5m < vwap_now and live_price > vwap_now:
                                match = True
                                signal_detail = f"Reclaimed VWAP {format_indian_number(vwap_now,True)} — Bullish"

                        elif "VWAP Rejection" in strategy:
                            typical = (df5["High"] + df5["Low"] + df5["Close"]) / 3
                            cum_vol = df5["Volume"].cumsum()
                            vwap = (typical * df5["Volume"]).cumsum() / cum_vol
                            vwap_now = float(vwap.iloc[-1])
                            prev_close_5m = float(df5["Close"].iloc[-2])
                            if prev_close_5m > vwap_now and live_price < vwap_now:
                                match = True
                                signal_detail = f"Rejected at VWAP {format_indian_number(vwap_now,True)} — Bearish"

                        elif "Inside Bar" in strategy:
                            if len(df5) >= 3:
                                prev_h = float(df5["High"].iloc[-2])
                                prev_l = float(df5["Low"].iloc[-2])
                                curr_h = float(df5["High"].iloc[-1])
                                curr_l = float(df5["Low"].iloc[-1])
                                if curr_h > prev_h * 1.001 or curr_l < prev_l * 0.999:
                                    match = True
                                    direction = "LONG breakout" if curr_h > prev_h else "SHORT breakdown"
                                    signal_detail = f"Inside Bar {direction} — {format_indian_number(live_price,True)}"

                        if match:
                            day_chg = ((live_price - float(df5["Open"].iloc[0])) / float(df5["Open"].iloc[0])) * 100
                            vol_today = int(df5["Volume"].sum())
                            in_wl = ticker in [w["ticker"] for w in st.session_state.get("pre_watchlist", [])]
                            results.append({
                                "Ticker": ticker,
                                "Name": name,
                                "Live Price": format_indian_number(live_price, True),
                                "Day Chg %": f"{day_chg:+.2f}%",
                                "Volume": f"{vol_today:,}",
                                "Signal": signal_detail,
                                "Watchlist": "⭐" if in_wl else "—",
                            })
                    except Exception:
                        continue
                prog.empty()
                status.empty()

                st.session_state["live_strat_results"] = results
                st.session_state["live_strat_name"]    = strategy

            results = st.session_state.get("live_strat_results", [])
            strat_name = st.session_state.get("live_strat_name", strategy)
            if results:
                st.success(f"✅ **{len(results)} stocks** match **{strat_name}**")
                wl_hits = [r for r in results if r["Watchlist"] == "⭐"]
                if wl_hits:
                    st.markdown(f"⭐ **{len(wl_hits)} watchlist stock(s) triggered this strategy — top priority!**")
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

                csv = pd.DataFrame(results).to_csv(index=False)
                st.download_button("⬇️ Export Results CSV", csv,
                    file_name=f"live_scan_{now_ist.strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv", key="strat_csv")

                if st.button("📋 Add all to Watchlist", key="strat_add_wl"):
                    for r in results:
                        if r["Ticker"] not in st.session_state.watchlist:
                            st.session_state.watchlist.append(r["Ticker"])
                    st.success(f"Added {len(results)} stocks!")
            elif st.session_state.get("live_strat_results") is not None:
                st.warning("No stocks matched this strategy. Try another strategy or expand the scan universe.")

            st.divider()
            st.markdown("#### 📚 Strategy Reference Guide")
            strategy_guide = [
                ("Opening Range Breakout (ORB)", "The high/low of the first N minutes is the 'Opening Range'. A breakout above = long; below = short. Most reliable when combined with volume surge and trend alignment.", "Best in trending markets. Avoid on expiry days."),
                ("Open = High (O=H)", "When open ≈ high of the candle, sellers are dominant from the start. Ideal for shorting on a pullback to VWAP or resistance.", "Works well in bear markets or when Nifty is weak."),
                ("Open = Low (O=L)", "When open ≈ low of the candle, buyers absorbed all selling immediately. Ideal for longs on first pullback.", "Works well in bull markets or when Nifty is strong."),
                ("VWAP Reclaim", "Price was below VWAP but closes back above it — institutional buyers stepping in. Long entry with VWAP as stop.", "Higher accuracy when volume is above average on the reclaim candle."),
                ("VWAP Rejection", "Price tags VWAP from above and closes below it — sellers defending. Short entry with VWAP as stop.", "Strongest signal when RSI is also below 50."),
                ("Inside Bar Breakout", "A candle contained within the prior candle's high/low (compression). Breakout of this range signals directional expansion.", "Use with prior trend. Long breakout in uptrend, short in downtrend."),
            ]
            for name_s, desc, note in strategy_guide:
                with st.expander(f"📌 {name_s}"):
                    st.write(desc)
                    st.info(f"💡 {note}")

        # ─── TAB 3: PRICE ACTION SIGNALS ───────────────────────
        with tabs[2]:
            st.markdown("### 📉 Price Action Signals")
            st.markdown("Detect real-time candlestick price action setups on your watchlist stocks.")

            pa_ticker_raw = st.text_input("Enter ticker(s) comma-separated",
                value=", ".join([w["ticker"] for w in st.session_state.get("pre_watchlist",[])[:5]]),
                placeholder="RELIANCE.NS, TCS.NS", key="pa_tickers")
            pa_tf = st.selectbox("Timeframe", ["5m","15m","1h","1d"], index=1, key="pa_tf")

            if pa_ticker_raw and st.button("🔍 Detect Price Action", use_container_width=True, key="run_pa"):
                pa_tickers = [normalize_ticker(t.strip(), st.session_state.exchange)
                              for t in pa_ticker_raw.split(",") if t.strip()]
                for ticker in pa_tickers[:6]:
                    try:
                        period_map = {"5m":"1d","15m":"5d","1h":"1mo","1d":"6mo"}
                        df_pa = yf.download(ticker, period=period_map[pa_tf],
                                            interval=pa_tf, progress=False)
                        df_pa = flatten_yf_columns(df_pa)
                        df_pa = df_pa.dropna()
                        if df_pa is None or len(df_pa) < 5:
                            st.warning(f"Not enough data for {ticker}")
                            continue

                        signals_found = []
                        c = df_pa.iloc[-1]
                        p = df_pa.iloc[-2]
                        pp = df_pa.iloc[-3]
                        o,h,l,cl = float(c["Open"]),float(c["High"]),float(c["Low"]),float(c["Close"])
                        po,ph,pl,pc = float(p["Open"]),float(p["High"]),float(p["Low"]),float(p["Close"])
                        ppo,pph,ppl,ppc = float(pp["Open"]),float(pp["High"]),float(pp["Low"]),float(pp["Close"])
                        body = abs(cl - o)
                        candle_range = h - l

                        # Doji
                        if candle_range > 0 and body / candle_range < 0.1:
                            signals_found.append(("🔵 Doji", "Indecision candle — potential reversal. Wait for next candle confirmation.", "neutral"))
                        # Bullish Engulfing
                        if pc < po and cl > o and cl > po and o < pc:
                            signals_found.append(("🟢 Bullish Engulfing", "Strong reversal signal. Current candle engulfs prior bearish candle — buy on confirmation.", "bullish"))
                        # Bearish Engulfing
                        if pc > po and cl < o and cl < po and o > pc:
                            signals_found.append(("🔴 Bearish Engulfing", "Strong reversal signal. Current bearish candle engulfs prior bullish — sell/short on confirmation.", "bearish"))
                        # Hammer
                        if cl > o and (o - l) > 2 * body and (h - cl) < body * 0.3:
                            signals_found.append(("🟢 Hammer", "Bullish reversal — sellers pushed price down but buyers recovered strongly.", "bullish"))
                        # Shooting Star
                        if cl < o and (h - o) > 2 * body and (cl - l) < body * 0.3:
                            signals_found.append(("🔴 Shooting Star", "Bearish reversal — buyers pushed price up but sellers took control.", "bearish"))
                        # Marubozu Bullish
                        if cl > o and (h - cl) < body * 0.05 and (o - l) < body * 0.05:
                            signals_found.append(("🟢 Bullish Marubozu", "Full bullish candle with no wicks — strong buying conviction throughout the session.", "bullish"))
                        # Marubozu Bearish
                        if cl < o and (h - o) < body * 0.05 and (cl - l) < body * 0.05:
                            signals_found.append(("🔴 Bearish Marubozu", "Full bearish candle with no wicks — strong selling conviction throughout.", "bearish"))
                        # Morning Star
                        if ppc > ppo and body < candle_range * 0.3 and cl > (ppo + ppc)/2:
                            signals_found.append(("🟢 Morning Star", "3-candle bullish reversal pattern. Prior bear candle, small body, then bullish candle closing above midpoint.", "bullish"))
                        # Evening Star
                        if ppo > ppc and body < candle_range * 0.3 and cl < (ppo + ppc)/2:
                            signals_found.append(("🔴 Evening Star", "3-candle bearish reversal. Prior bull candle, small body, then bearish candle closing below midpoint.", "bearish"))
                        # Inside Bar
                        if h < ph and l > pl:
                            signals_found.append(("🟡 Inside Bar (Compression)", "Current candle inside prior range — expect breakout. Trade the direction of breakout.", "neutral"))
                        # Pin Bar Bullish
                        if (o - l) > 2 * (h - cl) and body < candle_range * 0.3:
                            signals_found.append(("🟢 Bullish Pin Bar", "Long lower wick shows rejection of lower prices — buyers in control.", "bullish"))
                        # Pin Bar Bearish
                        if (h - o) > 2 * (cl - l) and body < candle_range * 0.3:
                            signals_found.append(("🔴 Bearish Pin Bar", "Long upper wick shows rejection of higher prices — sellers in control.", "bearish"))

                        bullish_count = sum(1 for s in signals_found if s[2] == "bullish")
                        bearish_count = sum(1 for s in signals_found if s[2] == "bearish")
                        overall = "🟢 Bullish Bias" if bullish_count > bearish_count else \
                                  "🔴 Bearish Bias" if bearish_count > bullish_count else \
                                  "🟡 Neutral / Wait"

                        with st.expander(f"📊 {ticker} — {overall} ({len(signals_found)} signals)", expanded=True):
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            mc1.metric("Price", format_indian_number(cl, True))
                            mc2.metric("High", format_indian_number(h, True))
                            mc3.metric("Low", format_indian_number(l, True))
                            mc4.metric("Signals", len(signals_found))
                            if not signals_found:
                                st.info("No strong price action signals detected on the latest candle. Market may be ranging.")
                            else:
                                for sig_name, sig_desc, sig_type in signals_found:
                                    col_fn = st.success if sig_type == "bullish" else st.error if sig_type == "bearish" else st.warning
                                    col_fn(f"**{sig_name}** — {sig_desc}")
                            if PLOTLY_AVAILABLE:
                                fig = go.Figure(go.Candlestick(
                                    x=df_pa.tail(40).index,
                                    open=df_pa.tail(40)["Open"], high=df_pa.tail(40)["High"],
                                    low=df_pa.tail(40)["Low"],  close=df_pa.tail(40)["Close"],
                                    name=ticker,
                                ))
                                fig.update_layout(template="plotly_dark", height=280,
                                    margin=dict(l=0,r=0,t=20,b=0), showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error processing {ticker}: {e}")

        # ─── TAB 4: TA LIVE DASHBOARD ───────────────────────────
        with tabs[3]:
            st.markdown("### 🔧 Technical Analysis Live Dashboard")
            st.markdown("Live TA signals for any stock — RSI, MACD, Bollinger Bands, VWAP, ATR and more.")

            ta_ticker_raw = st.text_input("Stock Ticker", placeholder="RELIANCE.NS", key="ta_live_ticker")
            ta_tf = st.selectbox("Timeframe", ["5m","15m","1h","1d"], index=1, key="ta_live_tf")
            ta_ticker = normalize_ticker(ta_ticker_raw, st.session_state.exchange) if ta_ticker_raw else None

            if ta_ticker and st.button("📊 Load TA Dashboard", use_container_width=True, key="load_ta_live"):
                period_map = {"5m":"1d","15m":"5d","1h":"1mo","1d":"3mo"}
                try:
                    df_ta = yf.download(ta_ticker, period=period_map[ta_tf],
                                        interval=ta_tf, progress=False)
                    df_ta = flatten_yf_columns(df_ta)
                    df_ta = df_ta.dropna()
                    if df_ta is None or len(df_ta) < 20:
                        st.error("Not enough data. Try a higher timeframe.")
                    else:
                        close = df_ta["Close"].squeeze()
                        high  = df_ta["High"].squeeze()
                        low   = df_ta["Low"].squeeze()
                        vol   = df_ta["Volume"].squeeze()

                        # RSI
                        delta = close.diff()
                        gain  = delta.clip(lower=0).rolling(14).mean()
                        loss  = (-delta.clip(upper=0)).rolling(14).mean()
                        rs    = gain / loss.replace(0, np.nan)
                        rsi   = (100 - 100 / (1 + rs)).iloc[-1]

                        # MACD
                        ema12 = close.ewm(span=12).mean()
                        ema26 = close.ewm(span=26).mean()
                        macd_line = ema12 - ema26
                        signal_line = macd_line.ewm(span=9).mean()
                        macd_val = float(macd_line.iloc[-1])
                        sig_val  = float(signal_line.iloc[-1])
                        macd_hist = macd_val - sig_val

                        # Bollinger
                        sma20 = close.rolling(20).mean()
                        std20 = close.rolling(20).std()
                        bb_upper = float((sma20 + 2*std20).iloc[-1])
                        bb_lower = float((sma20 - 2*std20).iloc[-1])
                        bb_mid   = float(sma20.iloc[-1])
                        live_p   = float(close.iloc[-1])
                        bb_pos   = (live_p - bb_lower)/(bb_upper - bb_lower)*100 if (bb_upper-bb_lower) > 0 else 50

                        # VWAP
                        typical = (high + low + close) / 3
                        vwap_val = float((typical * vol).cumsum().iloc[-1] / vol.cumsum().iloc[-1])

                        # ATR
                        tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
                        atr_val = float(tr.rolling(14).mean().iloc[-1])

                        # SMA/EMA
                        sma50_val  = float(close.rolling(50).mean().iloc[-1]) if len(close)>=50 else None
                        sma200_val = float(close.rolling(200).mean().iloc[-1]) if len(close)>=200 else None
                        ema9_val   = float(close.ewm(span=9).mean().iloc[-1])
                        ema21_val  = float(close.ewm(span=21).mean().iloc[-1])

                        # Stochastic
                        low14  = low.rolling(14).min()
                        high14 = high.rolling(14).max()
                        stoch_k = float(((close - low14)/(high14 - low14)*100).iloc[-1]) if float((high14-low14).iloc[-1]) > 0 else 50

                        # Signal summary
                        live_signals = []
                        live_signals.append(("RSI", f"{rsi:.1f}", "Overbought (>70)" if rsi>70 else "Oversold (<30)" if rsi<30 else "Neutral (30–70)", "#ff4444" if rsi>70 else "#00d084" if rsi<30 else "#aaa"))
                        live_signals.append(("MACD", f"{macd_val:.3f} vs {sig_val:.3f}", "Bullish (above signal)" if macd_val>sig_val else "Bearish (below signal)", "#00d084" if macd_val>sig_val else "#ff4444"))
                        live_signals.append(("Bollinger", f"{bb_pos:.0f}% of band", "Near upper (overbought risk)" if bb_pos>80 else "Near lower (oversold)" if bb_pos<20 else "Mid-range", "#ff4444" if bb_pos>80 else "#00d084" if bb_pos<20 else "#aaa"))
                        live_signals.append(("vs VWAP", f"{format_indian_number(live_p,True)} vs {format_indian_number(vwap_val,True)}", "Above VWAP (bullish)" if live_p>vwap_val else "Below VWAP (bearish)", "#00d084" if live_p>vwap_val else "#ff4444"))
                        live_signals.append(("EMA 9/21", f"9:{format_indian_number(ema9_val,True)} / 21:{format_indian_number(ema21_val,True)}", "EMA9 above EMA21 (bullish)" if ema9_val>ema21_val else "EMA9 below EMA21 (bearish)", "#00d084" if ema9_val>ema21_val else "#ff4444"))
                        live_signals.append(("Stochastic %K", f"{stoch_k:.1f}", "Overbought (>80)" if stoch_k>80 else "Oversold (<20)" if stoch_k<20 else "Neutral", "#ff4444" if stoch_k>80 else "#00d084" if stoch_k<20 else "#aaa"))
                        if sma50_val:
                            live_signals.append(("vs SMA 50", f"{format_indian_number(sma50_val,True)}", "Above SMA50 (bullish)" if live_p>sma50_val else "Below SMA50 (bearish)", "#00d084" if live_p>sma50_val else "#ff4444"))

                        bullish_ta = sum(1 for s in live_signals if s[3]=="#00d084")
                        bearish_ta = sum(1 for s in live_signals if s[3]=="#ff4444")
                        ta_bias = "🟢 BULLISH" if bullish_ta > bearish_ta else "🔴 BEARISH" if bearish_ta > bullish_ta else "🟡 NEUTRAL"
                        ta_color = "#00d084" if "BULLISH" in ta_bias else "#ff4444" if "BEARISH" in ta_bias else "#ffa500"

                        st.markdown(f"""<div class="metric-card" style="text-align:center;border:2px solid {ta_color};">
                            <div style="color:#888;">Overall TA Bias — {ta_ticker} ({ta_tf})</div>
                            <div style="color:{ta_color};font-size:2rem;font-weight:700;">{ta_bias}</div>
                            <div style="color:#aaa;">{bullish_ta} bullish · {bearish_ta} bearish signals</div>
                        </div>""", unsafe_allow_html=True)

                        st.markdown("#### 📊 Individual Indicator Signals")
                        for ind_name, ind_val, ind_interp, ind_color in live_signals:
                            c1,c2,c3 = st.columns([1,1,2])
                            c1.markdown(f"**{ind_name}**")
                            c2.markdown(f"`{ind_val}`")
                            c3.markdown(f"<span style='color:{ind_color};'>{ind_interp}</span>", unsafe_allow_html=True)

                        st.divider()
                        st.metric("ATR (14)", format_indian_number(atr_val, True))
                        st.caption(f"ATR-based SL suggestion: {format_indian_number(live_p - 1.5*atr_val, True)} (Long) / {format_indian_number(live_p + 1.5*atr_val, True)} (Short)")

                        if PLOTLY_AVAILABLE:
                            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                row_heights=[0.6,0.2,0.2], vertical_spacing=0.04)
                            tail = df_ta.tail(80)
                            fig.add_trace(go.Candlestick(x=tail.index, open=tail["Open"],
                                high=tail["High"],low=tail["Low"],close=tail["Close"],name="Price"), row=1,col=1)
                            fig.add_hline(y=bb_upper, line_color="#a78bfa", line_dash="dot", row=1, col=1,
                                annotation_text="BB Upper")
                            fig.add_hline(y=bb_lower, line_color="#a78bfa", line_dash="dot", row=1, col=1,
                                annotation_text="BB Lower")
                            fig.add_hline(y=vwap_val, line_color="#ffd700", line_dash="dash", row=1, col=1,
                                annotation_text="VWAP")
                            rsi_series = 100 - 100/(1+close.diff().clip(lower=0).rolling(14).mean()/(-close.diff().clip(upper=0)).rolling(14).mean().replace(0,np.nan))
                            fig.add_trace(go.Scatter(x=tail.index, y=rsi_series.tail(80), name="RSI",
                                line=dict(color="#a78bfa")), row=2, col=1)
                            fig.add_hline(y=70, line_color="#ff4444", line_dash="dash", row=2, col=1)
                            fig.add_hline(y=30, line_color="#00d084", line_dash="dash", row=2, col=1)
                            hist_colors = ["#00d084" if v>=0 else "#ff4444" for v in macd_line.tail(80)-signal_line.tail(80)]
                            fig.add_trace(go.Bar(x=tail.index, y=(macd_line-signal_line).tail(80),
                                name="MACD Hist", marker_color=hist_colors), row=3, col=1)
                            fig.update_layout(template="plotly_dark", height=500,
                                margin=dict(l=0,r=0,t=20,b=0))
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading TA data: {e}")

        # ─── TAB 5: VOLUME & PRICE SHOCKERS ────────────────────
        with tabs[4]:
            st.markdown("### 🌊 Volume & Price Shockers")
            st.markdown("Identify stocks showing unusual price movement or volume spikes during today's live session.")

            col1, col2 = st.columns([1, 2])
            with col1:
                shocker_type = st.radio("Scan Type", ["Volume Shockers", "Price Shockers", "Both"], key="shocker_type")
                vol_mult_thresh = st.slider("Volume spike (×average)", 1.5, 10.0, 3.0, 0.5, key="vol_spike_thresh")
                price_move_thresh = st.slider("Price move % threshold", 1.0, 10.0, 3.0, 0.5, key="price_move_thresh")
                shocker_universe = st.multiselect("Universe", list(NIFTY_50_DATA.keys()),
                    default=list(NIFTY_50_DATA.keys()), key="shocker_universe")
                run_shocker = st.button("🔍 Find Shockers", use_container_width=True, key="run_shockers")

            with col2:
                if run_shocker:
                    shocker_results = []
                    prog = st.progress(0)
                    status = st.empty()
                    for i, ticker in enumerate(shocker_universe):
                        prog.progress((i+1)/len(shocker_universe))
                        name = NIFTY_50_DATA.get(ticker,(ticker,"Unknown"))[0]
                        status.text(f"Checking {name}…")
                        try:
                            df_1d = yf.download(ticker, period="5d", interval="1d", progress=False)
                            df_1d = flatten_yf_columns(df_1d)
                            df_5m = yf.download(ticker, period="1d", interval="5m", progress=False)
                            df_5m = flatten_yf_columns(df_5m)
                            if df_1d is None or len(df_1d) < 2 or df_5m is None or df_5m.empty:
                                continue
                            prev_close = float(df_1d["Close"].iloc[-2])
                            live_price = float(df_5m["Close"].iloc[-1])
                            today_vol  = int(df_5m["Volume"].sum())
                            avg_daily_vol = float(df_1d["Volume"].iloc[:-1].mean())
                            vol_ratio  = today_vol / avg_daily_vol if avg_daily_vol > 0 else 0
                            price_chg  = (live_price - prev_close) / prev_close * 100

                            is_vol_shocker   = vol_ratio >= vol_mult_thresh
                            is_price_shocker = abs(price_chg) >= price_move_thresh

                            if (shocker_type == "Volume Shockers" and is_vol_shocker) or \
                               (shocker_type == "Price Shockers" and is_price_shocker) or \
                               (shocker_type == "Both" and (is_vol_shocker or is_price_shocker)):
                                tags = []
                                if is_vol_shocker:   tags.append(f"🌊 Vol {vol_ratio:.1f}×")
                                if is_price_shocker: tags.append(f"{'⬆️' if price_chg>0 else '⬇️'} {price_chg:+.2f}%")
                                in_wl = ticker in [w["ticker"] for w in st.session_state.get("pre_watchlist",[])]
                                shocker_results.append({
                                    "Ticker": ticker, "Name": name,
                                    "Price": format_indian_number(live_price, True),
                                    "Change %": f"{price_chg:+.2f}%",
                                    "Volume ×avg": round(vol_ratio, 1),
                                    "Tags": "  ".join(tags),
                                    "Watchlist": "⭐" if in_wl else "—",
                                    "_chg": price_chg, "_vol": vol_ratio,
                                })
                        except Exception:
                            continue
                    prog.empty()
                    status.empty()

                    if shocker_results:
                        shocker_results.sort(key=lambda x: abs(x["_chg"]) + x["_vol"], reverse=True)
                        st.success(f"✅ Found **{len(shocker_results)} shockers**")
                        wl_hits = [r for r in shocker_results if r["Watchlist"]=="⭐"]
                        if wl_hits:
                            st.success(f"⭐ **{len(wl_hits)} of your watchlist stocks are shockers — check immediately!**")
                        display_df = pd.DataFrame(shocker_results).drop(columns=["_chg","_vol"])
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

                        nse_shockers_links = {
                            "NSE Volume Shockers": "https://www.nseindia.com/market-data/most-active-securities",
                            "NSE Price Gainers": "https://www.nseindia.com/market-data/live-equity-market?series=EQ",
                            "BSE Top Gainers/Losers": "https://www.bseindia.com/markets/equity/EQReports/TopGainers_Losers.aspx",
                        }
                        st.markdown("#### 🔗 NSE Live Shocker Links")
                        for label, url in nse_shockers_links.items():
                            st.markdown(f"• [{label}]({url})")
                    else:
                        st.info("No shockers found with current thresholds. Try lowering them.")
                else:
                    st.info("Click 'Find Shockers' to scan for unusual volume or price activity.")
                    st.markdown("#### 🔗 NSE Real-Time Resources")
                    for label, url in {
                        "NSE Most Active (Volume)": "https://www.nseindia.com/market-data/most-active-securities",
                        "NSE Top Gainers/Losers": "https://www.nseindia.com/market-data/live-equity-market",
                        "BSE Top Movers": "https://www.bseindia.com/markets/equity/EQReports/TopGainers_Losers.aspx",
                    }.items():
                        st.markdown(f"• [{label}]({url})")

        # ─── TAB 6: TRADE EXECUTION LOG ────────────────────────
        with tabs[5]:
            st.markdown("### 📋 Live Trade Execution Log")
            st.markdown("Log trades as you execute them during the live session. Separate from the full Trade Journal.")

            with st.form("live_trade_log_form"):
                lc1, lc2, lc3 = st.columns(3)
                with lc1:
                    lt_ticker = st.text_input("Ticker", placeholder="RELIANCE.NS", key="lt_ticker")
                    lt_type   = st.radio("Type", ["BUY","SELL","SHORT","COVER"], horizontal=True, key="lt_type")
                with lc2:
                    lt_price  = st.number_input("Execution Price (₹)", min_value=0.0, step=0.5, key="lt_price")
                    lt_qty    = st.number_input("Qty", min_value=1, step=1, key="lt_qty")
                with lc3:
                    lt_sl     = st.number_input("Stop Loss (₹)", min_value=0.0, step=0.5, key="lt_sl")
                    lt_target = st.number_input("Target (₹)", min_value=0.0, step=0.5, key="lt_target")
                lt_reason = st.text_input("Reason / Setup", placeholder="ORB breakout, volume surge…", key="lt_reason")
                lt_submit = st.form_submit_button("📝 Log Trade", use_container_width=True)

            if lt_submit and lt_ticker and lt_price > 0:
                if "live_trade_log" not in st.session_state:
                    st.session_state.live_trade_log = []
                rr_live = 0
                if lt_type in ("BUY","COVER") and lt_target > lt_price > lt_sl > 0:
                    rr_live = (lt_target - lt_price) / (lt_price - lt_sl)
                elif lt_type in ("SELL","SHORT") and lt_target < lt_price < lt_sl:
                    rr_live = (lt_price - lt_target) / (lt_sl - lt_price)
                st.session_state.live_trade_log.append({
                    "time": now_ist.strftime("%H:%M:%S IST"),
                    "ticker": normalize_ticker(lt_ticker, st.session_state.exchange),
                    "type": lt_type,
                    "price": lt_price,
                    "qty": int(lt_qty),
                    "value": lt_price * lt_qty,
                    "sl": lt_sl,
                    "target": lt_target,
                    "rr": round(rr_live, 2),
                    "reason": lt_reason,
                    "pnl": None,
                })
                st.success(f"✅ Logged {lt_type} {int(lt_qty)} × {lt_ticker} @ {format_indian_number(lt_price, True)}")
                st.rerun()

            live_log = st.session_state.get("live_trade_log", [])
            if live_log:
                st.markdown(f"#### 📜 Today's Executions ({len(live_log)} trades)")
                total_value = sum(t["value"] for t in live_log)
                realized_pnl = sum(t["pnl"] for t in live_log if t["pnl"] is not None)
                lm1, lm2, lm3 = st.columns(3)
                lm1.metric("Total Trades", len(live_log))
                lm2.metric("Total Traded Value", format_indian_number(total_value))
                lm3.metric("Realized P&L", format_indian_number(realized_pnl))

                for i, trade in enumerate(reversed(live_log)):
                    idx = len(live_log) - 1 - i
                    type_color = "#00d084" if trade["type"] in ("BUY","COVER") else "#ff4444"
                    st.markdown(f"""<div class="metric-card" style="border-left:4px solid {type_color};margin-bottom:6px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
                            <div>
                                <span style="color:#888;font-size:0.8rem;">{trade['time']}</span>
                                <span style="margin-left:10px;font-weight:700;">{trade['ticker']}</span>
                                <span style="margin-left:8px;background:{type_color}22;color:{type_color};padding:1px 7px;border-radius:10px;font-size:0.8rem;">{trade['type']}</span>
                            </div>
                            <div style="color:#fff;">{trade['qty']} × {format_indian_number(trade['price'],True)} = {format_indian_number(trade['value'])}</div>
                        </div>
                        <div style="font-size:0.82rem;color:#aaa;margin-top:5px;">
                            SL: {format_indian_number(trade['sl'],True) if trade['sl'] else '–'} &nbsp;|&nbsp;
                            Target: {format_indian_number(trade['target'],True) if trade['target'] else '–'} &nbsp;|&nbsp;
                            R:R: {'1:'+str(trade['rr']) if trade['rr']>0 else '–'}
                            {' &nbsp;|&nbsp; Setup: '+trade['reason'] if trade['reason'] else ''}
                        </div>
                    </div>""", unsafe_allow_html=True)

                    # Mark as closed with exit price
                    if trade["pnl"] is None:
                        with st.expander(f"Mark {trade['ticker']} as closed", expanded=False):
                            exit_col1, exit_col2 = st.columns([2,1])
                            with exit_col1:
                                exit_price = st.number_input("Exit Price (₹)", min_value=0.0,
                                    step=0.5, key=f"exit_p_{idx}")
                            with exit_col2:
                                if st.button("✅ Close Trade", key=f"close_t_{idx}"):
                                    if exit_price > 0:
                                        multiplier = 1 if trade["type"] in ("BUY","COVER") else -1
                                        pnl = multiplier * (exit_price - trade["price"]) * trade["qty"]
                                        st.session_state.live_trade_log[idx]["pnl"] = pnl
                                        st.session_state.live_trade_log[idx]["exit_price"] = exit_price
                                        st.rerun()

                st.divider()
                if st.button("📊 Push to Trade Journal", use_container_width=True, key="push_to_journal"):
                    pushed = 0
                    for trade in live_log:
                        journal_entry = {
                            "date": now_ist.strftime("%Y-%m-%d"),
                            "stock": trade["ticker"],
                            "type": "Long" if trade["type"] in ("BUY","COVER") else "Short",
                            "entry": trade["price"],
                            "qty": trade["qty"],
                            "stop_loss": trade["sl"],
                            "target": trade["target"],
                            "strategy": trade.get("reason","Live Market"),
                            "notes": f"Executed at {trade['time']}",
                            "P&L": trade.get("pnl"),
                            "P&L %": round((trade["pnl"]/(trade["price"]*trade["qty"]))*100,2) if trade.get("pnl") else None,
                            "Exit": trade.get("exit_price"),
                        }
                        st.session_state.trade_journal.append(journal_entry)
                        pushed += 1
                    st.success(f"✅ Pushed {pushed} trades to Trade Journal!")

                csv_log = pd.DataFrame([{
                    "Time": t["time"], "Ticker": t["ticker"], "Type": t["type"],
                    "Price": t["price"], "Qty": t["qty"], "Value": t["value"],
                    "SL": t["sl"], "Target": t["target"], "R:R": t["rr"],
                    "Setup": t["reason"], "P&L": t.get("pnl",""),
                } for t in live_log]).to_csv(index=False)
                st.download_button("⬇️ Export Execution Log CSV", csv_log,
                    file_name=f"live_trades_{now_ist.strftime('%Y%m%d')}.csv",
                    mime="text/csv", key="live_log_csv")

                if st.button("🗑️ Clear Today's Log", key="clear_live_log"):
                    st.session_state.live_trade_log = []
                    st.rerun()
            else:
                st.info("No trades logged yet. Use the form above to log executions as you trade.")

    # -------------------------------------------------------
    # PAGE DISPATCH ROUTER
    # -------------------------------------------------------
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🌅 Pre-Market Prep":
        show_pre_market_prep()
    elif page == "⚡ Pre-Open Session (9–9:15 AM)":
        show_pre_open_session()
    elif page == "🔴 Live Market (After 9:15 AM)":
        show_live_market()
    elif page == "📊 Fundamental Analysis":
        show_fundamental_analysis()
    elif page == "📈 Technical Analysis":
        show_technical_analysis()
    elif page == "🧮 Options Analyzer":
        show_options_analyzer()
    elif page == "📈 Strategy Backtester":
        show_backtester()
    elif page == "💼 Portfolio Manager":
        show_portfolio_manager()
    elif page == "📝 Trade Journal":
        show_trade_journal()
    elif page == "🎯 Position Sizer":
        show_position_sizer()
    elif page == "💰 Tax Calculator":
        show_tax_calculator()
    elif page == "📉 Tax P&L Report":
        show_tax_pnl_report()
    elif page == "🔍 Stock Screener":
        show_stock_screener()
    elif page == "📉 Risk Analytics":
        show_risk_analytics()
    elif page == "📱 Quick Trade Setup":
        show_quick_trade_setup()
    elif page == "🔔 Alerts":
        show_alerts()
    elif page == "📈 Futures & Options":
        show_futures_options()
    elif page == "🇮🇳 India Market Hub":
        show_india_market_hub()
    elif page == "❓ FAQ":
        show_faq()
    elif page == "📚 Education Center":
        show_education_center()
    elif page == "🔀 Multi-Stock Comparison":
        show_multi_stock_comparison()
    elif page == "📅 Corporate Actions":
        show_corporate_actions()
    elif page == "🎯 Stock Selection Engine":
        show_stock_selection_engine()


# Main App Entry Point
if __name__ == "__main__":
    main()
