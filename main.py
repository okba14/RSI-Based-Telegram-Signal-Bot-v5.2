import os
import time
import math
import traceback
import logging
import signal
import sys
from typing import List, Dict, Optional, Tuple, Any
import ccxt
import pandas as pd
import numpy as np
import requests
from scipy.signal import argrelextrema
from dotenv import load_dotenv
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rsi_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    logger.info("Shutdown signal received. Finishing current cycle and stopping...")
    shutdown_requested = True

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --------------------------- CONFIG -------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET]):
    raise ValueError("Please set all required environment variables in .env file.")

EXCHANGE_ID = os.getenv('EXCHANGE_ID', 'binance')
TIMEFRAMES = os.getenv('TIMEFRAMES', '1h').split(',')
TIMEFRAME = TIMEFRAMES[0]
FETCH_LIMIT = int(os.getenv('FETCH_LIMIT', '500'))

RSI_SHORT = int(os.getenv('RSI_SHORT', '6'))
RSI_MID = int(os.getenv('RSI_MID', '14'))
RSI_LONG = int(os.getenv('RSI_LONG', '28'))
RSI_SMA_PERIOD = int(os.getenv('RSI_SMA_PERIOD', '20'))
ADX_PERIOD = int(os.getenv('ADX_PERIOD', '14'))
ATR_PERIOD = int(os.getenv('ATR_PERIOD', '14'))
BB_PERIOD = int(os.getenv('BB_PERIOD', '20'))
BB_STD = float(os.getenv('BB_STD', '2'))

# SuperTrend Settings
SUPERTREND_PERIOD = int(os.getenv('SUPERTREND_PERIOD', '10'))
SUPERTREND_MULTIPLIER = float(os.getenv('SUPERTREND_MULTIPLIER', '3.0'))
SUPERTREND_CHANGE_ATR = os.getenv('SUPERTREND_CHANGE_ATR', 'true').lower() == 'true'

# Squeeze Momentum Settings
SQUEEZE_BB_LENGTH = int(os.getenv('SQUEEZE_BB_LENGTH', '20'))
SQUEEZE_BB_MULT = float(os.getenv('SQUEEZE_BB_MULT', '2.0'))
SQUEEZE_KC_LENGTH = int(os.getenv('SQUEEZE_KC_LENGTH', '20'))
SQUEEZE_KC_MULT = float(os.getenv('SQUEEZE_KC_MULT', '1.5'))
SQUEEZE_USE_TRUE_RANGE = os.getenv('SQUEEZE_USE_TRUE_RANGE', 'true').lower() == 'true'

# Laguerre RSI Settings
LAGUERRE_GAMMA = float(os.getenv('LAGUERRE_GAMMA', '0.7'))

# Moving Average Converging Settings
MAC_LENGTH = int(os.getenv('MAC_LENGTH', '100'))
MAC_INCR = int(os.getenv('MAC_INCR', '10'))
MAC_FAST = int(os.getenv('MAC_FAST', '10'))

# SMC Settings
SMC_SWING_LENGTH = int(os.getenv('SMC_SWING_LENGTH', '50'))
SMC_INTERNAL_LENGTH = int(os.getenv('SMC_INTERNAL_LENGTH', '5'))
SMC_ORDER_BLOCKS_SIZE = int(os.getenv('SMC_ORDER_BLOCKS_SIZE', '5'))

# Signal Detection Settings
PREV_TREND_LOOKBACK = int(os.getenv('PREV_TREND_LOOKBACK', '20'))
EXHAUSTION_GAP = float(os.getenv('EXHAUSTION_GAP', '12'))
VOLUME_INCREASE_FACTOR = float(os.getenv('VOLUME_INCREASE_FACTOR', '1.15'))
MIN_CONFIDENCE_TO_ALERT = float(os.getenv('MIN_CONFIDENCE_TO_ALERT', '0.6'))
LOOP_SLEEP = float(os.getenv('LOOP_SLEEP', '30'))
ALERT_COOLDOWN_MINUTES = int(os.getenv('ALERT_COOLDOWN_MINUTES', '10'))

# New settings for market scanning
MARKET_SCAN_INTERVAL = int(os.getenv('MARKET_SCAN_INTERVAL', '300'))  # 5 minutes
API_CALL_DELAY = float(os.getenv('API_CALL_DELAY', '0.5'))

# Minimum data points required for reliable analysis
MIN_DATA_POINTS = int(os.getenv('MIN_DATA_POINTS', '200'))

# Developer Info
DEVELOPER_NAME = os.getenv('DEVELOPER_NAME', 'GUIAR-OQBA')
DEVELOPER_EMAIL = os.getenv('DEVELOPER_EMAIL', 'techokba@gmail.com')

# --------------------------- EXCHANGE SETUP ---------------------------------------
exchange = getattr(ccxt, EXCHANGE_ID)({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

# --------------------------- TELEGRAM HELPERS -------------------------------------

def telegram_send(text: str) -> bool:
    """Send message to Telegram with robust error handling."""
    if 'YOUR_TELEGRAM_BOT_TOKEN' in TELEGRAM_BOT_TOKEN:
        logger.info('Telegram token not set; printing message instead:\n%s', text)
        return True
    
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    # Use plain text to avoid parsing issues
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': text}
    
    try:
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()  # Raises an exception for bad status codes
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Telegram send failed: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while sending Telegram message: {e}")
        return False

# --------------------------- INDICATORS -------------------------------------------

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).mean()

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50)

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def bollinger_bands(series: pd.Series, period: int, std: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period).mean()
    sd = series.rolling(period).std()
    return mid + std * sd, mid, mid - std * sd

# =============================
#  New ADX Implementation (Corrected)
# =============================
def adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX, DI+, and DI- based on the provided implementation using Wilder's smoothing.
    """
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        true_range = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = np.where(
            (high - high.shift()) > (low.shift() - low),
            np.maximum(high - high.shift(), 0),
            0
        )
        dm_minus = np.where(
            (low.shift() - low) > (high - high.shift()),
            np.maximum(low.shift() - low, 0),
            0
        )
        
        # Use Wilder's smoothing (alpha = 1/period)
        alpha = 1.0 / period
        smoothed_true_range = true_range.ewm(alpha=alpha, adjust=False).mean()
        smoothed_dm_plus = pd.Series(dm_plus, index=df.index).ewm(alpha=alpha, adjust=False).mean()
        smoothed_dm_minus = pd.Series(dm_minus, index=df.index).ewm(alpha=alpha, adjust=False).mean()
        
        # Calculate DI+ and DI-
        di_plus = 100 * smoothed_dm_plus / smoothed_true_range
        di_minus = 100 * smoothed_dm_minus / smoothed_true_range
        
        # Calculate DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)
        adx_val = dx.ewm(alpha=alpha, adjust=False).mean()
        
        # Fill NaN values that result from division by zero or initial calculations
        di_plus = di_plus.fillna(0)
        di_minus = di_minus.fillna(0)
        adx_val = adx_val.fillna(0)
        
        return adx_val, di_plus, di_minus
    except Exception as e:
        logger.error(f"Error calculating ADX: {e}")
        n = len(df)
        return pd.Series(0, index=df.index), pd.Series(0, index=df.index), pd.Series(0, index=df.index)

# =============================
#  New SuperTrend Implementation (Corrected)
# =============================
def supertrend(df: pd.DataFrame, periods: int = SUPERTREND_PERIOD, multiplier: float = SUPERTREND_MULTIPLIER, 
               change_atr: bool = SUPERTREND_CHANGE_ATR) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate SuperTrend based on the provided implementation.
    Corrected to use .iloc for pandas compatibility.
    """
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate ATR
        if change_atr:
            atr_val = atr(df, periods)
        else:
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr_val = tr.rolling(window=periods).mean()
        
        # Calculate basic upper and lower bands
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr_val
        lower_band = hl2 - multiplier * atr_val
        
        # Initialize final bands with the same index as df
        final_upper = upper_band.copy()
        final_lower = lower_band.copy()
        
        # Calculate final bands using .iloc for position-based access
        for i in range(1, len(df)):
            if upper_band.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = upper_band.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]
                
            if lower_band.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = lower_band.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]
        
        # Calculate trend using .iloc
        trend = pd.Series(0, index=df.index)
        for i in range(len(df)):
            if i == 0:
                trend.iloc[i] = 1
            elif close.iloc[i] > final_upper.iloc[i-1]:
                trend.iloc[i] = 1
            elif close.iloc[i] < final_lower.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]
        
        # Calculate supertrend line
        supertrend_line = pd.Series(np.nan, index=df.index)
        for i in range(len(df)):
            if trend.iloc[i] == 1:
                supertrend_line.iloc[i] = final_upper.iloc[i]
            else:
                supertrend_line.iloc[i] = final_lower.iloc[i]
        
        # Calculate buy/sell signals
        buy_signal = (trend == 1) & (trend.shift() == -1)
        sell_signal = (trend == -1) & (trend.shift() == 1)
        
        return trend, supertrend_line, buy_signal, sell_signal
    except Exception as e:
        logger.error(f"Error calculating SuperTrend: {e}")
        n = len(df)
        return pd.Series(0, index=df.index), pd.Series(np.nan, index=df.index), pd.Series(False, index=df.index), pd.Series(False, index=df.index)

# =============================
#  Squeeze Momentum Indicator
# =============================
def squeeze_momentum(df: pd.DataFrame, bb_length: int = SQUEEZE_BB_LENGTH, bb_mult: float = SQUEEZE_BB_MULT,
                     kc_length: int = SQUEEZE_KC_LENGTH, kc_mult: float = SQUEEZE_KC_MULT,
                     use_true_range: bool = SQUEEZE_USE_TRUE_RANGE) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Squeeze Momentum Indicator based on the provided implementation
    """
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Calculate Bollinger Bands
        bb_basis = sma(close, bb_length)
        bb_dev = bb_mult * close.rolling(window=bb_length).std()
        bb_upper = bb_basis + bb_dev
        bb_lower = bb_basis - bb_dev
        
        # Calculate Keltner Channels
        kc_ma = sma(close, kc_length)
        if use_true_range:
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
        else:
            tr = high - low
        
        kc_range = sma(tr, kc_length)
        kc_upper = kc_ma + kc_range * kc_mult
        kc_lower = kc_ma - kc_range * kc_mult
        
        # Determine squeeze condition
        sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        sqz_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)
        no_sqz = (~sqz_on) & (~sqz_off)
        
        # Calculate momentum value
        highest_high = high.rolling(window=kc_length).max()
        lowest_low = low.rolling(window=kc_length).min()
        avg_high_low = (highest_high + lowest_low) / 2
        val = close - avg_high_low
        
        # Linear regression
        def linear_regression(series, period):
            x = np.arange(period)
            y = series.values
            if len(y) < period:
                return np.nan
            y = y[-period:]
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            if denominator == 0:
                return 0
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            return slope * (period - 1) + intercept
        
        linreg = pd.Series(index=close.index, dtype=float)
        for i in range(kc_length, len(close)):
            linreg.iloc[i] = linear_regression(close.iloc[i-kc_length+1:i+1], kc_length)
        
        # Determine colors
        bcolor = pd.Series(index=close.index, dtype=str)
        for i in range(1, len(close)):
            if linreg.iloc[i] > 0:
                if linreg.iloc[i] > linreg.iloc[i-1]:
                    bcolor.iloc[i] = 'lime'
                else:
                    bcolor.iloc[i] = 'green'
            else:
                if linreg.iloc[i] < linreg.iloc[i-1]:
                    bcolor.iloc[i] = 'red'
                else:
                    bcolor.iloc[i] = 'maroon'
        
        # Determine squeeze color
        scolor = pd.Series(index=close.index, dtype=str)
        for i in range(len(close)):
            if no_sqz.iloc[i]:
                scolor.iloc[i] = 'blue'
            elif sqz_on.iloc[i]:
                scolor.iloc[i] = 'black'
            else:
                scolor.iloc[i] = 'gray'
        
        return linreg, bcolor, scolor
    except Exception as e:
        logger.error(f"Error calculating Squeeze Momentum: {e}")
        n = len(df)
        return pd.Series(0, index=df.index), pd.Series('gray', index=df.index), pd.Series('gray', index=df.index)

# =============================
#  Smart Money Concepts (SMC)
# =============================
def smc(df: pd.DataFrame, swing_length: int = SMC_SWING_LENGTH, internal_length: int = SMC_INTERNAL_LENGTH,
        order_blocks_size: int = SMC_ORDER_BLOCKS_SIZE) -> Dict[str, Any]:
    """
    Calculate Smart Money Concepts indicators
    """
    try:
        # Initialize result dictionary
        result = {
            'swing_highs': pd.Series(np.nan, index=df.index),
            'swing_lows': pd.Series(np.nan, index=df.index),
            'internal_highs': pd.Series(np.nan, index=df.index),
            'internal_lows': pd.Series(np.nan, index=df.index),
            'bullish_order_blocks': pd.DataFrame(columns=['high', 'low', 'time']),
            'bearish_order_blocks': pd.DataFrame(columns=['high', 'low', 'time']),
            'bullish_structure': pd.Series(0, index=df.index),
            'bearish_structure': pd.Series(0, index=df.index),
            'trend': pd.Series(0, index=df.index)
        }
        
        # Calculate swing points
        swing_highs_idx = argrelextrema(df['high'].values, np.greater, order=swing_length)[0]
        swing_lows_idx = argrelextrema(df['low'].values, np.less, order=swing_length)[0]
        
        for idx in swing_highs_idx:
            result['swing_highs'].iloc[idx] = df['high'].iloc[idx]
        
        for idx in swing_lows_idx:
            result['swing_lows'].iloc[idx] = df['low'].iloc[idx]
        
        # Calculate internal points
        internal_highs_idx = argrelextrema(df['high'].values, np.greater, order=internal_length)[0]
        internal_lows_idx = argrelextrema(df['low'].values, np.less, order=internal_length)[0]
        
        for idx in internal_highs_idx:
            result['internal_highs'].iloc[idx] = df['high'].iloc[idx]
        
        for idx in internal_lows_idx:
            result['internal_lows'].iloc[idx] = df['low'].iloc[idx]
        
        # Calculate trend
        trend = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['close'].iloc[i] > result['swing_highs'].iloc[i-1]:
                trend.iloc[i] = 1
            elif df['close'].iloc[i] < result['swing_lows'].iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]
        
        result['trend'] = trend
        
        # Calculate order blocks
        bullish_order_blocks = []
        bearish_order_blocks = []
        
        # Find bullish order blocks (last down candle before an up move)
        for i in range(2, len(df)):
            if trend.iloc[i] == 1 and trend.iloc[i-1] == -1:
                # Find the last down candle
                for j in range(i-1, 0, -1):
                    if df['close'].iloc[j] < df['open'].iloc[j]:
                        bullish_order_blocks.append({
                            'high': df['high'].iloc[j],
                            'low': df['low'].iloc[j],
                            'time': df.index[j]
                        })
                        break
        
        # Find bearish order blocks (last up candle before a down move)
        for i in range(2, len(df)):
            if trend.iloc[i] == -1 and trend.iloc[i-1] == 1:
                # Find the last up candle
                for j in range(i-1, 0, -1):
                    if df['close'].iloc[j] > df['open'].iloc[j]:
                        bearish_order_blocks.append({
                            'high': df['high'].iloc[j],
                            'low': df['low'].iloc[j],
                            'time': df.index[j]
                        })
                        break
        
        # Keep only the most recent order blocks
        bullish_order_blocks = bullish_order_blocks[-order_blocks_size:]
        bearish_order_blocks = bearish_order_blocks[-order_blocks_size:]
        
        result['bullish_order_blocks'] = pd.DataFrame(bullish_order_blocks)
        result['bearish_order_blocks'] = pd.DataFrame(bearish_order_blocks)
        
        # Calculate structure breaks
        bullish_structure = pd.Series(0, index=df.index)
        bearish_structure = pd.Series(0, index=df.index)
        
        for i in range(1, len(df)):
            # Bullish structure break
            if trend.iloc[i] == 1 and trend.iloc[i-1] == -1:
                bullish_structure.iloc[i] = 1
            
            # Bearish structure break
            if trend.iloc[i] == -1 and trend.iloc[i-1] == 1:
                bearish_structure.iloc[i] = 1
        
        result['bullish_structure'] = bullish_structure
        result['bearish_structure'] = bearish_structure
        
        return result
    except Exception as e:
        logger.error(f"Error calculating SMC: {e}")
        return {
            'swing_highs': pd.Series(np.nan, index=df.index),
            'swing_lows': pd.Series(np.nan, index=df.index),
            'internal_highs': pd.Series(np.nan, index=df.index),
            'internal_lows': pd.Series(np.nan, index=df.index),
            'bullish_order_blocks': pd.DataFrame(columns=['high', 'low', 'time']),
            'bearish_order_blocks': pd.DataFrame(columns=['high', 'low', 'time']),
            'bullish_structure': pd.Series(0, index=df.index),
            'bearish_structure': pd.Series(0, index=df.index),
            'trend': pd.Series(0, index=df.index)
        }

# =============================
#  Laguerre RSI (LRSI)
# =============================
def calc_laguerre_rsi(df: pd.DataFrame, gamma: float = LAGUERRE_GAMMA) -> pd.DataFrame:
    """
    Calculate Laguerre RSI for smoother momentum signals
    Completely rewritten to handle dataframes with insufficient data
    """
    try:
        df = df.copy()
        n = len(df)
        close = df['close']
        
        # Initialize with default values
        df['lrsi'] = 0.5  # Default to neutral
        
        # If we don't have enough data, return with default values
        if n < 2:
            logger.warning(f"Not enough data for Laguerre RSI: need 2, got {n}")
            return df
        
        # Initialize arrays with same length as df
        L0 = np.zeros(n)
        L1 = np.zeros(n)
        L2 = np.zeros(n)
        L3 = np.zeros(n)
        lrsi = np.full(n, 0.5)  # Default to neutral
        
        for i in range(1, n):
            L0[i] = (1 - gamma) * close.iloc[i] + gamma * L0[i-1]
            L1[i] = -gamma * L0[i] + L0[i-1] + gamma * L1[i-1]
            L2[i] = -gamma * L1[i] + L1[i-1] + gamma * L2[i-1]
            L3[i] = -gamma * L2[i] + L2[i-1] + gamma * L3[i-1]
            
            cu = max(L0[i] - L1[i], 0) + max(L1[i] - L2[i], 0) + max(L2[i] - L3[i], 0)
            cd = max(L1[i] - L0[i], 0) + max(L2[i] - L1[i], 0) + max(L3[i] - L2[i], 0)
            
            if (cu + cd) != 0:
                lrsi[i] = cu / (cu + cd)
            else:
                lrsi[i] = lrsi[i-1] if i > 0 else 0.5
        
        # Create a Series with same index as df
        lrsi_series = pd.Series(lrsi, index=df.index)
        df['lrsi'] = lrsi_series
        return df
    except Exception as e:
        logger.error(f"Error calculating Laguerre RSI: {e}")
        # Add empty column with default value
        df['lrsi'] = 0.5  # Default to neutral
        return df

# =============================
#  Moving Average Converging [LuxAlgo]
# =============================
def calc_moving_average_converging(df: pd.DataFrame, length: int = MAC_LENGTH, 
                                  incr: int = MAC_INCR, 
                                  fast: int = MAC_FAST) -> pd.DataFrame:
    """
    Calculate Moving Average Converging indicator by LuxAlgo
    This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
    """
    try:
        df = df.copy()
        n = len(df)
        src = df['close']
        
        # Initialize with default values
        df['mac_ma'] = src  # Default to source price
        df['mac_fma'] = src  # Default to source price
        
        # If we don't have enough data, return with default values
        if n < length:
            logger.warning(f"Not enough data for Moving Average Converging: need {length}, got {n}")
            return df
        
        # Initialize arrays
        ma_values = np.zeros(n)
        fma_values = np.zeros(n)
        alpha_values = np.zeros(n)
        
        # Constants
        k = 1 / incr
        
        # Calculate initial values
        upper = src.rolling(length).max()
        lower = src.rolling(length).min()
        init_ma = src.rolling(length).mean()
        
        # Initialize first values
        ma_values[0] = src.iloc[0]
        fma_values[0] = src.iloc[0]
        alpha_values[0] = 2 / (length + 1)
        
        # Calculate values for each bar
        for i in range(1, n):
            # Calculate cross (price crossing the MA)
            cross = (src.iloc[i-1] <= ma_values[i-1] and src.iloc[i] > ma_values[i-1]) or \
                    (src.iloc[i-1] >= ma_values[i-1] and src.iloc[i] < ma_values[i-1])
            
            # Update alpha
            if cross:
                alpha_values[i] = 2 / (length + 1)
            elif src.iloc[i] > ma_values[i-1] and upper.iloc[i] > upper.iloc[i-1]:
                alpha_values[i] = min(1.0, alpha_values[i-1] + k)
            elif src.iloc[i] < ma_values[i-1] and lower.iloc[i] < lower.iloc[i-1]:
                alpha_values[i] = min(1.0, alpha_values[i-1] + k)
            else:
                alpha_values[i] = alpha_values[i-1]
            
            # Calculate converging MA
            if i < length:
                # Use SMA for initial period
                ma_values[i] = src.iloc[:i+1].mean()
            else:
                # Use adaptive MA after initial period
                ma_values[i] = ma_values[i-1] + alpha_values[i-1] * (src.iloc[i] - ma_values[i-1])
            
            # Calculate fast MA
            if cross:
                fma_values[i] = (src.iloc[i] + fma_values[i-1]) / 2
            elif src.iloc[i] > ma_values[i]:
                fma_values[i] = max(src.iloc[i], fma_values[i-1]) + (src.iloc[i] - fma_values[i-1]) / fast
            else:
                fma_values[i] = min(src.iloc[i], fma_values[i-1]) + (src.iloc[i] - fma_values[i-1]) / fast
        
        # Add to dataframe
        df['mac_ma'] = ma_values
        df['mac_fma'] = fma_values
        
        # Calculate MAC trend (1 for bullish, -1 for bearish)
        mac_trend = np.zeros(n)
        for i in range(1, n):
            if fma_values[i] > ma_values[i]:
                mac_trend[i] = 1
            else:
                mac_trend[i] = -1
        
        df['mac_trend'] = mac_trend
        
        return df
    except Exception as e:
        logger.error(f"Error calculating Moving Average Converging: {e}")
        # Add empty columns with default values
        df['mac_ma'] = df['close']
        df['mac_fma'] = df['close']
        df['mac_trend'] = 0
        return df

# --------------------------- DATA FETCHING ----------------------------------------

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = FETCH_LIMIT, retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data with retry mechanism
    Completely rewritten to handle pairs with insufficient data
    """
    for attempt in range(retries):
        try:
            if shutdown_requested: return None # Exit early if shutdown is requested
            
            # First, try to fetch the requested limit
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            
            # Check if we got any data
            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"No data received for {symbol}")
                return None
                
            # Create DataFrame with exact data we received
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Check if we have enough data
            if len(df) < MIN_DATA_POINTS:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} points, need at least {MIN_DATA_POINTS}")
                return None
                
            return df
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {symbol}: {e}")
            if attempt == retries-1:
                logger.error(f"Failed to fetch data for {symbol} after {retries} attempts")
                return None
            time.sleep(2 ** attempt)
    return None

# --------------------------- DYNAMIC SYMBOL FETCHING --------------------------------

def get_usdt_symbols() -> List[str]:
    """
    Dynamically fetches all active USDT spot trading pairs from Binance.
    """
    try:
        logger.info("Loading all markets from Binance...")
        markets = exchange.load_markets()
        usdt_symbols = [
            symbol for symbol, market in markets.items()
            if market['quote'] == 'USDT' and market['type'] == 'spot' and market['active']
        ]
        logger.info(f"Successfully found {len(usdt_symbols)} active USDT pairs.")
        return sorted(usdt_symbols)
    except Exception as e:
        logger.error(f"Failed to load markets from Binance: {e}")
        return []

# --------------------------- STRUCTURAL DETECTORS --------------------------------

def detect_previous_trend(df: pd.DataFrame, lookback: int = PREV_TREND_LOOKBACK) -> str:
    """
    Detect previous trend based on linear regression
    """
    # Ensure we don't try to access more data than we have
    actual_lookback = min(lookback, len(df))
    window = df['close'].iloc[-actual_lookback:]
    if len(window) < 3:
        return 'flat'
    x = np.arange(len(window))
    coeffs = np.polyfit(x, window.values, 1)
    slope = coeffs[0]
    slope_norm = slope / window.mean()
    if slope_norm > 0.0005:
        return 'up'
    if slope_norm < -0.0005:
        return 'down'
    return 'flat'

def last_swing_points(df: pd.DataFrame, order: int = 3) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Find the last swing high and low points
    """
    # Adjust order based on available data
    actual_order = min(order, len(df) // 4)
    highs = argrelextrema(df['high'].values, np.greater, order=actual_order)[0]
    lows = argrelextrema(df['low'].values, np.less, order=actual_order)[0]
    last_high = pd.Timestamp(df.index[highs[-1]]) if len(highs) else None
    last_low = pd.Timestamp(df.index[lows[-1]]) if len(lows) else None
    return last_high, last_low

# --------------------------- ADVANCED DETECTORS -----------------------------------

def detect_rsi_context_break(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Detect RSI context break signals
    """
    if 'rsi14' not in df.columns:
        return None
        
    rsi14 = df['rsi14']
    prev_trend = detect_previous_trend(df)
    last_rsi = float(rsi14.iloc[-1])
    price = float(df['close'].iloc[-1])
    result = None

    if prev_trend == 'down' and last_rsi >= 70:
        last_high_idx, _ = last_swing_points(df)
        last_swing_price = float(df['high'].loc[last_high_idx]) if last_high_idx is not None else None
        breakout = last_swing_price is not None and price > last_swing_price
        vol = float(df['volume'].iloc[-1])
        vol_prev = float(df['volume'].iloc[-4:-1].mean()) if len(df) > 4 else vol
        vol_ok = vol > vol_prev * VOLUME_INCREASE_FACTOR
        if breakout and vol_ok:
            result = {'type': 'context_break_buy', 'prev_trend': prev_trend, 'price': price, 'rsi14': last_rsi}

    if prev_trend == 'up' and last_rsi <= 30:
        _, last_low_idx = last_swing_points(df)
        last_swing_price = float(df['low'].loc[last_low_idx]) if last_low_idx is not None else None
        breakdown = last_swing_price is not None and price < last_swing_price
        vol = float(df['volume'].iloc[-1])
        vol_prev = float(df['volume'].iloc[-4:-1].mean()) if len(df) > 4 else vol
        vol_ok = vol > vol_prev * VOLUME_INCREASE_FACTOR
        if breakdown and vol_ok:
            result = {'type': 'context_break_sell', 'prev_trend': prev_trend, 'price': price, 'rsi14': last_rsi}

    return result

def detect_rsi_exhaustion(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Detect RSI exhaustion signals
    """
    if 'rsi14' not in df.columns or 'rsi14_sma' not in df.columns:
        return None
        
    rsi14 = df['rsi14']
    rsisma = df['rsi14_sma']
    gap = float(rsi14.iloc[-1] - rsisma.iloc[-1])
    if gap <= -EXHAUSTION_GAP:
        return {'type': 'exhaustion_buy', 'gap': gap, 'rsi14': float(rsi14.iloc[-1])}
    if gap >= EXHAUSTION_GAP:
        return {'type': 'exhaustion_sell', 'gap': gap, 'rsi14': float(rsi14.iloc[-1])}
    return None

def detect_liquidity_sweep(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Detect liquidity sweep signals
    """
    # Adjust lookback based on available data
    actual_lookback = min(30, len(df) // 3)
    window = df.iloc[-actual_lookback:]
    recent_high = float(window['high'].max())
    recent_low = float(window['low'].min())
    last = window.iloc[-1]
    if float(last['low']) < recent_low * 0.995 and float(last['close']) > recent_low:
        return {'type': 'sweep_low', 'pivot': recent_low}
    if float(last['high']) > recent_high * 1.005 and float(last['close']) < recent_high:
        return {'type': 'sweep_high', 'pivot': recent_high}
    return None

def detect_triple_divergence(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Detect triple divergence signals
    """
    if 'rsi14' not in df.columns:
        return None
        
    closes = df['close']
    rsi14 = df['rsi14']
    
    # Adjust order based on available data
    actual_order = min(3, len(df) // 10)
    lows_idx = argrelextrema(closes.values, np.less, order=actual_order)[0]
    highs_idx = argrelextrema(closes.values, np.greater, order=actual_order)[0]
    
    if len(lows_idx) >= 3:
        idxs = lows_idx[-3:]
        prices = closes.values[idxs]
        rsi_vals = rsi14.values[idxs]
        if (prices[0] > prices[1] > prices[2]) and (rsi_vals[0] < rsi_vals[1] < rsi_vals[2]):
            return {'type': 'triple_bull_div', 'points': idxs.tolist(), 'rsi': rsi_vals.tolist()}
    if len(highs_idx) >= 3:
        idxs = highs_idx[-3:]
        prices = closes.values[idxs]
        rsi_vals = rsi14.values[idxs]
        if (prices[0] < prices[1] < prices[2]) and (rsi_vals[0] > rsi_vals[1] > rsi_vals[2]):
            return {'type': 'triple_bear_div', 'points': idxs.tolist(), 'rsi': rsi_vals.tolist()}
    return None

def detect_acceleration(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Detect RSI acceleration signals
    """
    if 'rsi14' not in df.columns:
        return None
        
    rsi14 = df['rsi14']
    acc = rsi14.diff()
    if len(acc) < 3:
        return None
    if acc.iloc[-2] < 0 and acc.iloc[-1] > 0 and rsi14.iloc[-1] < 40:
        return {'type': 'acceleration_buy', 'acc': float(acc.iloc[-1]), 'rsi14': float(rsi14.iloc[-1])}
    if acc.iloc[-2] > 0 and acc.iloc[-1] < 0 and rsi14.iloc[-1] > 60:
        return {'type': 'acceleration_sell', 'acc': float(acc.iloc[-1]), 'rsi14': float(rsi14.iloc[-1])}
    return None

# --------------------------- COMPOSITE SCORING ------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators with error handling
    Completely rewritten to handle dataframes with insufficient data
    """
    try:
        df = df.copy()
        n = len(df)
        
        # Initialize all indicator columns with default values first
        df['rsi6'] = 50.0
        df['rsi14'] = 50.0
        df['rsi28'] = 50.0
        df['rsi14_sma'] = 50.0
        df['atr'] = 0.0
        df['adx'] = 0.0
        df['di_plus'] = 0.0
        df['di_minus'] = 0.0
        df['bb_upper'] = df['close']
        df['bb_mid'] = df['close']
        df['bb_lower'] = df['close']
        df['lrsi'] = 0.5
        df['mac_ma'] = df['close']
        df['mac_fma'] = df['close']
        df['mac_trend'] = 0
        
        # Calculate indicators only if we have enough data
        if n >= RSI_SHORT:
            try:
                rsi6_values = rsi(df['close'], RSI_SHORT)
                # Ensure length matches
                if len(rsi6_values) == n:
                    df['rsi6'] = rsi6_values
                else:
                    logger.warning(f"RSI6 length mismatch: expected {n}, got {len(rsi6_values)}")
            except Exception as e:
                logger.error(f"Error calculating RSI6: {e}")
        
        if n >= RSI_MID:
            try:
                rsi14_values = rsi(df['close'], RSI_MID)
                # Ensure length matches
                if len(rsi14_values) == n:
                    df['rsi14'] = rsi14_values
                else:
                    logger.warning(f"RSI14 length mismatch: expected {n}, got {len(rsi14_values)}")
            except Exception as e:
                logger.error(f"Error calculating RSI14: {e}")
        
        if n >= RSI_LONG:
            try:
                rsi28_values = rsi(df['close'], RSI_LONG)
                # Ensure length matches
                if len(rsi28_values) == n:
                    df['rsi28'] = rsi28_values
                else:
                    logger.warning(f"RSI28 length mismatch: expected {n}, got {len(rsi28_values)}")
            except Exception as e:
                logger.error(f"Error calculating RSI28: {e}")
        
        if n >= RSI_SMA_PERIOD:
            try:
                rsi14_sma_values = sma(df['rsi14'], RSI_SMA_PERIOD)
                # Ensure length matches
                if len(rsi14_sma_values) == n:
                    df['rsi14_sma'] = rsi14_sma_values
                else:
                    logger.warning(f"RSI14 SMA length mismatch: expected {n}, got {len(rsi14_sma_values)}")
            except Exception as e:
                logger.error(f"Error calculating RSI14 SMA: {e}")
        
        if n >= ATR_PERIOD:
            try:
                atr_values = atr(df, ATR_PERIOD)
                # Ensure length matches
                if len(atr_values) == n:
                    df['atr'] = atr_values
                else:
                    logger.warning(f"ATR length mismatch: expected {n}, got {len(atr_values)}")
            except Exception as e:
                logger.error(f"Error calculating ATR: {e}")
        
        if n >= ADX_PERIOD:
            try:
                adx_values, di_plus_values, di_minus_values = adx(df, ADX_PERIOD)
                # Ensure length matches
                if len(adx_values) == n and len(di_plus_values) == n and len(di_minus_values) == n:
                    df['adx'] = adx_values
                    df['di_plus'] = di_plus_values
                    df['di_minus'] = di_minus_values
                else:
                    logger.warning(f"ADX length mismatch")
            except Exception as e:
                logger.error(f"Error calculating ADX: {e}")
        
        if n >= BB_PERIOD:
            try:
                bb_upper_values, bb_mid_values, bb_lower_values = bollinger_bands(df['close'], BB_PERIOD, BB_STD)
                # Ensure length matches
                if len(bb_upper_values) == n and len(bb_mid_values) == n and len(bb_lower_values) == n:
                    df['bb_upper'] = bb_upper_values
                    df['bb_mid'] = bb_mid_values
                    df['bb_lower'] = bb_lower_values
                else:
                    logger.warning(f"Bollinger Bands length mismatch")
            except Exception as e:
                logger.error(f"Error calculating Bollinger Bands: {e}")
        
        # Calculate SuperTrend
        if n >= SUPERTREND_PERIOD:
            try:
                supertrend_trend, supertrend_line, supertrend_buy, supertrend_sell = supertrend(df)
                df['supertrend_trend'] = supertrend_trend
                df['supertrend_line'] = supertrend_line
                df['supertrend_buy'] = supertrend_buy
                df['supertrend_sell'] = supertrend_sell
            except Exception as e:
                logger.error(f"Error calculating SuperTrend: {e}")
                df['supertrend_trend'] = 0
                df['supertrend_line'] = np.nan
                df['supertrend_buy'] = False
                df['supertrend_sell'] = False
        
        # Calculate Squeeze Momentum
        if n >= max(SQUEEZE_BB_LENGTH, SQUEEZE_KC_LENGTH):
            try:
                squeeze_val, squeeze_bcolor, squeeze_scolor = squeeze_momentum(df)
                df['squeeze_val'] = squeeze_val
                df['squeeze_bcolor'] = squeeze_bcolor
                df['squeeze_scolor'] = squeeze_scolor
            except Exception as e:
                logger.error(f"Error calculating Squeeze Momentum: {e}")
                df['squeeze_val'] = 0
                df['squeeze_bcolor'] = 'gray'
                df['squeeze_scolor'] = 'gray'
        
        # Calculate SMC
        if n >= max(SMC_SWING_LENGTH, SMC_INTERNAL_LENGTH):
            try:
                smc_data = smc(df)
                df['smc_swing_high'] = smc_data['swing_highs']
                df['smc_swing_low'] = smc_data['swing_lows']
                df['smc_internal_high'] = smc_data['internal_highs']
                df['smc_internal_low'] = smc_data['internal_lows']
                df['smc_trend'] = smc_data['trend']
                df['smc_bullish_structure'] = smc_data['bullish_structure']
                df['smc_bearish_structure'] = smc_data['bearish_structure']
            except Exception as e:
                logger.error(f"Error calculating SMC: {e}")
                df['smc_swing_high'] = np.nan
                df['smc_swing_low'] = np.nan
                df['smc_internal_high'] = np.nan
                df['smc_internal_low'] = np.nan
                df['smc_trend'] = 0
                df['smc_bullish_structure'] = 0
                df['smc_bearish_structure'] = 0
        
        # Calculate Laguerre RSI (no minimum data requirement)
        try:
            df = calc_laguerre_rsi(df)
        except Exception as e:
            logger.error(f"Error calculating Laguerre RSI: {e}")
        
        # Calculate Moving Average Converging
        if n >= MAC_LENGTH:
            try:
                df = calc_moving_average_converging(df)
            except Exception as e:
                logger.error(f"Error calculating Moving Average Converging: {e}")
        
        return df
    except Exception as e:
        logger.error(f"Error in compute_indicators: {e}")
        # Add default values for all indicators
        df['rsi6'] = 50.0
        df['rsi14'] = 50.0
        df['rsi28'] = 50.0
        df['rsi14_sma'] = 50.0
        df['atr'] = 0.0
        df['adx'] = 0.0
        df['di_plus'] = 0.0
        df['di_minus'] = 0.0
        df['bb_upper'] = df['close']
        df['bb_mid'] = df['close']
        df['bb_lower'] = df['close']
        df['supertrend_trend'] = 0
        df['supertrend_line'] = np.nan
        df['supertrend_buy'] = False
        df['supertrend_sell'] = False
        df['squeeze_val'] = 0
        df['squeeze_bcolor'] = 'gray'
        df['squeeze_scolor'] = 'gray'
        df['smc_swing_high'] = np.nan
        df['smc_swing_low'] = np.nan
        df['smc_internal_high'] = np.nan
        df['smc_internal_low'] = np.nan
        df['smc_trend'] = 0
        df['smc_bullish_structure'] = 0
        df['smc_bearish_structure'] = 0
        df['lrsi'] = 0.5
        df['mac_ma'] = df['close']
        df['mac_fma'] = df['close']
        df['mac_trend'] = 0
        return df

def score_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Score signals based on multiple indicators with priority to new indicators
    """
    signals = []
    score = 0.0
    max_score = 15.0  # Increased max score to accommodate new indicators
    
    # Check if all required indicators are available
    required_indicators = ['rsi6', 'rsi14', 'rsi28', 'lrsi', 'supertrend_trend', 'adx', 'di_plus', 'di_minus', 
                          'atr', 'bb_upper', 'bb_lower', 'mac_trend', 'squeeze_val', 'squeeze_bcolor', 
                          'squeeze_scolor', 'smc_trend', 'smc_bullish_structure', 'smc_bearish_structure']
    missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
    
    if missing_indicators:
        logger.error(f"Missing indicators: {missing_indicators}")
        return {
            'signals': [], 
            'score': 0.0, 
            'confidence': 0.0, 
            'direction': 'neutral',
            'adx': 0.0, 
            'di_plus': 0.0,
            'di_minus': 0.0,
            'atr': 0.0, 
            'price': 0.0, 
            'rsi6': 0.0, 
            'rsi14': 0.0, 
            'rsi28': 0.0,
            'lrsi': 0.5,
            'supertrend_trend': 0,
            'mac_trend': 0,
            'squeeze_val': 0,
            'smc_trend': 0
        }
    
    # Get current values
    r6 = float(df['rsi6'].iloc[-1])
    r14 = float(df['rsi14'].iloc[-1])
    r28 = float(df['rsi28'].iloc[-1])
    lrsi = float(df['lrsi'].iloc[-1])
    supertrend_trend = float(df['supertrend_trend'].iloc[-1])
    mac_trend = float(df['mac_trend'].iloc[-1])
    squeeze_val = float(df['squeeze_val'].iloc[-1])
    squeeze_bcolor = df['squeeze_bcolor'].iloc[-1]
    squeeze_scolor = df['squeeze_scolor'].iloc[-1]
    smc_trend = float(df['smc_trend'].iloc[-1])
    smc_bullish_structure = float(df['smc_bullish_structure'].iloc[-1])
    smc_bearish_structure = float(df['smc_bearish_structure'].iloc[-1])
    
    # 1. SuperTrend (High Priority)
    if supertrend_trend == 1:
        signals.append(('supertrend_uptrend', 2.0))
        score += 2.0
    elif supertrend_trend == -1:
        signals.append(('supertrend_downtrend', 2.0))
        score += 2.0
    
    # 2. Squeeze Momentum (High Priority)
    if squeeze_val > 0 and squeeze_bcolor == 'lime':
        signals.append(('squeeze_momentum_buy', 1.8))
        score += 1.8
    elif squeeze_val < 0 and squeeze_bcolor == 'red':
        signals.append(('squeeze_momentum_sell', 1.8))
        score += 1.8
    
    # 3. SMC Trend (High Priority)
    if smc_trend == 1:
        signals.append(('smc_uptrend', 1.5))
        score += 1.5
    elif smc_trend == -1:
        signals.append(('smc_downtrend', 1.5))
        score += 1.5
    
    # 4. SMC Structure Break
    if smc_bullish_structure == 1:
        signals.append(('smc_bullish_structure', 1.5))
        score += 1.5
    elif smc_bearish_structure == 1:
        signals.append(('smc_bearish_structure', 1.5))
        score += 1.5
    
    # 5. Laguerre RSI
    if lrsi < 0.25:
        signals.append(('lrsi_oversold', 1.2))
        score += 1.2
    elif lrsi > 0.75:
        signals.append(('lrsi_overbought', 1.2))
        score += 1.2
    
    # 6. Moving Average Converging
    if mac_trend == 1:
        signals.append(('mac_bullish', 1.2))
        score += 1.2
    elif mac_trend == -1:
        signals.append(('mac_bearish', 1.2))
        score += 1.2
    
    # 7. ADX and DI
    adx_val = float(df['adx'].iloc[-1])
    di_plus_val = float(df['di_plus'].iloc[-1])
    di_minus_val = float(df['di_minus'].iloc[-1])
    
    if adx_val > 25:
        signals.append(('adx_trending', 0.5))
        score += 0.5
        
        if di_plus_val > di_minus_val:
            signals.append(('di_plus_above_di_minus', 0.5))
            score += 0.5
        else:
            signals.append(('di_minus_above_di_plus', 0.5))
            score += 0.5
    
    # 8. Multi-tempo RSI
    if r6 < 25 and r14 < 35 and r28 < 45:
        signals.append(('multi_rsi_buy', 1.0))
        score += 1.0
    if r6 > 75 and r14 > 65 and r28 > 55:
        signals.append(('multi_rsi_sell', 1.0))
        score += 1.0
    
    # 9. Combined SuperTrend + Squeeze + SMC signals (Highest Priority)
    if supertrend_trend == 1 and squeeze_val > 0 and smc_trend == 1:
        signals.append(('supertrend_squeeze_smc_buy', 3.0))
        score += 3.0
    elif supertrend_trend == -1 and squeeze_val < 0 and smc_trend == -1:
        signals.append(('supertrend_squeeze_smc_sell', 3.0))
        score += 3.0
    
    # 10. Context break
    ctx = detect_rsi_context_break(df)
    if ctx:
        if ctx['type'] == 'context_break_buy':
            signals.append(('context_break_buy', 1.0))
            score += 1.0
        if ctx['type'] == 'context_break_sell':
            signals.append(('context_break_sell', 1.0))
            score += 1.0
    
    # 11. Exhaustion
    ex = detect_rsi_exhaustion(df)
    if ex:
        if ex['type'] == 'exhaustion_buy':
            signals.append(('exhaustion_buy', 0.8))
            score += 0.8
        if ex['type'] == 'exhaustion_sell':
            signals.append(('exhaustion_sell', 0.8))
            score += 0.8
    
    # 12. Sweep
    sweep = detect_liquidity_sweep(df)
    if sweep:
        if sweep['type'] == 'sweep_low':
            signals.append(('sweep_low', 0.6))
            score += 0.6
        if sweep['type'] == 'sweep_high':
            signals.append(('sweep_high', 0.6))
            score += 0.6
    
    # 13. Triple div
    tri = detect_triple_divergence(df)
    if tri:
        if 'bull' in tri['type']:
            signals.append(('triple_bull_div', 1.0))
            score += 1.0
        else:
            signals.append(('triple_bear_div', 1.0))
            score += 1.0
    
    # 14. Acceleration
    acc = detect_acceleration(df)
    if acc:
        if 'buy' in acc['type']:
            signals.append(('acceleration_buy', 0.6))
            score += 0.6
        else:
            signals.append(('acceleration_sell', 0.6))
            score += 0.6
    
    # 15. Volatility
    atr_val = float(df['atr'].iloc[-1])
    price = float(df['close'].iloc[-1])
    bb_upper = float(df['bb_upper'].iloc[-1])
    bb_lower = float(df['bb_lower'].iloc[-1])
    
    if price < bb_lower:
        signals.append(('bb_below_lower', 0.5))
        score += 0.5
    if price > bb_upper:
        signals.append(('bb_above_upper', 0.5))
        score += 0.5
    
    # Calculate confidence and direction
    confidence = min(1.0, score / max_score)
    buy_weight = sum(w for (n, w) in signals if 'buy' in n or 'bull' in n or 'low' in n or 'oversold' in n or 'uptrend' in n)
    sell_weight = sum(w for (n, w) in signals if 'sell' in n or 'bear' in n or 'high' in n or 'overbought' in n or 'downtrend' in n)
    
    direction = 'neutral'
    if buy_weight > sell_weight and confidence >= MIN_CONFIDENCE_TO_ALERT:
        direction = 'buy'
    elif sell_weight > buy_weight and confidence >= MIN_CONFIDENCE_TO_ALERT:
        direction = 'sell'
    
    return {
        'signals': signals, 
        'score': score, 
        'confidence': confidence, 
        'direction': direction,
        'adx': adx_val,
        'di_plus': di_plus_val,
        'di_minus': di_minus_val,
        'atr': atr_val, 
        'price': price, 
        'rsi6': r6, 
        'rsi14': r14, 
        'rsi28': r28,
        'lrsi': lrsi,
        'supertrend_trend': supertrend_trend,
        'squeeze_val': squeeze_val,
        'squeeze_bcolor': squeeze_bcolor,
        'smc_trend': smc_trend,
        'mac_trend': mac_trend
    }

# --------------------------- TELEGRAM FORMATTING -------------------------------------

def format_alert(symbol: str, timeframe: str, result: Dict[str, Any]) -> str:
    """
    Format alert message for Telegram with enhanced UI
    """
    title = '🚀 Advanced RSI Signal'
    dir_sym = '🟢 BUY' if result['direction'] == 'buy' else '🔴 SELL' if result['direction'] == 'sell' else '⚪ NEUTRAL'
    
    # Create header
    lines = [
        f"{title} - {symbol} ({timeframe})",
        f"Direction: {dir_sym}",
        f"Price: {result['price']:.8f}",
        f"Confidence: {result['confidence']:.2f}",
        f"Score: {result['score']:.2f}",
        ""
    ]
    
    # Create indicators table
    indicators_data = [
        ['RSI 6/14/28', f"{result['rsi6']:.1f} / {result['rsi14']:.1f} / {result['rsi28']:.1f}"],
        ['Laguerre RSI', f"{result['lrsi']:.3f}"],
        ['SuperTrend', '🔼 Uptrend' if result['supertrend_trend'] == 1 else '🔽 Downtrend' if result['supertrend_trend'] == -1 else '➡️ Neutral'],
        ['Squeeze Momentum', f"{result['squeeze_val']:.4f} ({result['squeeze_bcolor']})"],
        ['SMC Trend', '🔼 Uptrend' if result['smc_trend'] == 1 else '🔽 Downtrend' if result['smc_trend'] == -1 else '➡️ Neutral'],
        ['MAC', '🔼 Bullish' if result['mac_trend'] == 1 else '🔽 Bearish' if result['mac_trend'] == -1 else '➡️ Neutral'],
        ['ADX', f"{result['adx']:.1f}"],
        ['DI+/DI-', f"{result['di_plus']:.1f}/{result['di_minus']:.1f}"],
        ['ATR', f"{result['atr']:.6f}"]
    ]
    
    indicators_table = tabulate(indicators_data, headers=['Indicator', 'Value'], tablefmt='grid')
    lines.append("📊 Technical Indicators:")
    lines.append(f"```\n{indicators_table}\n```")
    lines.append("")
    
    # Add signal components
    if result['signals']:
        lines.append("🧩 Signal Components:")
        for name, weight in result['signals']:
            lines.append(f"- {name} (weight={weight})")
        lines.append("")
    
    # Add TP/SL suggestions
    atr = result.get('atr', None)
    if atr and atr > 0:
        if result['direction'] == 'buy':
            sl = result['price'] - 1.5 * atr
            tp = result['price'] + 3 * atr
        elif result['direction'] == 'sell':
            sl = result['price'] + 1.5 * atr
            tp = result['price'] - 3 * atr
        else:
            sl = None
            tp = None
        
        if sl:
            lines.append(f"🛑 Suggested SL: {sl:.8f}")
        if tp:
            lines.append(f"🎯 Suggested TP: {tp:.8f}")
        lines.append("")
    
    # Add developer signature
    lines.append("")
    lines.append("Developed by: GUIAR-OQBA")
    lines.append("Email: techokba@gmail.com")
    lines.append("")
    lines.append("⚠️ Signals are for personal use only — backtest before trading.")
    
    return '\n'.join(lines)

# --------------------------- ANALYSIS LOOP ----------------------------------------

def analyze_symbol(symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
    """
    Analyze a symbol for trading signals
    """
    try:
        if shutdown_requested: return None
        df = fetch_ohlcv(symbol, timeframe)
        if df is None:
            return None
        
        # Check if we have enough data
        min_data_points = max(RSI_LONG, RSI_SMA_PERIOD, ADX_PERIOD, ATR_PERIOD, BB_PERIOD, 
                             SUPERTREND_PERIOD, MAC_LENGTH, SQUEEZE_BB_LENGTH, SQUEEZE_KC_LENGTH,
                             SMC_SWING_LENGTH, SMC_INTERNAL_LENGTH) * 2
        if len(df) < min_data_points:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} points, need at least {min_data_points}")
            return None
        
        df = compute_indicators(df)
        
        # Check if all indicators were calculated successfully
        if 'lrsi' not in df.columns or 'supertrend_trend' not in df.columns or 'mac_trend' not in df.columns:
            logger.error(f"Missing indicators for {symbol}")
            return None
        
        result = score_signals(df)
        return result
    except Exception as e:
        logger.error(f'Error analyzing {symbol}: {e}')
        return None

# --------------------------- MAIN LOOP ----------------------------------------

def main_loop():
    """
    Main loop for monitoring symbols and sending alerts
    """
    global shutdown_requested
    
    logger.info(f'Starting Advanced RSI Bot v5.2 - initializing market scan...')
    
    # Get all USDT symbols dynamically
    SYMBOLS = get_usdt_symbols()
    if not SYMBOLS:
        logger.error("Could not fetch any symbols. Bot cannot start.")
        return
    
    logger.info(f'Bot is now monitoring {len(SYMBOLS)} USDT pairs.')
    last_alerts: Dict[str, float] = {}
    cooldown = ALERT_COOLDOWN_MINUTES * 60  # Convert to seconds
    
    # Send a startup message to Telegram
    startup_msg = f"🚀 Advanced RSI Bot v5.2 Started\n\nMonitoring {len(SYMBOLS)} USDT pairs on Binance\n\nEnhanced with SuperTrend, Squeeze Momentum, and Smart Money Concepts\n\nDeveloped by: GUIAR-OQBA\nEmail: techokba@gmail.com"
    telegram_send(startup_msg)
    
    while not shutdown_requested:
        try:
            start_time = time.time()
            logger.info(f"Starting new market scan cycle for {len(SYMBOLS)} symbols...")
            processed_count = 0
            signals_found = 0
            
            for i, symbol in enumerate(SYMBOLS):
                if shutdown_requested: break
                for tf in TIMEFRAMES:
                    if shutdown_requested: break
                    key = f"{symbol}_{tf}"
                    result = analyze_symbol(symbol, tf)
                    
                    if not result:
                        continue
                    
                    if result['direction'] in ['buy', 'sell']:
                        signals_found += 1
                        now = time.time()
                        last_time = last_alerts.get(key, 0)
                        
                        if now - last_time > cooldown:
                            text = format_alert(symbol, tf, result)
                            success = telegram_send(text)
                            
                            if success:
                                last_alerts[key] = now
                                logger.info(f'🚨 ALERT SENT for {key} at {pd.Timestamp.now()}')
                            else:
                                logger.warning(f'Failed to send alert for {key}')
                    
                    time.sleep(API_CALL_DELAY)  # Delay between API calls
                processed_count += 1
                
                # Log progress every 50 symbols
                if processed_count % 50 == 0:
                    logger.info(f"Processed {processed_count}/{len(SYMBOLS)} symbols...")
            
            if not shutdown_requested:
                end_time = time.time()
                cycle_duration = end_time - start_time
                logger.info(f"Market scan cycle finished in {cycle_duration:.2f} seconds. Found {signals_found} potential signals. Sleeping for {MARKET_SCAN_INTERVAL} seconds.")
                time.sleep(MARKET_SCAN_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info('Interrupted by user')
            shutdown_requested = True
        except Exception as e:
            logger.error(f'Unhandled error in main loop: {e}')
            traceback.print_exc()
            time.sleep(10)
    
    # Send shutdown message after loop exits
    shutdown_msg = f"⏹️ Advanced RSI Bot v5.2 Stopped\n\nBot has been shut down.\n\nDeveloped by: GUIAR-OQBA\nEmail: techokba@gmail.com"
    telegram_send(shutdown_msg)
    logger.info("Bot shutdown complete.")

# --------------------------- BACKTEST HELPER --------------------------------------

def label_signals_on_history(symbol: str, timeframe: str, lookback: int = 1000) -> pd.DataFrame:
    """
    Label signals on historical data for backtesting
    """
    df = fetch_ohlcv(symbol, timeframe, limit=lookback)
    if df is None:
        return pd.DataFrame()
    
    df = compute_indicators(df)
    labels = []
    
    for i in range(30, len(df)):
        window = df.iloc[:i+1]
        res = score_signals(window)
        labels.append({
            'timestamp': df.index[i], 
            'direction': res['direction'], 
            'confidence': res['confidence'], 
            'score': res['score']
        })
    
    lab_df = pd.DataFrame(labels).set_index('timestamp')
    return lab_df

# --------------------------- ENTRY POINT ------------------------------------------
if __name__ == '__main__':
    main_loop()
