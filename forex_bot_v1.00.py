import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from datetime import date
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import json
import requests
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import queue
from abc import ABC, abstractmethod

MAGIC_NUMBER = 234000

plt.interactive(True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='[+] %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class AccountType(Enum):
    DEMO = "demo"
    LIVE = "live"

@dataclass
class NewsEvent:
    time: datetime
    currency: str
    impact: str
    event: str
    actual: str = ""
    forecast: str = ""
    previous: str = ""

@dataclass
class TechnicalLevels:
    resistance_levels: List[float]
    support_levels: List[float]
    current_price: float
    trend_direction: str
    strength: float

@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str

@dataclass
class TradeConfig:
    risk_reward_ratio: float = 2.0
    fixed_risk_amount: float = 10.0
    min_account_balance: float = 100.0
    max_risk_percentage: float = 0.10
    trailing_stop_points: int = 50
    account_type: AccountType = AccountType.DEMO
    dynamic_sizing: bool = True
    break_even_atr_multiplier: float = 2.0  # Move to BE after 2x ATR profit
    break_even_plus_atr_multiplier: float = 0.1  # Set SL 0.1x ATR beyond BE
    trailing_start_atr_multiplier: float = 3.0  # Start trailing after 3x ATR profit

@dataclass
class ForexPairs:
    majors: List[str] = field(default_factory=lambda: [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 
        'USDCAD', 'AUDUSD', 'NZDUSD'
    ])
    minors: List[str] = field(default_factory=lambda: [
        'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURNZD', 'EURCAD',
        'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPNZD', 'GBPCAD',
        'CHFJPY', 'AUDJPY', 'AUDCHF', 'AUDNZD', 'AUDCAD',
        'NZDJPY', 'NZDCHF', 'NZDCAD',
        'CADJPY', 'CADCHF'
    ])
    all: List[str] = field(init=False)
    
    def __post_init__(self):
        self.all = self.majors + self.minors
        
    def get_currencies(self) -> List[str]:
        """Extract unique currencies from all pairs"""
        currencies = set()
        for pair in self.all:
            currencies.add(pair[:3])
            currencies.add(pair[3:])
        return sorted(currencies)

class TechnicalIndicators:
    """Custom technical indicators to replace talib dependency"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

class TradeVisualizer:
    def __init__(self):
        self.plot_queue = queue.Queue()
        self.shutdown_flag = False
        self.current_figure = None

        """Initialize the MT5 connection and set up the visualizer"""
        if not mt5.initialize():
            print("MT5 initialization failed")
            raise ConnectionError("Failed to initialize MT5 connection")
        
        self.timezone = pytz.timezone("Etc/UTC")
        self.current_figure = None

    def _background_monitoring(self, check_interval=60):
        """Runs in a background thread to fetch trade data."""
        while not self.shutdown_flag:
            trades = mt5.positions_get()
            if trades:
                open_trades = pd.DataFrame(list(trades), columns=trades[0]._asdict().keys())
                open_trades['time'] = pd.to_datetime(open_trades['time'], unit='s')
                
                for instrument in open_trades['symbol']:
                    hist = self.get_instrument_historical(instrument)
                    self.plot_queue.put((open_trades, hist, instrument))
            
            time.sleep(check_interval)

    def start(self, check_interval=60, display_duration=10):
        """Starts monitoring in a background thread and processes plots in the main thread."""
        monitor_thread = threading.Thread(
            target=self._background_monitoring,
            kwargs={'check_interval': check_interval},
            daemon=True
        )
        monitor_thread.start()

        try:
            while True:
                try:
                    open_trades, hist, instrument = self.plot_queue.get(timeout=0.1)
                    self._update_plot(open_trades, hist, instrument, display_duration)
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            self.shutdown_flag = True

    def _update_plot(self, trades_df, hist, instrument, duration):
        """Updates and displays the plot, then closes it after `duration` seconds."""
        if self.current_figure is None:
            self.current_figure = plt.figure(figsize=(16, 6))
        else:
            plt.figure(self.current_figure.number)
            plt.clf()

        sns.lineplot(data=hist, x='time', y='close', color='black')
        plt.axhline(y=self.get_open_price(trades_df, instrument), color='blue')
        plt.axhline(y=self.get_stop_loss(trades_df, instrument), color='red')
        plt.axhline(y=self.get_take_profit(trades_df, instrument), color='green')
        plt.title(f"{instrument} - {self.get_trade_direction(trades_df, instrument)}")
        
        plt.show(block=False)
        plt.pause(duration)
        plt.close(self.current_figure)
        self.current_figure = None
    
    def __del__(self):
        """Clean up when the object is destroyed"""
        mt5.shutdown()
    
    def get_today(self):
        """Get today's date in UTC timezone"""
        current_date = str(date.today())
        current_year = int(current_date[:4])
        current_month = int(current_date[5:7])
        current_day = int(current_date[8:10])
        today = datetime(current_year, current_month, current_day, tzinfo=self.timezone)
        return today
    
    def get_instrument_historical(self, instrument):
        """Get historical data for an instrument"""
        today = self.get_today()
        hist = mt5.copy_rates_from(instrument, mt5.TIMEFRAME_D1, today, 90)
        hist_df = pd.DataFrame(hist)
        hist_df['time'] = pd.to_datetime(hist_df['time'], unit='s')
        del hist_df['tick_volume']
        del hist_df['spread']
        del hist_df['real_volume']
        return hist_df
    
    def get_open_price(self, trades_df, instrument):
        """Get the open price for a specific instrument from trades dataframe"""
        open_price_loc = trades_df[trades_df['symbol'] == instrument].index.values
        open_price = trades_df.iloc[open_price_loc]['price_open']
        return float(open_price.iloc[0])

    def get_stop_loss(self, trades_df, instrument):
        """Get the stop loss for a specific instrument from trades dataframe"""
        stop_loss_loc = trades_df[trades_df['symbol'] == instrument].index.values
        stop_loss = trades_df.iloc[stop_loss_loc]['sl']
        return float(stop_loss.iloc[0])

    def get_take_profit(self, trades_df, instrument):
        """Get the take profit for a specific instrument from trades dataframe"""
        take_profit_loc = trades_df[trades_df['symbol'] == instrument].index.values
        take_profit = trades_df.iloc[take_profit_loc]['tp']
        return float(take_profit.iloc[0])
    
    def get_trade_direction(self, trades_df, instrument):
        """Get the trade direction (Long/Short) for a specific instrument"""
        trade_direction_loc = trades_df[trades_df['symbol'] == instrument].index.values
        trade_direction_indicator = trades_df.iloc[trade_direction_loc]['type']
        trade_direction_indicator = str(trade_direction_indicator)
        trade_direction = int(trade_direction_indicator[5])
        if trade_direction == 0:
            return 'Long'
        elif trade_direction == 1:
            return 'Short'
        else:
            return 'Unknown'
    
    def show_graph(self, trades_df, instrument_history, ticker, duration=15):
        """Display the trade visualization graph"""
        open_price = self.get_open_price(trades_df, ticker)
        stop_loss = self.get_stop_loss(trades_df, ticker)
        take_profit = self.get_take_profit(trades_df, ticker)
        trade_direction = str(self.get_trade_direction(trades_df, ticker))
        symbol = str(ticker)
        
        if self.current_figure is None:
            self.current_figure = plt.figure(figsize=(16, 6))
        else:
            plt.figure(self.current_figure.number)
            plt.clf()
        
        sns.lineplot(data=instrument_history,
                     x=instrument_history['time'],
                     y=instrument_history['close'],
                     color='black')
        plt.axhline(y=open_price, color='blue', label='Open Price')
        plt.axhline(y=stop_loss, color='red', label='Stop Loss')
        plt.axhline(y=take_profit, color='green', label='Take Profit')
        plt.title(f'{symbol} - {trade_direction}')
        plt.legend()
        
        plt.show(block=False)
        plt.pause(duration)
    
    def update_trades(self, duration=15):
        """Check for open trades and update the visualization"""
        trades = mt5.positions_get()
        if not trades:
            print("No open trades.")
            return False
        else:
            open_trades = pd.DataFrame(list(trades),
                                       columns=trades[0]._asdict().keys())
            open_trades['time'] = pd.to_datetime(open_trades['time'],
                                               unit='s')
            
            for instrument in open_trades['symbol']:
                hist = self.get_instrument_historical(instrument)
                self.show_graph(open_trades, hist, instrument, duration)
            
            return True
    
    def continuous_monitoring(self, check_interval=60, display_duration=15):
        """Continuously monitor for trades and update the display"""
        print("Starting continuous monitoring...")
        try:
            while True:
                has_trades = self.update_trades(display_duration)
                if not has_trades:
                    time.sleep(check_interval)
        except KeyboardInterrupt:
            print("Stopping continuous monitoring...")

class SettingsOptimizer:
    """Optimizes trading settings based on account balance and currency pairs"""
    
    @staticmethod
    def get_optimized_config(account_balance: float, symbol: str) -> TradeConfig:
        """Get optimized settings based on account balance and symbol"""
        if account_balance < 100:
            base_config = TradeConfig(
                risk_reward_ratio=2.0,
                fixed_risk_amount=6.0,  # 6% at $100
                max_risk_percentage=0.06,
                min_account_balance=100.0,
                break_even_atr_multiplier=1.2,  # Very tight for tiny accounts
                break_even_plus_atr_multiplier=0.03,
                trailing_start_atr_multiplier=1.8,
                trailing_stop_points=35,
                dynamic_sizing=True
            )
        elif account_balance < 500:
            base_config = TradeConfig(
                risk_reward_ratio=2.0,
                fixed_risk_amount=4.0,  # 4% at $100, 0.8% at $500
                max_risk_percentage=0.04,
                min_account_balance=100.0,
                break_even_atr_multiplier=1.5,
                break_even_plus_atr_multiplier=0.05,
                trailing_start_atr_multiplier=2.0,
                trailing_stop_points=40,
                dynamic_sizing=True
            )
        elif account_balance < 2000:
            base_config = TradeConfig(
                risk_reward_ratio=2.0,
                fixed_risk_amount=20.0,  # 2% at $1000
                max_risk_percentage=0.02,
                min_account_balance=100.0,
                break_even_atr_multiplier=2.0,
                break_even_plus_atr_multiplier=0.1,
                trailing_start_atr_multiplier=3.0,
                trailing_stop_points=35,
                dynamic_sizing=True
            )
        else:
            base_config = TradeConfig(
                risk_reward_ratio=2.0,
                fixed_risk_amount=20.0,  # 1% at $2000
                max_risk_percentage=0.02,
                min_account_balance=100.0,
                break_even_atr_multiplier=2.5,
                break_even_plus_atr_multiplier=0.15,
                trailing_start_atr_multiplier=4.0,
                trailing_stop_points=30,
                dynamic_sizing=True
            )
        
        return SettingsOptimizer.adjust_for_currency_pair(base_config, symbol)
    
    @staticmethod
    def adjust_for_currency_pair(config: TradeConfig, symbol: str) -> TradeConfig:
        """Adjust settings based on currency pair characteristics"""
        jpy_multiplier = 0.8 if 'JPY' in symbol else 1.0  # Lower multiplier for JPY pairs
        config.break_even_atr_multiplier *= jpy_multiplier
        config.break_even_plus_atr_multiplier *= jpy_multiplier
        config.trailing_start_atr_multiplier *= jpy_multiplier
        config.trailing_stop_points = int(config.trailing_stop_points * jpy_multiplier)
        
        volatile_pairs = ['GBPJPY', 'EURGBP', 'GBPAUD', 'GBPNZD']
        if symbol in volatile_pairs:
            config.break_even_atr_multiplier *= 1.5
            config.break_even_plus_atr_multiplier *= 1.5
            config.trailing_start_atr_multiplier *= 1.5
            config.trailing_stop_points = int(config.trailing_stop_points * 1.5)
        
        conservative_pairs = ['EURUSD', 'USDCHF']
        if symbol in conservative_pairs:
            config.break_even_atr_multiplier *= 0.9
            config.break_even_plus_atr_multiplier *= 0.9
            config.trailing_start_atr_multiplier *= 0.9
            config.trailing_stop_points = int(config.trailing_stop_points * 0.9)
        
        return config
    
    @staticmethod
    def adjust_for_volatility(config: TradeConfig, high_volatility: bool) -> TradeConfig:
        """Adjust settings based on market volatility"""
        if high_volatility:
            config.break_even_atr_multiplier *= 1.5
            config.break_even_plus_atr_multiplier *= 1.5
            config.trailing_start_atr_multiplier *= 1.5
            config.trailing_stop_points = int(config.trailing_stop_points * 1.5)
        else:
            config.break_even_atr_multiplier *= 0.75
            config.break_even_plus_atr_multiplier *= 0.75
            config.trailing_start_atr_multiplier *= 0.75
            config.trailing_stop_points = int(config.trailing_stop_points * 0.75)
        
        return config

class NewsFilter:
    """Handles news events and trading restrictions"""
    
    def __init__(self):
        self.news_cache = {}
        self.cache_expiry = None
        self.api_key = None
    
    def get_news_events(self, hours_ahead: int = 4) -> List[NewsEvent]:
        """Get upcoming news events from ForexFactory"""
        try:
            now = datetime.now()
            if (self.cache_expiry and now < self.cache_expiry and 
                'events' in self.news_cache):
                return self.news_cache['events']
            
            events = self._get_sample_news_events(hours_ahead)
            logger.info(f"Fetched {len(events)} news events for next {hours_ahead} hours")

            self.news_cache['events'] = events
            self.cache_expiry = now + timedelta(minutes=30)
            
            return events
            
        except Exception as e:
            logger.error(f"Error fetching news events: {e}")
            return []
    
    def _get_sample_news_events(self, hours_ahead: int) -> List[NewsEvent]:
        """Sample news events - replace with actual news API"""
        sample_events = []
        try:
            high_impact_times = [
                ("08:30", "USD"),
                ("13:30", "USD"),
                ("09:30", "GBP"),
                ("08:00", "EUR"),
            ]
            
            base_time = datetime.now()
            for i, (time_str, currency) in enumerate(high_impact_times):
                event_time = base_time.replace(
                    hour=int(time_str.split(':')[0]),
                    minute=int(time_str.split(':')[1]),
                    second=0,
                    microsecond=0
                )
                
                if event_time < base_time:
                    event_time += timedelta(days=1)
                
                if event_time <= base_time + timedelta(hours=hours_ahead):
                    sample_events.append(NewsEvent(
                        time=event_time,
                        currency=currency,
                        impact="High",
                        event=f"Sample {currency} Economic Data"
                    ))
            
            return sample_events
            
        except Exception as e:
            logger.error(f"Error creating sample events: {e}")
            return []
    
    def should_avoid_currency(self, currency: str, minutes_before: int = 30, 
                            minutes_after: int = 60) -> tuple[bool, str]:
        """Check if we should avoid trading a currency due to upcoming news"""
        try:
            events = self.get_news_events(hours_ahead=6)
            now = datetime.now()
            
            for event in events:
                if event.currency == currency and event.impact == "High":
                    time_diff = (event.time - now).total_seconds() / 60
                    
                    if 0 <= time_diff <= minutes_before:
                        return True, f"High impact {currency} news in {int(time_diff)} minutes"
                    
                    if -minutes_after <= time_diff <= 0:
                        return True, f"High impact {currency} news {int(abs(time_diff))} minutes ago"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error checking news for {currency}: {e}")
            return False, ""
    
    def should_avoid_pair(self, symbol: str) -> tuple[bool, str]:
        """Check if we should avoid trading a pair due to news events"""
        if len(symbol) >= 6:
            base_currency = symbol[:3]
            quote_currency = symbol[3:6]
            
            avoid_base, reason_base = self.should_avoid_currency(base_currency)
            avoid_quote, reason_quote = self.should_avoid_currency(quote_currency)
            
            if avoid_base:
                return True, f"Base currency: {reason_base}"
            if avoid_quote:
                return True, f"Quote currency: {reason_quote}"
        
        return False, ""
    
    def close_risky_positions(self, mt5_bot) -> int:
        """Close positions that have upcoming news events"""
        closed_count = 0
        
        try:
            positions = mt5.positions_get()
            if positions is None:
                return 0
            
            for position in positions:
                if position.magic != MAGIC_NUMBER:
                    continue
                
                should_close, reason = self.should_avoid_pair(position.symbol)
                
                if should_close:
                    symbol_info = mt5.symbol_info(position.symbol)
                    if symbol_info is None:
                        continue
                    
                    if position.type == mt5.ORDER_TYPE_BUY:
                        close_price = symbol_info.bid
                        order_type = mt5.ORDER_TYPE_SELL
                    else:
                        close_price = symbol_info.ask
                        order_type = mt5.ORDER_TYPE_BUY
                    
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": position.symbol,
                        "volume": position.volume,
                        "type": order_type,
                        "position": position.ticket,
                        "price": close_price,
                        "deviation": 20,
                        "magic": MAGIC_NUMBER,
                        "comment": f"News-Close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"🚨 CLOSED {position.symbol} due to news: {reason}")
                        closed_count += 1
                    else:
                        error_msg = result.comment if result else "Unknown error"
                        logger.error(f"Failed to close {position.symbol}: {error_msg}")
        
        except Exception as e:
            logger.error(f"Error closing risky positions: {e}")
        
        return closed_count

class CurrencyStrengthAnalyzer:
    def __init__(self, pairs: ForexPairs):
        self.pairs = pairs
        self.currency_strength = {currency: 0.0 for currency in self.pairs.get_currencies()}
        self.strength_history = {currency: [] for currency in self.pairs.get_currencies()}
        self.history_length = 10
        self.last_calculation_time = None
        self.best_pairs = []
        self.historical_lookback = 200

    def calculate_strength(self, timeframe=mt5.TIMEFRAME_H4, lookback=100) -> bool:
        """Calculate currency strength based on RSI across pairs"""
        if self.last_calculation_time and (datetime.now() - self.last_calculation_time).total_seconds() < 300:
            return True

        try:
            # Initialize default strengths to avoid 0.0 for unprocessed currencies
            new_strength = {currency: 50.0 for currency in self.currency_strength}
            currency_counts = {currency: 0 for currency in self.currency_strength}

            # Populate strength history if empty
            if not any(self.strength_history[currency] for currency in self.strength_history):
                logger.info("Initializing currency strength history with historical data...")
                for pair in self.pairs.all:
                    rates = mt5.copy_rates_from_pos(pair, timeframe, 0, self.historical_lookback)
                    if rates is None or len(rates) < 20:
                        logger.warning(f"No historical data for {pair}. Skipping.")
                        continue

                    df = pd.DataFrame(rates)
                    base, quote = pair[:3], pair[3:]
                    for i in range(0, len(df), 10):
                        if i >= len(df):
                            break
                        rsi_data = df['close'].iloc[max(0, i-13):i+1]
                        rsi = TechnicalIndicators.rsi(rsi_data, 14).iloc[-1] if len(rsi_data) >= 14 else 50.0
                        if pd.isna(rsi):
                            rsi = 50.0
                            logger.debug(f"NaN RSI for {pair} at index {i}. Using default 50.0.")

                        for currency, value in [(base, rsi), (quote, 100 - rsi)]:
                            if currency in self.strength_history:
                                self.strength_history[currency].append(value)

                        if len(self.strength_history[base]) > self.history_length:
                            self.strength_history[base] = self.strength_history[base][-self.history_length:]
                        if len(self.strength_history[quote]) > self.history_length:
                            self.strength_history[quote] = self.strength_history[quote][-self.history_length:]

            # Calculate current strength
            for pair in self.pairs.all:
                rates = mt5.copy_rates_from_pos(pair, timeframe, 0, lookback)
                if rates is None or len(rates) < 20:
                    logger.warning(f"No data for {pair}. Skipping.")
                    continue

                df = pd.DataFrame(rates)
                rsi = TechnicalIndicators.rsi(df['close'], 14).iloc[-1]
                if pd.isna(rsi):
                    rsi = 50.0
                    logger.debug(f"NaN RSI for {pair}. Using default 50.0.")

                base, quote = pair[:3], pair[3:]
                new_strength[base] += rsi
                new_strength[quote] += (100 - rsi)
                currency_counts[base] += 1
                currency_counts[quote] += 1

            # Normalize strengths
            for currency in new_strength:
                if currency_counts[currency] > 0:
                    new_strength[currency] /= currency_counts[currency]
                else:
                    logger.warning(f"No data processed for {currency}. Using default strength 50.0.")
                    new_strength[currency] = 50.0

            # Scale to 0-100 range
            min_s = min(new_strength.values())
            max_s = max(new_strength.values())
            if max_s != min_s:
                for currency in new_strength:
                    new_strength[currency] = ((new_strength[currency] - min_s) / (max_s - min_s)) * 100
            else:
                logger.warning("All currencies have same strength. Setting all to 50.0.")
                for currency in new_strength:
                    new_strength[currency] = 50.0

            # Update history and current strengths
            for currency in self.strength_history:
                self.strength_history[currency].append(new_strength[currency])
                if len(self.strength_history[currency]) > self.history_length:
                    self.strength_history[currency].pop(0)

            self.currency_strength = new_strength
            self.last_calculation_time = datetime.now()

            logger.info("Currency Strength Scores:")
            for currency, score in sorted(self.currency_strength.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {currency}: {score:.1f}")

            self.best_pairs = self.rank_pairs_by_strength()
            return True

        except Exception as e:
            logger.error(f"Error calculating currency strength: {e}")
            for currency in self.currency_strength:
                self.currency_strength[currency] = 50.0
                self.strength_history[currency].append(50.0)
                if len(self.strength_history[currency]) > self.history_length:
                    self.strength_history[currency].pop(0)
            return False

    def get_strength_trend(self, currency: str) -> str:
        """Determine if strength is increasing, plateauing, or decreasing"""
        if len(self.strength_history[currency]) < 2:
            return "UNKNOWN"
        
        recent = self.strength_history[currency][-1]
        previous = self.strength_history[currency][-2]
        change = recent - previous
        
        if change > 2.0:
            return "INCREASING"
        elif change < -2.0:
            return "DECREASING"
        else:
            return "PLATEAUING"

    def rank_pairs_by_strength(self) -> List[Tuple[str, float]]:
        """Rank pairs by absolute strength differential between base and quote currencies."""
        pair_scores = []
        for pair in self.pairs.all:
            base, quote = pair[:3], pair[3:]
            if base in self.currency_strength and quote in self.currency_strength:
                diff = self.currency_strength[base] - self.currency_strength[quote]
                pair_scores.append((pair, diff))

        return sorted(pair_scores, key=lambda x: abs(x[1]), reverse=True)

    def get_best_pair_to_trade(self, bullish=True) -> Optional[Tuple[str, float]]:
        """Get the strongest pair based on strength differential."""
        for pair, diff in self.best_pairs:
            if (bullish and diff > 0) or (not bullish and diff < 0):
                return pair, diff
        return None

class Strategy(ABC):
    @abstractmethod
    def analyze(self, symbol: str, analyzer: 'ForexAnalyzer', strength_analyzer: CurrencyStrengthAnalyzer,
                df_w: pd.DataFrame, df_d1: pd.DataFrame, df_h4: pd.DataFrame, df_h2: pd.DataFrame,
                df_h1: pd.DataFrame, df_m30: pd.DataFrame, df_m15: pd.DataFrame) -> TradingSignal:
        """Analyze a symbol and return a trading signal"""
        pass

class ConfluenceStrategy(Strategy):
    def analyze(self, symbol: str, analyzer: 'ForexAnalyzer', strength_analyzer: CurrencyStrengthAnalyzer,
                df_w: pd.DataFrame, df_d1: pd.DataFrame, df_h4: pd.DataFrame, df_h2: pd.DataFrame,
                df_h1: pd.DataFrame, df_m30: pd.DataFrame, df_m15: pd.DataFrame) -> TradingSignal:
        """Implements the original confluence-based strategy from analyze_symbol"""
        if df_h1.empty:
            return TradingSignal(symbol, SignalType.HOLD, 0.0, 0.0, 0.0, 0.0, "No data available")
        
        # Calculate support/resistance and trends for all timeframes
        levels_w = analyzer.calculate_support_resistance(df_w)
        levels_d1 = analyzer.calculate_support_resistance(df_d1)
        levels_h4 = analyzer.calculate_support_resistance(df_h4)
        levels_h2 = analyzer.calculate_support_resistance(df_h2)
        levels_h1 = analyzer.calculate_support_resistance(df_h1)
        levels_m30 = analyzer.calculate_support_resistance(df_m30)
        levels_m15 = analyzer.calculate_support_resistance(df_m15)
        indicators = analyzer.calculate_technical_indicators(df_h1)

        # Log trend analysis for all timeframes
        logger.info(f"Analyzing {symbol} - W Trend: {levels_w.trend_direction}, Strength: {levels_w.strength}")
        logger.info(f"Analyzing {symbol} - D1 Trend: {levels_d1.trend_direction}, Strength: {levels_d1.strength}")
        logger.info(f"Analyzing {symbol} - H4 Trend: {levels_h4.trend_direction}, Strength: {levels_h4.strength}")
        logger.info(f"Analyzing {symbol} - H2 Trend: {levels_h2.trend_direction}, Strength: {levels_h2.strength}")
        logger.info(f"Analyzing {symbol} - H1 Trend: {levels_h1.trend_direction}, Strength: {levels_h1.strength}")
        logger.info(f"Analyzing {symbol} - M30 Trend: {levels_m30.trend_direction}, Strength: {levels_m30.strength}")
        logger.info(f"Analyzing {symbol} - M15 Trend: {levels_m15.trend_direction}, Strength: {levels_m15.strength}")

        limits = analyzer.get_broker_limits(symbol)
        if not limits:
            logger.error(f"Cannot get broker limits for {symbol}")
            return TradingSignal(symbol, SignalType.HOLD, 0.0, 0.0, 0.0, 0.0, "Broker limits not available")
        
        digits = limits.get('digits', 5)
        logger.info(f"Current Price: {round(float(levels_h4.current_price), digits)}, ATR: {indicators.get('atr', 'N/A')}")

        if not indicators or pd.isna(indicators.get('atr', 0)):
            return TradingSignal(symbol, SignalType.HOLD, 0.0, 0.0, 0.0, 0.0, "Insufficient data for indicators")
        
        current_price = levels_h4.current_price
        # Initialize default signal values
        signal_type = SignalType.HOLD
        entry_price = current_price
        stop_loss = 0.0
        take_profit = 0.0
        confidence = 0.0
        reason = ""
        
        trends = {
            'W': levels_w.trend_direction,
            'D1': levels_d1.trend_direction,
            'H4': levels_h4.trend_direction,
            'H2': levels_h2.trend_direction,
            'H1': levels_h1.trend_direction,
            'M30': levels_m30.trend_direction,
            'M15': levels_m15.trend_direction
        }
        
        trend_bullish, trend_bearish, trend_reason = analyzer.calculate_trend_confluence(
            trends['W'], trends['D1'], trends['H4'], trends['H2'], trends['H1'], trends['M30'], trends['M15']
        )
        bullish_signals = trend_bullish
        bearish_signals = trend_bearish
        reason += trend_reason
        
        if bullish_signals == 0 and bearish_signals == 0:
            reason = f"No trend agreement in W/D1/H4 with Group 2 confirmation | {reason}"
            return TradingSignal(symbol, SignalType.HOLD, current_price, 0.0, 0.0, 0.0, reason)
        
        rsi = indicators.get('rsi', 50)
        if pd.notna(rsi):
            if rsi < 30 and bullish_signals > 0:
                bullish_signals += 1
                reason += " | RSI oversold"
            elif rsi > 70 and bearish_signals > 0:
                bearish_signals += 1
                reason += " | RSI overbought"
        
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_hist = indicators.get('macd_hist', 0)
        
        if pd.notna(macd) and pd.notna(macd_signal) and pd.notna(macd_hist):
            if macd > macd_signal and macd_hist > 0 and bullish_signals > 0:
                bullish_signals += 1
                reason += " | MACD bullish crossover"
            elif macd < macd_signal and macd_hist < 0 and bearish_signals > 0:
                bearish_signals += 1
                reason += " | MACD bearish crossover"
        
        ema_12 = indicators.get('ema_12', 0)
        ema_26 = indicators.get('ema_26', 0)
        sma_20 = indicators.get('sma_20', 0)
        
        if pd.notna(ema_12) and pd.notna(ema_26) and pd.notna(sma_20):
            if ema_12 > ema_26 and current_price > sma_20 and bullish_signals > 0:
                bullish_signals += 1
                reason += " | EMA bullish alignment"
            elif ema_12 < ema_26 and current_price < sma_20 and bearish_signals > 0:
                bearish_signals += 1
                reason += " | EMA bearish alignment"

        base_currency = symbol[:3]
        quote_currency = symbol[3:6]
        base_strength = strength_analyzer.currency_strength.get(base_currency, 50.0)
        quote_strength = strength_analyzer.currency_strength.get(quote_currency, 50.0)
        strength_diff = base_strength - quote_strength
        
        strength_threshold = 10.0
        base_trend = strength_analyzer.get_strength_trend(base_currency)
        quote_trend = strength_analyzer.get_strength_trend(quote_currency)
        reason += f" | Currency Strength: {base_currency}: {base_strength:.1f} ({base_trend}) vs {quote_currency}: {quote_strength:.1f} ({quote_trend})"
        
        if strength_diff > strength_threshold and bullish_signals > 0:
            bullish_signals += 1
            if base_trend == "INCREASING":
                bullish_signals += 1
                reason += " | Rising base strength"
        elif strength_diff < -strength_threshold and bearish_signals > 0:
            bearish_signals += 1
            if quote_trend == "INCREASING":
                bearish_signals += 1
                reason += " | Rising quote strength"
        
        atr = indicators.get('atr', 0.0)

        if bullish_signals >= 3 and bearish_signals <= 1:
            signal_type = SignalType.BUY
            entry_price = current_price
            stop_loss, take_profit = analyzer.calculate_atr_stop_loss(df_h1, SignalType.BUY, current_price)
            confidence = min(bullish_signals * 15, 100)
            reason = f"Strong bullish confluence ({bullish_signals} bullish signals) | {reason}"
            
        elif bearish_signals >= 3 and bullish_signals <= 1:
            signal_type = SignalType.SELL
            entry_price = current_price
            stop_loss, take_profit = analyzer.calculate_atr_stop_loss(df_h1, SignalType.SELL, current_price)
            confidence = min(bearish_signals * 15, 100)
            reason = f"Strong bearish confluence ({bearish_signals} bearish signals) | {reason}"
        else:
            reason = f"Insufficient signal confluence (Bullish: {bullish_signals}, Bearish: {bearish_signals}) | {reason}"

        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason
        )

class ForexAnalyzer:
    def __init__(self):
        self.pairs = ForexPairs()
        self.strength_analyzer = CurrencyStrengthAnalyzer(ForexPairs())
        self.strategy = ConfluenceStrategy()  # Set default strategy to Confluence

    def get_all_symbols(self) -> List[str]:
        """Get all available currency pairs"""
        return self.pairs.all
    
    def get_broker_limits(self, symbol: str) -> Dict[str, float]:
        """Get broker-specific limits for a symbol"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {}
            
            return {
                'digits': symbol_info.digits
            }
        except Exception as e:
            logger.error(f"Error getting broker limits for {symbol}: {e}")
            return {}
    
    def get_ohlc_data(self, symbol: str, timeframe: int, bars: int = 1000) -> pd.DataFrame:
        """Get OHLC data for a symbol"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None:
                logger.error(f"Failed to get data for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            logger.error(f"Error getting OHLC data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_atr_stop_loss(self, df: pd.DataFrame, signal_type: SignalType, 
                            current_price: float, atr_period: int = 14) -> tuple[float, float]:
        """Calculate ATR-based stop loss and take profit with 2:1 risk-reward ratio"""
        try:
            if df.empty or len(df) < atr_period + 10:
                fallback_sl_pct = 0.015
                if signal_type == SignalType.BUY:
                    return (current_price * (1 - fallback_sl_pct), 
                            current_price * (1 + fallback_sl_pct * 2))
                else:
                    return (current_price * (1 + fallback_sl_pct), 
                            current_price * (1 - fallback_sl_pct * 2))
            
            atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'], atr_period)
            current_atr = atr.iloc[-1]
            
            if pd.isna(current_atr) or current_atr <= 0:
                recent_range = df['high'].tail(20).max() - df['low'].tail(20).min()
                current_atr = recent_range / 20
            
            atr_values = atr.tail(50).dropna()
            if len(atr_values) > 10:
                atr_mean = atr_values.mean()
                atr_std = atr_values.std()
                
                if atr_std > 0:
                    atr_z_score = (current_atr - atr_mean) / atr_std
                else:
                    atr_z_score = 0
                
                if atr_z_score > 1:
                    sl_multiplier = 2.8
                elif atr_z_score < -1:
                    sl_multiplier = 1.8
                else:
                    sl_multiplier = 2.2
            else:
                sl_multiplier = 2.2
            
            tp_multiplier = sl_multiplier * 2.0  # Enforce 2:1 risk-reward ratio
            atr_sl_distance = current_atr * sl_multiplier
            atr_tp_distance = current_atr * tp_multiplier
            
            if signal_type == SignalType.BUY:
                stop_loss = current_price - atr_sl_distance
                take_profit = current_price + atr_tp_distance
            else:
                stop_loss = current_price + atr_sl_distance
                take_profit = current_price - atr_tp_distance
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating ATR stop loss: {e}")
            fallback_sl_pct = 0.02
            if signal_type == SignalType.BUY:
                return (current_price * (1 - fallback_sl_pct), 
                        current_price * (1 + fallback_sl_pct * 2))
            else:
                return (current_price * (1 + fallback_sl_pct), 
                        current_price * (1 - fallback_sl_pct * 2))
            
    def _cluster_levels(self, levels: list, threshold: float = 0.005) -> list:
        """Group nearby levels into zones based on price proximity"""
        if not levels:
            return []

        levels = sorted(set(levels))
        clustered = []

        while levels:
            base = levels[0]
            cluster = [x for x in levels if abs(x - base) / base <= threshold]
            cluster_avg = sum(cluster) / len(cluster)
            clustered.append(round(cluster_avg, 2))
            levels = [x for x in levels if x not in cluster]

        return clustered

    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> TechnicalLevels:
        """Calculate clustered support and resistance levels with trend strength"""
        if df.empty or len(df) < window * 2:
            return TechnicalLevels([], [], 0, "UNKNOWN", 0)

        highs = df['high'].rolling(window=window).max()
        lows = df['low'].rolling(window=window).min()

        resistance_candidates = []
        support_candidates = []

        for i in range(window, len(df)):
            if df['high'].iloc[i] >= highs.iloc[i - 1]:
                resistance_candidates.append(df['high'].iloc[i])
            if df['low'].iloc[i] <= lows.iloc[i - 1]:
                support_candidates.append(df['low'].iloc[i])

        resistance_levels = self._cluster_levels(resistance_candidates)[-5:]
        support_levels = self._cluster_levels(support_candidates)[-5:]

        current_price = df['close'].iloc[-1]
        sma_20 = TechnicalIndicators.sma(df['close'], 20).iloc[-1]
        sma_50 = TechnicalIndicators.sma(df['close'], 50).iloc[-1]
        atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'], 14).iloc[-1]

        if pd.notna(sma_20) and pd.notna(sma_50) and pd.notna(atr):
            if sma_20 > sma_50 and current_price > sma_20:
                trend = "BULLISH"
                strength = min((sma_20 - sma_50) / atr * 10, 100)
            elif sma_20 < sma_50 and current_price < sma_20:
                trend = "BEARISH"
                strength = min((sma_50 - sma_20) / atr * 10, 100)
            else:
                trend = "SIDEWAYS"
                strength = 50
        else:
            trend = "SIDEWAYS"
            strength = 50

        return TechnicalLevels(
            resistance_levels=resistance_levels,
            support_levels=support_levels,
            current_price=current_price,
            trend_direction=trend,
            strength=round(strength, 2)
        )
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate various technical indicators"""
        if df.empty or len(df) < 50:
            return {}
        
        try:
            indicators = {}
            
            indicators['sma_20'] = TechnicalIndicators.sma(df['close'], 20).iloc[-1]
            indicators['sma_50'] = TechnicalIndicators.sma(df['close'], 50).iloc[-1]
            indicators['ema_12'] = TechnicalIndicators.ema(df['close'], 12).iloc[-1]
            indicators['ema_26'] = TechnicalIndicators.ema(df['close'], 26).iloc[-1]
            
            indicators['rsi'] = TechnicalIndicators.rsi(df['close']).iloc[-1]
            
            macd, macdsignal, macdhist = TechnicalIndicators.macd(df['close'])
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = macdsignal.iloc[-1]
            indicators['macd_hist'] = macdhist.iloc[-1]
            
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
            indicators['bb_upper'] = bb_upper.iloc[-1]
            indicators['bb_middle'] = bb_middle.iloc[-1]
            indicators['bb_lower'] = bb_lower.iloc[-1]
            
            indicators['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close']).iloc[-1]
            
            return indicators
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def calculate_trend_confluence(self, w: str, d1: str, h4: str, h2: str, h1: str, m30: str, m15: str) -> Tuple[int, int, str]:
        """Determine bullish/bearish signal weights based on multi-timeframe trend agreement."""
        bullish, bearish = 0, 0
        reason = ""

        # Group 1: Weekly (W), Daily (D1), 4-Hour (H4)
        # Require at least two consecutive timeframes to agree
        if w == d1 == "BULLISH":
            bullish += 2
            reason = "Medium bullish alignment (W/D1)"
        elif w == d1 == "BEARISH":
            bearish += 2
            reason = "Medium bearish alignment (W/D1)"
        elif d1 == h4 == "BULLISH":
            bullish += 2
            reason = "Medium bullish alignment (D1/H4)"
        elif d1 == h4 == "BEARISH":
            bearish += 2
            reason = "Medium bearish alignment (D1/H4)"
        elif w == d1 == h4 == "BULLISH":
            bullish += 3
            reason = "Strong bullish alignment (W/D1/H4)"
        elif w == d1 == h4 == "BEARISH":
            bearish += 3
            reason = "Strong bearish alignment (W/D1/H4)"
        elif w == "BULLISH":
            bullish += 1
            reason = "Weak bullish signal (W only)"
        elif w == "BEARISH":
            bearish += 1
            reason = "Weak bearish signal (W only)"

        # Group 2: 2-Hour (H2), 1-Hour (H1), 30-Min (M30), 15-Min (M15)
        # At least one timeframe must agree with Group 1's trend
        group2_timeframes = {"H2": h2, "H1": h1, "M30": m30, "M15": m15}
        group2_confirmation = False
        group2_reason = ""

        if bullish > 0:  # Group 1 indicates bullish
            confirming_timeframes = [tf for tf, trend in group2_timeframes.items() if trend == "BULLISH"]
            if confirming_timeframes:
                group2_confirmation = True
                bullish += len(confirming_timeframes)  # Add 1 signal per confirming timeframe
                group2_reason = f"Group 2 confirmation: {', '.join(confirming_timeframes)} bullish"
        elif bearish > 0:  # Group 1 indicates bearish
            confirming_timeframes = [tf for tf, trend in group2_timeframes.items() if trend == "BEARISH"]
            if confirming_timeframes:
                group2_confirmation = True
                bearish += len(confirming_timeframes)  # Add 1 signal per confirming timeframe
                group2_reason = f"Group 2 confirmation: {', '.join(confirming_timeframes)} bearish"

        # Combine reasons
        if group2_reason:
            reason = f"{reason} | {group2_reason}"
        elif bullish > 0 or bearish > 0:
            reason = f"{reason} | No Group 2 confirmation"

        # If no Group 2 confirmation, reset signals unless Group 1 has strong alignment
        if not group2_confirmation and bullish < 3 and bearish < 3:
            bullish, bearish = 0, 0
            reason = f"No Group 2 confirmation for weak/medium alignment | {reason}"

        return bullish, bearish, reason

    def analyze_symbol(self, symbol: str, strength_analyzer: CurrencyStrengthAnalyzer) -> TradingSignal:
            """Analyze a forex pair using the selected strategy"""
            # Fetch data for all timeframes
            df_w = self.get_ohlc_data(symbol, mt5.TIMEFRAME_W1, 100)
            df_d1 = self.get_ohlc_data(symbol, mt5.TIMEFRAME_D1, 200)
            df_h4 = self.get_ohlc_data(symbol, mt5.TIMEFRAME_H4, 300)
            df_h2 = self.get_ohlc_data(symbol, mt5.TIMEFRAME_H2, 400)
            df_h1 = self.get_ohlc_data(symbol, mt5.TIMEFRAME_H1, 500)
            df_m30 = self.get_ohlc_data(symbol, mt5.TIMEFRAME_M30, 600)
            df_m15 = self.get_ohlc_data(symbol, mt5.TIMEFRAME_M15, 800)
            
            # Delegate analysis to the strategy
            return self.strategy.analyze(
                symbol, self, strength_analyzer,
                df_w, df_d1, df_h4, df_h2, df_h1, df_m30, df_m15
        )

class MT5TradingBot:
    def __init__(self, config: TradeConfig):
        self.base_config = config
        self.config = config
        self.active_trades = {}
        self.news_filter = NewsFilter()
        self.settings_optimizer = SettingsOptimizer()

    def initialize_mt5(self, login: int = None, password: str = None, server: str = None) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                logger.error("Failed to initialize MT5")
                return False
            
            if login and password and server:
                if not mt5.login(login, password, server):
                    logger.error(f"Failed to login to MT5: {mt5.last_error()}")
                    return False
            
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return False
            
            terminal_info = mt5.terminal_info()
            if terminal_info is not None:
                autotrading_status = "ENABLED" if terminal_info.trade_allowed else "❌ DISABLED"
                logger.info(f"AutoTrading Status: {autotrading_status}")
                
                if not terminal_info.trade_allowed:
                    logger.warning("AutoTrading is DISABLED. Bot will not be able to place trades.")
                    logger.warning(" To enable: Tools → Options → Expert Advisors → Allow automated trading")
                    logger.warning(" Or click the 'AutoTrading' button in MT5 toolbar")
            
            logger.info(f"Connected to MT5 - Account: {account_info.login}, Balance: {account_info.balance}")
            return True
        except Exception as e:
            logger.error(f"Error initializing MT5: {e}")
            return False
    
    def get_ohlc_data_for_trailing(self, symbol: str) -> pd.DataFrame:
        """Get OHLC data specifically for trailing stop calculations"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
            if rates is None:
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            logger.error(f"Error getting OHLC data for trailing {symbol}: {e}")
            return pd.DataFrame()

    def calculate_atr_trailing_distance(self, symbol: str, position_type: int) -> float:
        """Calculate dynamic trailing distance based on ATR"""
        try:
            df = self.get_ohlc_data_for_trailing(symbol)
            if df.empty:
                return self.config.trailing_stop_points
            
            atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'], 14)
            current_atr = atr.iloc[-1]
            
            if pd.isna(current_atr) or current_atr <= 0:
                return self.config.trailing_stop_points
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return self.config.trailing_stop_points
            
            atr_trailing_multiplier = 1.5
            atr_distance_price = current_atr * atr_trailing_multiplier
            atr_distance_points = atr_distance_price / symbol_info.point
            
            min_trailing = self.config.trailing_stop_points * 0.7
            max_trailing = self.config.trailing_stop_points * 2.0
            
            return max(min_trailing, min(atr_distance_points, max_trailing))
            
        except Exception as e:
            logger.error(f"Error calculating ATR trailing distance for {symbol}: {e}")
            return self.config.trailing_stop_points

    def update_settings_for_symbol(self, symbol: str):
        """Update settings based on current account balance and symbol"""
        current_balance = self.get_account_balance()
        
        optimized_config = self.settings_optimizer.get_optimized_config(current_balance, symbol)
        
        high_volatility = self._is_high_volatility_period()
        if high_volatility:
            optimized_config = self.settings_optimizer.adjust_for_volatility(optimized_config, True)
        
        self.config = optimized_config
        
        logger.info(f"Updated settings for {symbol} (Balance: ${current_balance:.2f}): "
                    f"BE={self.config.break_even_atr_multiplier:.2f}x ATR, "
                    f"Trail={self.config.trailing_stop_points}pts, "
                    f"Risk=${self.config.fixed_risk_amount}")

    def _is_high_volatility_period(self) -> bool:
        """Detect high volatility periods"""
        try:
            now = datetime.now()
            if (8 <= now.hour <= 10) or (13 <= now.hour <= 16):
                return True
            
            weekday = now.weekday()
            if weekday == 4 and 12 <= now.hour <= 16:
                return True
            
            return False
        except:
            return False
            
    def get_account_balance(self) -> float:
        """Get current account balance from MT5"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Cannot get account info")
                return 0.0
            
            return float(account_info.equity)
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 0.0
    
    def get_broker_limits(self, symbol: str) -> Dict[str, float]:
        """Get broker-specific limits for a symbol"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {}
            
            return {
                'min_lot': symbol_info.volume_min,
                'max_lot': symbol_info.volume_max,
                'lot_step': symbol_info.volume_step,
                'point': symbol_info.point,
                'digits': symbol_info.digits,
                'tick_value': symbol_info.trade_tick_value,
                'tick_size': symbol_info.trade_tick_size
            }
        except Exception as e:
            logger.error(f"Error getting broker limits for {symbol}: {e}")
            return {}
    
    def calculate_position_size(self, symbol: str, signal: TradingSignal) -> float:
        """Calculate position size based on account balance and risk management"""
        try:
            current_balance = self.get_account_balance()
            
            if current_balance < self.config.min_account_balance:
                logger.warning(f"Account balance ${current_balance:.2f} below minimum ${self.config.min_account_balance}")
                return 0.0
            
            limits = self.get_broker_limits(symbol)
            if not limits:
                logger.error(f"Cannot get broker limits for {symbol}")
                return limits.get('min_lot', 0.01)
            
            if self.config.dynamic_sizing and current_balance > 500:
                risk_amount = current_balance * 0.02
            else:
                base_risk = self.config.fixed_risk_amount
                risk_multiplier = max(1.0, current_balance / self.config.min_account_balance)
                risk_amount = min(base_risk * risk_multiplier, current_balance * self.config.max_risk_percentage)
            
            stop_loss_distance = abs(signal.entry_price - signal.stop_loss)
            
            if stop_loss_distance == 0:
                logger.warning(f"Stop loss distance is zero for {symbol}")
                return limits['min_lot']
            
            if limits['tick_value'] > 0 and limits['tick_size'] > 0:
                ticks_in_stop_loss = stop_loss_distance / limits['tick_size']
                loss_per_lot = ticks_in_stop_loss * limits['tick_value']
                
                if loss_per_lot > 0:
                    calculated_lots = risk_amount / loss_per_lot
                    
                    lot_step = limits['lot_step']
                    calculated_lots = round(calculated_lots / lot_step) * lot_step
                    
                    min_lot = limits['min_lot']
                    max_lot = limits['max_lot']
                    final_lots = max(min_lot, min(calculated_lots, max_lot))
                    
                    actual_risk = final_lots * loss_per_lot
                    max_allowed_risk = current_balance * self.config.max_risk_percentage
                    
                    if actual_risk > max_allowed_risk:
                        final_lots = (max_allowed_risk / loss_per_lot)
                        final_lots = max(min_lot, round(final_lots / lot_step) * lot_step)
                    
                    logger.info(f"Position sizing for {symbol}: Balance=${current_balance:.2f}, "
                              f"Risk=${risk_amount:.2f}, Lots={final_lots:.3f}, "
                              f"Actual Risk=${final_lots * loss_per_lot:.2f}")
                    
                    return final_lots
            
            logger.warning(f"Using minimum lot size for {symbol}")
            return limits['min_lot']
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.01
    
    def place_trade(self, signal: TradingSignal) -> bool:
        """Place a trade based on the signal"""
        if signal.signal_type == SignalType.HOLD:
            return False
        
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.error("Cannot get terminal info")
                return False
                
            if not terminal_info.trade_allowed:
                logger.error("AUTOTRADING DISABLED: Please enable AutoTrading in MT5 terminal")
                logger.error("   → Go to Tools → Options → Expert Advisors → Allow automated trading")
                logger.error("   → Click the 'AutoTrading' button in MT5 toolbar")
                return False
            
            volume = self.calculate_position_size(signal.symbol, signal)
            
            if volume <= 0:
                logger.warning(f"Position size calculation returned {volume} for {signal.symbol}")
                return False
            
            if not mt5.symbol_select(signal.symbol, True):
                logger.error(f"Failed to select symbol {signal.symbol}")
                return False
            
            symbol_info = mt5.symbol_info(signal.symbol)
            if symbol_info is None:
                logger.error(f"Cannot get symbol info for {signal.symbol}")
                return False

            if not symbol_info.visible:
                logger.warning(f"Market closed for {signal.symbol}")
                return False

            tick_info = mt5.symbol_info_tick(signal.symbol)
            if tick_info is None:
                logger.error(f"Cannot get tick info for {signal.symbol}")
                return False
            
            if signal.signal_type == SignalType.BUY:
                trade_type = mt5.ORDER_TYPE_BUY
                price = tick_info.ask
            else:
                trade_type = mt5.ORDER_TYPE_SELL
                price = tick_info.bid

            digits = symbol_info.digits
            
            raw_stop_level = symbol_info.trade_stops_level
            min_stop_level = raw_stop_level * symbol_info.point

            if raw_stop_level == 0:
                default_points = 10
                min_stop_level = default_points * symbol_info.point
                logger.info(f"{signal.symbol}: No stop level defined by broker. Using fallback: {min_stop_level:.5f} ({default_points} points)")
            else:
                logger.info(f"{signal.symbol}: Broker-defined min stop level is {raw_stop_level} points → {min_stop_level:.5f}")
            
            validated_sl = signal.stop_loss
            validated_tp = signal.take_profit
            
            if signal.signal_type == SignalType.BUY:
                if price - validated_sl < min_stop_level:
                    validated_sl = price - min_stop_level
                    logger.warning(f"Adjusted BUY SL for {signal.symbol}: {signal.stop_loss:.5f} → {validated_sl:.5f}")
                
                if validated_tp - price < min_stop_level:
                    validated_tp = price + min_stop_level
                    logger.warning(f"Adjusted BUY TP for {signal.symbol}: {signal.take_profit:.5f} → {validated_tp:.5f}")
            
            else:
                if validated_sl - price < min_stop_level:
                    validated_sl = price + min_stop_level
                    logger.warning(f"Adjusted SELL SL for {signal.symbol}: {signal.stop_loss:.5f} → {validated_sl:.5f}")
                
                if price - validated_tp < min_stop_level:
                    validated_tp = price - min_stop_level
                    logger.warning(f"Adjusted SELL TP for {signal.symbol}: {signal.take_profit:.5f} → {validated_tp:.5f}")
            
            if validated_sl <= 0:
                if signal.signal_type == SignalType.BUY:
                    validated_sl = price - min_stop_level
                else:
                    validated_sl = price + min_stop_level
                logger.warning(f"Fixed zero SL for {signal.symbol}: {validated_sl:.5f}")
            
            if validated_tp <= 0:
                if signal.signal_type == SignalType.BUY:
                    validated_tp = price + min_stop_level
                else:
                    validated_tp = price - min_stop_level
                logger.warning(f"Fixed zero TP for {signal.symbol}: {validated_tp:.5f}")
            
            logger.info(f"Trade validation for {signal.symbol}:")
            logger.info(f"   Price: {price:.5f}, Min Stop Level: {min_stop_level:.5f}")
            logger.info(f"   Original SL: {signal.stop_loss:.5f} → Validated SL: {validated_sl:.5f}")
            logger.info(f"   Original TP: {signal.take_profit:.5f} → Validated TP: {validated_tp:.5f}")
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": round(volume, 2),
                "type": trade_type,
                "price": round(float(price), digits),
                "sl": round(float(validated_sl), digits),
                "tp": round(float(validated_tp), digits),
                "deviation": 20,
                "magic": MAGIC_NUMBER,
                "comment": f"Bot-{signal.confidence:.0f}%",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            logger.info(f"Sending trade request: {request}")

            result = mt5.order_send(request)

            if result is None:
                last_error = mt5.last_error()
                logger.error(f"Failed to send order for {signal.symbol}: No result returned from MT5. Last error: {last_error}")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                if result.retcode == 10027:
                    logger.error("AUTOTRADING DISABLED: Enable AutoTrading in MT5 terminal")
                    logger.error("   → Tools → Options → Expert Advisors → Allow automated trading ✓")
                    logger.error("   → Click 'AutoTrading' button in toolbar (should be green)")
                    return False
                elif result.retcode == 10006:
                    logger.error(f"Trade rejected for {signal.symbol}: {result.comment}")
                    return False
                elif result.retcode == 10013:
                    logger.error(f"Invalid request for {signal.symbol}: {result.comment}")
                    return False
                elif result.retcode == 10016:
                    logger.error(f"Invalid stops for {signal.symbol}: {result.comment}")
                    logger.error(f"   Current price: {price:.5f}")
                    logger.error(f"   Stop Loss: {validated_sl:.5f}")
                    logger.error(f"   Take Profit: {validated_tp:.5f}")
                    logger.error(f"   Min stop level: {min_stop_level:.5f}")
                    return False
                else:
                    logger.error(f"Failed to place trade for {signal.symbol}: {result.comment} (Code: {result.retcode})")
                
                if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
                    logger.info(f"Retrying {signal.symbol} with FOK filling...")
                    request["type_filling"] = mt5.ORDER_FILLING_FOK
                    result = mt5.order_send(request)
                    
                    if result is None:
                        last_error = mt5.last_error()
                        logger.error(f"Retry failed for {signal.symbol}: No result. Last error: {last_error}")
                        return False
                    
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        logger.error(f"Retry failed for {signal.symbol}: {result.comment} (Code: {result.retcode})")
                        return False
                else:
                    return False
            
            current_balance = self.get_account_balance()
            logger.info(f"TRADE PLACED: {signal.symbol} {signal.signal_type.value} | "
                        f"Volume: {volume} | Entry: {price:.5f} | SL: {validated_sl:.5f} | "
                        f"TP: {validated_tp:.5f} | Balance: ${current_balance:.2f}")
            
            # Open Trade Visualiser Chart and plot SL, TP and Entry
            #visualizer = TradeVisualizer()
            #visualizer.start(check_interval=60, display_duration=10)  # Runs in main thread
            
            self.active_trades[result.order] = {
                'symbol': signal.symbol,
                'type': signal.signal_type,
                'entry': price,
                'volume': volume,
                'original_sl': validated_sl,
                'original_tp': validated_tp
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error placing trade for {signal.symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def update_trailing_stops(self):
        """Update trailing stops with ATR-based dynamic distances and break-even logic"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return
            
            for position in positions:
                if position.magic != MAGIC_NUMBER:
                    continue
                
                symbol = position.symbol
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    continue
                
                current_price = symbol_info.bid if position.type == mt5.ORDER_TYPE_BUY else symbol_info.ask
                point = symbol_info.point
                digits = symbol_info.digits
                
                # Calculate ATR for BE and trailing
                df = self.get_ohlc_data_for_trailing(symbol)
                if df.empty:
                    logger.warning(f"No OHLC data for {symbol}. Skipping stop updates.")
                    continue
                
                atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'], 14)
                current_atr = atr.iloc[-1]
                if pd.isna(current_atr) or current_atr <= 0:
                    current_atr = (df['high'].tail(20).max() - df['low'].tail(20).min()) / 20
                
                # Calculate profit in points
                if position.type == mt5.ORDER_TYPE_BUY:
                    profit_points = (current_price - position.price_open) / point
                else:
                    profit_points = (position.price_open - current_price) / point
                
                new_sl = None
                update_reason = ""
                trade_state = "INITIAL"
                
                # Break-even check
                be_trigger = current_atr * self.config.break_even_atr_multiplier / point
                if profit_points >= be_trigger:
                    trade_state = "BREAK_EVEN"
                    be_plus_distance = current_atr * self.config.break_even_plus_atr_multiplier
                    if position.type == mt5.ORDER_TYPE_BUY:
                        break_even_price = position.price_open + be_plus_distance
                        if position.sl < break_even_price:
                            new_sl = round(break_even_price, digits)
                            update_reason = f"Break-even +{be_plus_distance/point:.1f} points (ATR-based)"
                    else:
                        break_even_price = position.price_open - be_plus_distance
                        if position.sl > break_even_price or position.sl == 0:
                            new_sl = round(break_even_price, digits)
                            update_reason = f"Break-even +{be_plus_distance/point:.1f} points (ATR-based)"
                
                # Trailing stop check
                trailing_trigger = current_atr * self.config.trailing_start_atr_multiplier / point
                if profit_points >= trailing_trigger:
                    trade_state = "TRAILING"
                    atr_trailing_points = self.calculate_atr_trailing_distance(symbol, position.type)
                    
                    if position.type == mt5.ORDER_TYPE_BUY:
                        trailing_sl = current_price - (atr_trailing_points * point)
                        if new_sl is None or trailing_sl > new_sl:
                            if trailing_sl > position.sl:
                                new_sl = round(trailing_sl, digits)
                                update_reason = f"ATR trailing stop ({atr_trailing_points:.1f} points)"
                    
                    elif position.type == mt5.ORDER_TYPE_SELL:
                        trailing_sl = current_price + (atr_trailing_points * point)
                        if new_sl is None or trailing_sl < new_sl:
                            if trailing_sl < position.sl or position.sl == 0:
                                new_sl = round(trailing_sl, digits)
                                update_reason = f"ATR trailing stop ({atr_trailing_points:.1f} points)"
                
                if new_sl is not None and new_sl != position.sl:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "position": position.ticket,
                        "sl": new_sl,
                        "tp": position.tp,
                    }
                    
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"🔄 {update_reason} for {symbol} ({trade_state}): {position.sl:.5f} → {new_sl:.5f} "
                                   f"(Profit: {profit_points:.1f} points)")
                    else:
                        error_msg = result.comment if result else "Unknown error"
                        logger.error(f"Failed to update SL for {symbol}: {error_msg}")
                        
        except Exception as e:
            logger.error(f"Error updating trailing stops: {e}")
    
    def log_account_status(self):
        """Log current account status"""
        try:
            account_info = mt5.account_info()
            if account_info:
                positions = mt5.positions_get()
                position_count = len(positions) if positions else 0
                
                logger.info(f"ACCOUNT STATUS: Balance=${account_info.balance:.2f} | "
                           f"Equity=${account_info.equity:.2f} | "
                           f"Free Margin=${account_info.margin_free:.2f} | "
                           f"Open Positions={position_count}")
        except Exception as e:
            logger.error(f"Error logging account status: {e}")
    
    def recover_existing_trades(self):
        """Recover existing trades after bot restart"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                logger.info("No existing positions found")
                return
            
            recovered_count = 0
            
            for position in positions:
                if position.magic == MAGIC_NUMBER:
                    signal_type = SignalType.BUY if position.type == mt5.ORDER_TYPE_BUY else SignalType.SELL
                    
                    self.active_trades[position.ticket] = {
                        'symbol': position.symbol,
                        'type': signal_type,
                        'entry': position.price_open,
                        'volume': position.volume,
                        'original_sl': position.sl,
                        'original_tp': position.tp
                    }
                    
                    recovered_count += 1
                    
                    logger.info(f"RECOVERED: {position.symbol} {signal_type.value} | "
                            f"Ticket: {position.ticket} | Entry: {position.price_open:.5f} | "
                            f"Current SL: {position.sl:.5f} | Current TP: {position.tp:.5f} | Volume: {position.volume}")
            
            if recovered_count > 0:
                logger.info(f"Recovered {recovered_count} existing trades for management")
            else:
                logger.info("No bot trades found to recover")
                
        except Exception as e:
            logger.error(f"Error recovering existing trades: {e}")

    def get_open_positions_count(self) -> int:
        """Get count of currently open positions"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return 0
            
            bot_positions = [p for p in positions if p.magic == MAGIC_NUMBER]
            return len(bot_positions)
        
        except Exception as e:
            logger.error(f"Error getting position count: {e}")
            return 0

    def rank_signals(self, signals: List[TradingSignal], strength_analyzer: CurrencyStrengthAnalyzer) -> List[Tuple[TradingSignal, float]]:
        """Rank trading signals based on confidence, currency strength differential, and risk-reward ratio"""
        ranked_signals = []
        
        for signal in signals:
            if signal.signal_type == SignalType.HOLD:
                continue
                
            # Calculate risk-reward ratio
            sl_distance = abs(signal.entry_price - signal.stop_loss)
            tp_distance = abs(signal.take_profit - signal.entry_price)
            if sl_distance > 0:
                risk_reward_ratio = tp_distance / sl_distance
            else:
                risk_reward_ratio = 0.0
            
            # Get currency strength differential
            base_currency = signal.symbol[:3]
            quote_currency = signal.symbol[3:6]
            base_strength = strength_analyzer.currency_strength.get(base_currency, 50.0)
            quote_strength = strength_analyzer.currency_strength.get(quote_currency, 50.0)
            strength_diff = abs(base_strength - quote_strength)
            
            # Get total score from calculate_signal_score
            total_score = self.calculate_signal_score(signal, strength_analyzer)
            
            ranked_signals.append((signal, total_score))
        
        # Sort by score in descending order
        ranked_signals.sort(key=lambda x: x[1], reverse=True)
        
        # Log ranking details
        logger.info("Signal Rankings:")
        for signal, score in ranked_signals:
            base_currency = signal.symbol[:3]
            quote_currency = signal.symbol[3:6]
            base_strength = strength_analyzer.currency_strength.get(base_currency, 50.0)
            quote_strength = strength_analyzer.currency_strength.get(quote_currency, 50.0)
            strength_diff = abs(base_strength - quote_strength)
            # Calculate RR for logging
            sl_distance = abs(signal.entry_price - signal.stop_loss)
            tp_distance = abs(signal.take_profit - signal.entry_price)
            if sl_distance > 0:
                risk_reward_ratio = tp_distance / sl_distance
            else:
                risk_reward_ratio = 0.0
            logger.info(f"  {signal.symbol} ({signal.signal_type.value}): Score={score:.3f}, "
                        f"Confidence={signal.confidence:.1f}%, "
                        f"Strength Diff={strength_diff:.1f}, "
                        f"RR={risk_reward_ratio:.2f}:1")
        
        # Return the list of (signal, score) tuples
        return ranked_signals
    
    def calculate_signal_score(self, signal: TradingSignal, strength_analyzer: CurrencyStrengthAnalyzer) -> float:
        """Calculate a score for a trading signal based on confidence, strength differential, and risk-reward ratio"""
        try:
            # Normalize confidence (0-100) to 0-1
            confidence_score = signal.confidence / 100.0
            
            # Calculate strength differential
            base_currency = signal.symbol[:3]
            quote_currency = signal.symbol[3:6]
            base_strength = strength_analyzer.currency_strength.get(base_currency, 50.0)
            quote_strength = strength_analyzer.currency_strength.get(quote_currency, 50.0)
            strength_diff = abs(base_strength - quote_strength)
            # Normalize strength differential (assuming max diff is 100)
            strength_score = min(strength_diff / 100.0, 1.0)
            
            # Calculate risk-reward ratio
            sl_distance = abs(signal.entry_price - signal.stop_loss)
            tp_distance = abs(signal.take_profit - signal.entry_price)
            if sl_distance > 0:
                rr_ratio = tp_distance / sl_distance
            else:
                rr_ratio = 0.0
            # Cap RR ratio to avoid extreme values
            rr_score = min(rr_ratio / 3.0, 1.0)
            
            # Weighted score: 50% confidence, 30% strength diff, 20% RR
            total_score = (0.5 * confidence_score) + (0.3 * strength_score) + (0.2 * rr_score)
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error calculating signal score for {signal.symbol}: {e}")
            return 0.0

    def run_analysis(self):
        """Main analysis and trading loop with news filtering and signal ranking"""
        analyzer = ForexAnalyzer()

        current_balance = self.get_account_balance()
        if current_balance < self.config.min_account_balance:
            logger.warning(f"Account balance ${current_balance:.2f} below minimum ${self.config.min_account_balance:.2f}")
            return
        
        closed_positions = self.news_filter.close_risky_positions(self)
        if closed_positions > 0:
            logger.info(f"Closed {closed_positions} positions due to upcoming news events")
        
        open_positions = self.get_open_positions_count()
        
        if open_positions >= 1:
            logger.info(f"Maximum positions reached ({open_positions}/1). Skipping new trade analysis.")
            self.recover_existing_trades()
            return
        
        self.log_account_status()
        logger.info("Starting full analysis of all forex pairs...")

        strength_analyzer = CurrencyStrengthAnalyzer(ForexPairs())
        if strength_analyzer.calculate_strength():
            best_bullish = strength_analyzer.get_best_pair_to_trade(bullish=True)
            best_bearish = strength_analyzer.get_best_pair_to_trade(bullish=False)

            if best_bullish:
                logger.info(f"Best Bullish Pair: {best_bullish[0]} (+{best_bullish[1]:.1f})")
            if best_bearish:
                logger.info(f"Best Bearish Pair: {best_bearish[0]} ({best_bearish[1]:.1f})")
        
        signals = []
        
        # Analyze all pairs before making any trading decisions
        for symbol in analyzer.get_all_symbols():
            try:
                if not mt5.symbol_select(symbol, True):
                    logger.warning(f"Failed to select symbol {symbol}")
                    continue
                
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None or not symbol_info.visible:
                    logger.info(f"Skipping {symbol}: Market closed or unavailable")
                    continue
                
                should_avoid, news_reason = self.news_filter.should_avoid_pair(symbol)
                if should_avoid:
                    logger.info(f"Skipping {symbol}: {news_reason}")
                    continue
                
                self.update_settings_for_symbol(symbol)
                
                base_currency = symbol[:3]
                quote_currency = symbol[3:6]
                base_strength = strength_analyzer.currency_strength.get(base_currency, 50.0)
                quote_strength = strength_analyzer.currency_strength.get(quote_currency, 50.0)
                strength_diff = abs(base_strength - quote_strength)

                if strength_diff < 10:
                    logger.info(f"Skipping {symbol}: Weak currency strength diff ({base_currency}: {base_strength:.1f}, {quote_currency}: {quote_strength:.1f})")
                    continue

                signal = analyzer.analyze_symbol(symbol, strength_analyzer)
                
                if signal.signal_type != SignalType.HOLD and signal.confidence >= 60:
                    signals.append(signal)
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        logger.info(f"Completed analysis of all pairs. Found {len(signals)} valid trading signals.")
        
        if not signals:
            logger.info("No valid trading signals found.")
            return
        
        # Rank signals to find the most probable trade
        logger.info("Ranking signals to select the most probable trade...")
        ranked_signals = self.rank_signals(signals, strength_analyzer)
        
        if not ranked_signals:
            logger.info("No signals passed ranking criteria.")
            return
        
        # Log all ranked signals
        logger.info("All Ranked Signals:")
        for signal, score in ranked_signals:
            base_currency = signal.symbol[:3]
            quote_currency = signal.symbol[3:6]
            base_strength = strength_analyzer.currency_strength.get(base_currency, 50.0)
            quote_strength = strength_analyzer.currency_strength.get(quote_currency, 50.0)
            strength_diff = abs(base_strength - quote_strength)
            
            # Calculate RR for logging
            sl_distance = abs(signal.entry_price - signal.stop_loss)
            tp_distance = abs(signal.take_profit - signal.entry_price)
            if sl_distance > 0:
                rr_ratio = tp_distance / sl_distance
            else:
                rr_ratio = 0.0
            logger.info(f"  {signal.symbol} ({signal.signal_type.value}): Score={score:.3f}, "
                        f"Confidence={signal.confidence:.1f}%, "
                        f"Strength Diff={strength_diff:.1f}, "
                        f"RR={rr_ratio:.2f}:1 - {signal.reason}")

        # Select and log the top signal
        top_signal, top_score = ranked_signals[0]
        logger.info(f"Top Signal Selected: {top_signal.symbol} {top_signal.signal_type.value} "
            f"(Score: {top_score:.3f}, Confidence: {top_signal.confidence}%) - {top_signal.reason}")
        

        # Final checks before placing trade
        should_avoid, news_reason = self.news_filter.should_avoid_pair(top_signal.symbol)
        if should_avoid:
            logger.info(f"Trade cancelled for {top_signal.symbol}: {news_reason}")
            return
        
        current_open = self.get_open_positions_count()
        if current_open >= 1:
            logger.info(f"Position limit reached after ranking. Cannot place trade for {top_signal.symbol}.")
            return
        
        positions_in_symbol = mt5.positions_get(symbol=top_signal.symbol)
        symbol_positions = [p for p in (positions_in_symbol or []) if p.magic == MAGIC_NUMBER]
        
        if len(symbol_positions) == 0:
            if self.place_trade(top_signal):
                logger.info(f"Trade placed successfully for {top_signal.symbol}. Position limit now reached (1/1).")
            else:
                logger.error(f"Failed to place trade for {top_signal.symbol}.")
        else:
            logger.info(f"Already have position in {top_signal.symbol}, skipping trade placement.")
    
    def run(self, login: int = None, password: str = None, server: str = None):
        """Main bot execution loop"""
        if not self.initialize_mt5(login, password, server):
            logger.error("Failed to initialize MT5")
            return
        
        logger.info("🔍 Checking for existing trades...")
        self.recover_existing_trades()
        
        logger.info(f"Trading bot started in {self.config.account_type.value.upper()} mode")
        logger.info(f"Risk Management: Fixed Risk=${self.config.fixed_risk_amount}, "
                   f"Max Risk={self.config.max_risk_percentage*100}%, "
                   f"Break-Even={self.config.break_even_atr_multiplier}x ATR, "
                   f"Trailing Start={self.config.trailing_start_atr_multiplier}x ATR")
        
        try:
            cycle_count = 0
            trailing_check_count = 0
            
            while True:
                cycle_count += 1
                trailing_check_count += 1
                
                if cycle_count == 1 or trailing_check_count >= 20:
                    logger.info(f"{'='*60}")
                    logger.info(f"ANALYSIS CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info(f"{'='*60}")
                    
                    self.run_analysis()
                    trailing_check_count = 0
                
                self.update_trailing_stops()

                # Open Trade Visualiser Chart and plot SL, TP and Entry
                #visualizer = TradeVisualizer()
                #visualizer.start(check_interval=60, display_duration=30)  # Runs in main thread
                
                time.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
        finally:
            mt5.shutdown()
            logger.info("MT5 connection closed")

def main():
    """Main function to run the trading bot with optimized settings"""
    
    # Start with base configuration - will be optimized automatically
    config = TradeConfig(
        risk_reward_ratio=2.0,
        fixed_risk_amount=6.0,
        min_account_balance=100.0,
        max_risk_percentage=0.06,
        trailing_stop_points=35,
        account_type=AccountType.DEMO,
        dynamic_sizing=True,
        break_even_atr_multiplier=1.2,
        break_even_plus_atr_multiplier=0.03,
        trailing_start_atr_multiplier=1.8
    )
    
    # Create and run the bot
    bot = MT5TradingBot(config)
    
    logger.info("Advanced Trading Bot with News Filter & Optimized Settings")
    logger.info("News events will be monitored and trades adjusted accordingly")
    logger.info("Settings will be optimized based on account balance and currency pairs")
    
    # Run the bot
    bot.run(
        login=your_account_login,
        password="your_account_password",
        server="your_broker_server"
    )

if __name__ == "__main__":
    main()