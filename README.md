# Forex Trading Bot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MetaTrader5](https://img.shields.io/badge/MetaTrader5-Compatible-green.svg)](https://www.metatrader5.com/)
[![Status](https://img.shields.io/badge/Status-Demo%20Only-red.svg)](https://github.com/loopbacklogic/forex-trading-bot)

**An advanced automated Forex trading bot built with Python and MetaTrader5 (MT5) integration for educational and testing purposes.**

## ⚠️ Important Notice

**This bot is for educational and testing purposes only. The current trading strategy does not generate profitable trades and should ONLY be used on demo accounts.**

## Why I Created This Bot

Picture this: a guy with very little Python skills, a wild fascination with Forex trading, and a bold (slightly delusional) plan to become a coding wizard in just 12 weeks. That’s me! I decided to dive headfirst into Python, not because I love snakes, but because I wanted to build something cool for my obsession with currency markets. So, I threw down the gauntlet and challenged AI, yes, challenged it, to teach me how to code faster than you can say "pip install chaos." Why Forex? Because who doesn’t dream of sipping coffee while a bot makes them rich (or at least tries not to lose their shirt)? This bot is my love letter to learning Python, a crash course in coding fueled by my passion for pips, trends, and ATRs. It’s not a money-making machine (yet and is a demo account only, folks!). It’s proof that a newbie with a dream and a stubborn streak can wrestle Python into submission, one debug at a time. Joking apart, I have been learning to trade for several years, and there are a lot of manual tasks. The purpose of this is to automate those tasks, and the placing of those trades is the final step.

## 🚀 Features

This bot implements comprehensive trading functionality:

- **🔄 Multi-Timeframe Analysis**: Analyzes trends across Weekly, Daily, H4, H2, H1, M30, and M15 timeframes for robust signal generation
- **📊 Technical Indicators**: Includes custom implementations of SMA, EMA, RSI, MACD, Bollinger Bands, and ATR for trading signals
- **💪 Currency Strength Analysis**: Ranks currency pairs based on relative strength to identify high-probability trades
- **📰 News Filtering**: Avoids trading during high-impact news events to reduce risk
- **⚖️ Dynamic Risk Management**: Adjusts position sizes and stop-loss levels based on account balance and market volatility
- **🎯 Trailing Stops and Break-Even**: Implements ATR-based trailing stops and break-even logic for trade management
- **📈 Visualisation**: Includes real-time trade visualisation using Matplotlib and Seaborn
- **⚙️ Settings Optimisation**: Automatically adjusts trading parameters based on account balance and currency pair characteristics

## 📋 Prerequisites

Before running the bot, ensure you have:

- **Python 3.8+** installed on your system
- **MetaTrader5 Terminal** installed and configured with a demo account
- **Stable internet connection** for MT5 connectivity and data retrieval
- **Required Python Libraries**:
  ```bash
  pip install MetaTrader5 pandas numpy matplotlib seaborn
  ```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/loopbacklogic/forex-trading-bot.git
cd forex-trading-bot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. MetaTrader5 Setup
Ensure MetaTrader5 is installed and you have valid demo account credentials (login, password, server).

## 📖 Usage Guide

### Step 1: Demo Account Setup

1. **Register for a demo account** with a MetaTrader5-compatible broker:
   - ICMarkets
   - FXTM
   - OANDA
   - Other MT5-compatible brokers

2. **Obtain your credentials**:
   - Login number
   - Password
   - Server name (e.g., "ICMarketsSC-Demo")

3. **Download and install** the MetaTrader5 terminal from your broker's website

### Step 2: Bot Configuration

Open `forex_bot_v1.00.py` and update the `main()` function with your MT5 demo account credentials:

```python
bot.run(
    login=your_account_login,
    password="your_account_password",
    server="your_broker_server"
)
```

### Step 3: Run the Bot

```bash
python forex_bot_v1.00.py
```

**The bot will automatically**:
- Connect to your MT5 demo account
- Analyse forex pairs for trading signals
- Place trades based on the confluence strategy
- Manage open positions with trailing stops and break-even logic
- Log all activities to the console

### Step 4: Enable AutoTrading

**In MetaTrader5 terminal**:
1. Go to **Tools → Options → Expert Advisors**
2. Check **"Allow automated trading"**
3. Click the **AutoTrading** button in the MT5 toolbar (should turn green)

### Step 5: Monitoring

- The bot provides detailed logging of analysis, trades, and account status
- Trade visualisations are displayed periodically (requires display or VNC for GUI)
- Monitor console output for real-time updates

## 🎯 Strategy Status

⚠️ **IMPORTANT**: The current trading strategy implemented in this bot **does not generate profitable trades** and was developed solely to test and ensure that all functionality works as intended:

- ✅ MT5 integration
- ✅ Technical analysis
- ✅ Position sizing
- ✅ Trade management
- ✅ Risk management systems

Well, it sometimes generates profitable trades, but that's another thing altogether. 

**This bot should only be run on a demo account for testing purposes.**

## ⚙️ Configuration

The bot uses a `TradeConfig` dataclass to manage trading parameters:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `risk_reward_ratio` | 2.0 | Risk-to-reward ratio (2:1) |
| `fixed_risk_amount` | $6.0 | Base risk per trade |
| `max_risk_percentage` | 6% | Maximum risk per trade as percentage of balance |
| `trailing_stop_points` | Variable | Minimum trailing stop distance in points |
| `break_even_atr_multiplier` | Variable | ATR multiplier for moving to break-even |
| `account_type` | `AccountType.DEMO` | Account type restriction |

**Note**: These settings are automatically optimised based on account balance and currency pair volatility.

## 🛡️ Risk Management

The bot implements multiple risk management layers:

- **Position Limit**: Maximum of one open position at a time
- **Dynamic Position Sizing**: Risk calculated as a percentage of the account balance
- **News Avoidance**: Trades avoided during high-impact news events
- **ATR-Based Stops**: Volatility-adjusted stop-loss and trailing stops
- **Break-Even Logic**: Automatic move to break-even when profitable

## 📁 Project Structure

```
forex-trading-bot/
├── forex_bot_v1.00.py          # Main bot script
├── requirements.txt             # Python dependencies
├── README.md                   # This documentation
├── LICENSE                     # MIT License
```

## 🚨 Important Notes

### Demo Account Only
- **Never use on live accounts** due to untested trading strategy
- Strategy designed for functionality testing only
- No guarantee of profitable performance

### Market Risks
- **Forex trading carries high risk** of financial loss
- **You may lose all invested capital**
- **No performance guarantees** provided

### Broker Compatibility
- Ensure your broker supports MT5
- Verify currency pairs are available in `ForexPairs`
- Test the connection before running the bot

### Technical Requirements
- Stable internet connection required
- MT5 terminal must remain open during operation
- Sufficient system resources for analysis

## 🤝 Contributing

We welcome contributions to improve the bot:

- **Strategy Enhancement**: Improve trading algorithms
- **New Indicators**: Add technical analysis tools
- **News Integration**: Enhance news filtering
- **Documentation**: Improve guides and examples
- **Testing**: Add unit tests and validation

**How to Contribute**:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**IMPORTANT LEGAL NOTICE**

This software is provided "as is" without warranty of any kind, express or implied. The author(s) and contributors are not responsible for any losses incurred from using this bot.

**Trading Risks**:
- Forex trading involves significant financial risk
- Past performance does not indicate future results
- You should only trade with funds you can afford to lose
- Always consult with a qualified financial advisor before trading

**Educational Purpose**:
- This bot is designed for educational and testing purposes only
- The trading strategy is not optimised for profitability
- Use only on demo accounts to avoid financial loss

**No Financial Advice**:
- This software does not constitute financial advice
- Users are responsible for their own trading decisions
- Independent research and due diligence are essential

## 🆘 Support

**Getting Help**:
- Check [Issues](https://github.com/loopbacklogic/forex-trading-bot/issues) for known problems

**Reporting Issues**:
- Use the [Issue Tracker](https://github.com/loopbacklogic/forex-trading-bot/issues) for bugs
- Provide detailed information about your setup
- Include error messages and logs when applicable

---

**🔴 CRITICAL REMINDER: This bot is for educational purposes only. Use exclusively on demo accounts. Trading involves substantial risk of loss.**
