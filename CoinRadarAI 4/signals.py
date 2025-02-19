def generate_signals(indicators):
    cp = indicators.get('current_price')
    pivots = indicators.get('pivot_points')
    atr = indicators.get('atr')
    data = indicators.get('data')
    ema_long = cp > data['ema_20'].iloc[-1]
    ema_short = cp < data['ema_20'].iloc[-1]
    obv_long = data['obv'].iloc[-1] > data['obv'].iloc[-2] if len(data) > 1 else False
    obv_short = data['obv'].iloc[-1] < data['obv'].iloc[-2] if len(data) > 1 else False
    rsi = data['rsi'].iloc[-1]
    rsi_long = rsi < 70
    rsi_short = rsi > 30
    buy_strength = (data['macd'].iloc[-1] - data['macd_signal'].iloc[-1]) if (cp > pivots['pivot'] and ema_long and obv_long and rsi_long) else None
    sell_strength = (data['macd_signal'].iloc[-1] - data['macd'].iloc[-1]) if (cp < pivots['pivot'] and ema_short and obv_short and rsi_short) else None
    return {
        'buy_signal': buy_strength is not None,
        'sell_signal': sell_strength is not None,
        'buy_signal_strength': buy_strength,
        'sell_signal_strength': sell_strength,
        'tp_long': cp + (2 * atr),
        'sl_long': cp - (1 * atr),
        'tp_short': cp - (2 * atr),
        'sl_short': cp + (1 * atr)
    }
