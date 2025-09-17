import os
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta, timezone

end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=21)


# Binance API keys (optional for public data)
API_KEY = ""
API_SECRET = ""

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

# Parameters
exchange = "binance"
interval = Client.KLINE_INTERVAL_1MINUTE
# end_date = datetime.utcnow()
# start_date = end_date - timedelta(days=7)

# Base directory for saving data
base_dir = "all_data"
save_path = os.path.join(base_dir, exchange)
os.makedirs(save_path, exist_ok=True)

def get_spot_symbols():
    exchange_info = client.get_exchange_info()
    symbols = [
        s["symbol"] for s in exchange_info["symbols"]
        if s["isSpotTradingAllowed"]
        and s["status"] == "TRADING"
        and s["quoteAsset"] == "USDC"
        and not s["symbol"].endswith("DOWN")
        and not s["symbol"].endswith("UP")
        and not s["symbol"].endswith("BULL")
        and not s["symbol"].endswith("BEAR")
    ]
    return symbols


def download_klines(symbol, interval, start_time, end_time):
    """Download klines from Binance with timezone-aware datetime"""
    klines = []
    temp_start = start_time

    while temp_start < end_time:
        data = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=int(temp_start.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=1000
        )
        if not data:
            break
        klines.extend(data)

        # Convert last timestamp to timezone-aware datetime
        last_time = data[-1][0]
        last_datetime = datetime.fromtimestamp(last_time / 1000, tz=timezone.utc)

        # Advance start time by one minute
        temp_start = last_datetime + timedelta(minutes=1)

    return klines

def klines_to_df(klines):
    """Convert raw klines to clean DataFrame"""
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.astype({
        "open": float, "high": float, "low": float,
        "close": float, "volume": float
    })
    return df

def main():
    symbols = get_spot_symbols()
    print(f"Found {len(symbols)} spot trading pairs.")

    for symbol in symbols:
        print(f"Downloading {symbol}...")
        try:
            klines = download_klines(symbol, interval, start_date, end_date)
            if not klines:
                print(f"No data for {symbol}")
                continue

            df = klines_to_df(klines)
            file_name = f"{symbol.lower()}_{start_date.date()}_{end_date.date()}.txt"
            file_path = os.path.join(save_path, file_name)

            # Save as tab-separated values, no header, no index
            df.to_csv(file_path, sep="\t", index=False, header=False)
            print(f"Saved {symbol} to {file_path}")
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")

if __name__ == "__main__":
    main()
