import numpy as np
import tensorflow as tf
import pandas as pd
import os
import random as r
from datetime import datetime
from tqdm import tqdm
import gc
import sys
# -----------------------
# Set random seed
# -----------------------
r.seed(datetime.now().timestamp())
np.random.seed(int(datetime.now().timestamp()) % 2**32)
tf.random.set_seed(int(datetime.now().timestamp()) % 2**32)

# -----------------------
# Global constants
# -----------------------
window_size = 14
initial_cash = 100.0
data_dir = "all_data/binance"
required_length = 3 * 60  # 3 hours for testing (adjust as needed)

# -----------------------
# Data loading
# -----------------------
with open("symbols_used.txt", "r") as f:
    ordered_symbols = [line.strip().lower() for line in f if line.strip()]

all_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
symbol_to_file = {}
for file in all_files:
    symbol = file.split("_")[0].lower()
    if symbol not in symbol_to_file:
        symbol_to_file[symbol] = file

data_arrays = []
for symbol in ordered_symbols:
    if symbol not in symbol_to_file:
        print(f"Missing: {symbol}")
        continue

    file_path = os.path.join(data_dir, symbol_to_file[symbol])
    df = pd.read_csv(file_path, sep="\t", header=None,
                    names=["timestamp", "open", "high", "low", "close", "volume"])

    if len(df) < required_length:
        print(f"Skipping {symbol} (not enough data)")
        continue

    start_time = pd.to_datetime(df["timestamp"].iloc[0])
    df["time_sec"] = (pd.to_datetime(df["timestamp"]) - start_time).dt.total_seconds()
    df = df[["open", "high", "low", "close", "volume"]]
    data_arrays.append(df.to_numpy(dtype=np.float32))

data = np.stack(data_arrays, dtype=np.float32)
print("Data loaded:", data.shape)
print("NaNs found:", np.isnan(data).sum())

del df, data_arrays, ordered_symbols, symbol_to_file, all_files, file_path, start_time
gc.collect()

# -----------------------
# Model building blocks
# -----------------------
def lstm(inp, howmany, n):
    x_sum = None
    N, S = None, None

    for i in range(howmany):
        if i == 0:
            x, N, S = tf.keras.layers.LSTM(n, return_sequences=True, return_state=True)(inp[0])
        else:
            x, N, S = tf.keras.layers.LSTM(n, return_sequences=True, return_state=True)(inp[0], initial_state=[N, S])

        # Accumulate the outputs
        if x_sum is None:
            x_sum = x
        else:
            x_sum = tf.keras.layers.Add()([x_sum, x])

    return tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=0))(x_sum)


def tcn_layer(x, filters, dilation_rate):
    x1 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same", activation="elu",
                                dilation_rate=dilation_rate)(x)
    x1 = tf.keras.layers.Dense(filters, activation="elu")(x1)
    x2 = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding="same", activation="elu")(x)
    return tf.keras.layers.Add()([x1, x2])

def transformer_encoder(inputs, embed_dim, num_heads, ff_dim):
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn_output = tf.keras.layers.Dense(5, activation="elu")(attn_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = tf.keras.layers.Conv2D(ff_dim, kernel_size=(1, 1), activation="relu", padding="same")(out1)
    ffn_output = tf.keras.layers.Conv2D(5, kernel_size=(1, 1), activation="relu", padding="same")(ffn_output)
    ffn_output = tf.keras.layers.Dropout(0.1)(ffn_output)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return out2

def build_model(n, shape):
    inputs = tf.keras.Input(shape)
    # x = transformer_encoder(inputs, shape[0], 8, 7)

    x = lstm(inputs, 16,64 * n)
    # x2 = tcn_layer(inputs, 4 * n, (5, 5))

    # x= lstm(x, 8,n//2)
    # neck2 = tcn_layer(x2, n, (2, 2))

    
    x = tf.keras.layers.Dense(shape[0]*24, activation='elu')(x)
    # x = transformer_encoder(x, shape[0], 8, 7)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(shape[0] * 12, activation='elu')(x)
    x = tf.keras.layers.Dense(shape[0] * 6, activation='elu')(x)
    outputs = tf.keras.layers.Dense(2, activation='sigmoid')(x)

    return tf.keras.Model(inputs, outputs)

# -----------------------
# Loss Function (Trader Simulation)
# -----------------------
def loss_trader(model_output, input_data, prev_value, holdings, cash):
    close_prices = input_data[0, :, -1, 3]  # shape: (num_assets,)
    probs = tf.squeeze(model_output)       # shape: (num_assets, 2)

    loss_probs = model_output[0, 0]
    gain_probs = model_output[0, 1]


    # 📌 Portfolio value
    portfolio_value = cash + tf.reduce_sum(holdings * close_prices)

    # ✅ Trade only if gain_prob > 0.5 and loss_prob < 0.5
    trade_mask = tf.cast((gain_probs > 0.5) & (loss_probs < 0.5), tf.float32)
    conf_weights = gain_probs * trade_mask

    # ✅ Normalize confident weights (if any)
    weight_sum = tf.reduce_sum(conf_weights)
    safe_weights = tf.cond(
        weight_sum > 0,
        lambda: conf_weights / weight_sum,
        lambda: conf_weights  # all zeros, no trades
    )

    # 📌 Allocation logic
    target_alloc = safe_weights * portfolio_value
    current_alloc = holdings * close_prices
    delta_alloc = target_alloc - current_alloc
    delta_qty = delta_alloc / close_prices

    new_holdings = holdings + delta_qty
    trade_cost = tf.reduce_sum(delta_qty * close_prices)

    # 💰 Fee calculation
    fee_rate = 0.001  # 0.1% Binance trading fee
    trade_volume = tf.reduce_sum(tf.abs(delta_qty * close_prices))
    fees = fee_rate * trade_volume

    new_cash = cash - trade_cost - fees
    new_value = new_cash + tf.reduce_sum(new_holdings * close_prices)

    reward = 100 * (new_value - prev_value) - portfolio_value

    # 🚫 Penalty for gain_probs > 0.3 (overconfidence)
    penalty_overweight = tf.reduce_sum(tf.nn.relu(gain_probs - 0.3))
    penalty_strength = 10.0
    reward -= penalty_strength * penalty_overweight

    # 🚫 Penalty if no confident trades
    max_gain = tf.reduce_max(gain_probs * trade_mask)
    penalty_low_conf = tf.nn.relu(0.5 - max_gain)
    reward -= penalty_strength * penalty_low_conf

    # 🔄 Update state
    prev_value.assign(new_value)
    holdings.assign(new_holdings)

    return -reward, holdings, new_cash, new_value






# -----------------------
# Build model
# -----------------------
net = build_model(2, (1, window_size, data.shape[2]))
net.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# -----------------------
# Training loop
# -----------------------
epoch = 0
best_loss = float('inf')

while True:
    print(f"\nEpoch {epoch + 1}")
    epoch_loss = 0.0

    # Reset state at start of epoch
    prev_value = tf.Variable(initial_cash)
    cash = initial_cash
    holdings = tf.Variable(tf.zeros(data.shape[0], dtype=tf.float32))

    for currency in tqdm(range(data.shape[0])):
        for i in range(data.shape[1] - window_size):
            batch_input = data[currency, i:i + window_size, :]
            inp = tf.expand_dims(batch_input, axis=0)  # (1, 14, 5)
            inp = tf.expand_dims(inp, axis=0)  

            with tf.GradientTape() as tape:
                preds = net(inp, training=True)
                loss, holdings, cash, new_value = loss_trader(preds, inp, prev_value, holdings, cash)

            grads = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))

            epoch_loss += loss.numpy()
        if currency>0:    
            sys.stdout.write('\033[F')   # Move cursor up one line
            sys.stdout.write('\033[K')   # And clear it
        print(epoch_loss)

    print(f"Loss: {epoch_loss:.6f} | Final Portfolio Value: ${new_value:.4f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        net.save("optimized_model.keras")
        print(f"✅ Model saved (new best loss: {best_loss:.6f})")

    epoch += 1
