# data format: df[["timestamp", "open", "high", "low", "close", "volume"]]

import numpy as np
import tensorflow as tf
# from tensorflow.keras import tf.keras.layers
import random as r
from datetime import datetime
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
# Set random seed
r.seed(datetime.now().timestamp())

# Define global variables
window_size = 10
# batch_size = 64


# +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++
#              Data Preparation
# +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++

data_dir = "all_data/binance"
required_length = 3 * 60  # 5 days × 24 hours × 60 minutes = 30240

# Step 1: Read symbols from symbols_used.txt
with open("symbols_used.txt", "r") as f:
    ordered_symbols = [line.strip().lower() for line in f if line.strip()]

# Step 2: Map symbols to filenames
all_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
symbol_to_file = {}
for file in all_files:
    symbol = file.split("_")[0].lower()
    if symbol not in symbol_to_file:
        symbol_to_file[symbol] = file  # If multiple files per symbol, you may need to refine this

# Step 3: Load files in the order of symbols_used.txt
data_arrays = []
# symbols_order = []

for symbol in ordered_symbols:
    if symbol not in symbol_to_file:
        print(f"Warning: No file found for symbol '{symbol}'")
        continue

    file_path = os.path.join(data_dir, symbol_to_file[symbol])
    df = pd.read_csv(file_path, sep="\t", header=None, names=["timestamp", "open", "high", "low", "close", "volume"])

    if len(df) < required_length:
        print(f"Skipping {symbol.upper()} due to insufficient data")
        continue

    start_time = pd.to_datetime(df["timestamp"].iloc[0])
    df["time_sec"] = (pd.to_datetime(df["timestamp"]) - start_time).dt.total_seconds()

    df = df[["open", "high", "low", "close", "volume"]]
    data_arrays.append(df.to_numpy())
    # symbols_order.append(symbol.upper())

# Step 4: Stack into final array
data = np.stack(data_arrays,dtype=np.float16)
del df
del data_arrays
del ordered_symbols
del symbol_to_file
del all_files
del file_path
del start_time
import gc
gc.collect()
print("Imported & processed, memory free")
# print("Symbols order:", symbols_order)

# +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++
#                                      Declare The Thing (1982)
# +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++
global prev_pred
prev_pred=tf.zeros([np.shape(data)[0]], dtype=tf.float16)
with tf.device('/device:CPU:0'):
    def loss_trader(model_output, input_data):
        """
        model_output: shape (1, num_symbols) — softmax weights
        input_data: shape (1, num_symbols, window_size, 5) — full input batch
        """
        global prev_pred
        weights = tf.cast(tf.squeeze(model_output), tf.float16)  # shape: (num_symbols,)
        weights_prev=tf.cast(tf.squeeze(prev_pred), tf.float16)
        # close_prices =   # shape: (num_symbols,) — last timestep "close" values

        weighted_pred = tf.reduce_sum(weights * input_data[0,:, -1, 3])
        weighted_prev=tf.reduce_sum(weights_prev * input_data[0,:, -2, 3])

        # You can define your target however you like — for example:
        # target = tf.reduce_mean(close_prices)  # or use actual future value if available

        return -tf.subtract(weighted_pred, weighted_prev)

    # def inference(data,model):
    #     data= tf.expand_dims(data,axis=0)
    #     preds=model(data,training=False)
        
    #     idx=tf.argmax(preds[0])
    #     # tf.print("Predicted index:", idx)
        
    #     return tf.constant(data[0,idx,-1,3])
    
    # def minmax_normalization(data):
    #     return (data - np.min(data)) / (np.max(data) -np.min(data))
    def tcn_layer(x, filters, dilation_rate):
        x1 = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), padding="same",
                                    activation="relu", dilation_rate=dilation_rate)(x)
        x1=tf.keras.layers.Dense(filters,"elu")(x1)
        x2 = tf.keras.layers.Conv2D(filters, kernel_size=(1,1), padding="same",
                                    activation="relu", dilation_rate=dilation_rate)(x)
        return tf.keras.layers.Add()([x1, x2])
    def transformer_encoder(inputs, embed_dim, num_heads, ff_dim):
        # Multi-head self-attention
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
        attn_output = tf.keras.layers.Dense(5,"elu")(attn_output) # removed for inference compatibility
        out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

        # Feed-forward network
        ffn_output = tf.keras.layers.Conv2D(ff_dim, activation="relu", kernel_size=(1,1), padding="same")(out1)
        ffn_output = tf.keras.layers.Conv2D(5, activation="relu", kernel_size=(1,1), padding="same")(ffn_output)
        # ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output) # removed for inference compatibility
        out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

        return out2
    def build_model(n, shape):
        inputs = tf.keras.Input(shape)
        inputs=transformer_encoder(inputs,shape[0],8,7)
        
        x1 = tcn_layer(inputs, 4*n, (5,5))
        
        x2 = tcn_layer(inputs, 4*n, (5,5))
        
        neck1 = tcn_layer(x1, n, (2,2))
        neck2 = tcn_layer(x2, n, (2,2))
        
        mix = tf.keras.layers.Add()([neck1, neck2])
        mix = tf.keras.layers.Dense(5,'elu')(mix)
        mix = transformer_encoder(mix,shape[0],8,7)
        mix = tf.keras.layers.Flatten()(mix)  # Ensure correct output shape
        mix = tf.keras.layers.Dense(shape[0]//12,activation='elu')(mix)
        mix = tf.keras.layers.Dense(shape[0]//6,activation='elu')(mix)

        out = tf.keras.layers.Dense(shape[0],activation='softmax')(mix)  # No activation for regression output
        # print(out)

        return tf.keras.Model(inputs, out)

    # Build model
    net = build_model(2,(np.shape(data)[0],window_size,np.shape(data)[2]))
    net.summary()

    # +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++
    #                                       Compile the Model
    # +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++
    opti = tf.keras.optimizers.Adam(learning_rate=0.001)
    # loss_fn = tf.keras.losses.MeanSquaredError() # custom loss
    # net.compile(optimizer=opti, loss=loss_fn, metrics=["mae"])  # Using MAE as additional metric



    # print("Data shape:", np.shape(data))

    # +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++
    #                                           Training Loop
    # +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++
    # @tf.function
    
    def train_step(inp):
        global prev_pred
        inp=tf.expand_dims(inp,axis=0)
        with tf.GradientTape() as tape:
            preds = net(inp)
            loss = loss_trader(preds,inp)  # Proper use of TF loss function
        grads = tape.gradient(loss, net.trainable_variables)
        opti.apply_gradients(zip(grads, net.trainable_variables))
        return [loss,preds]

    epoch = 0
    best_loss = float('inf')
    
    # pred=tf.constant([0.0]*5)

    while True:
        epoch_loss = 0.0
        for i in tqdm(range(len(data[0]) - window_size)):
            batch_inp = data[:, i:i+window_size, :]
            # target = data[:, i+window_size, 3]  # "close" value
            # pred = inference(batch_inp, net)
            # print(pred)
            batch_loss = train_step(batch_inp)
            prev_pred=batch_loss[1]
            epoch_loss += batch_loss[0].numpy()
            # pred_prev=target


        print(f"Epoch {epoch + 1}, Loss = {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            net.save("optimized_model.keras")
            # print(f"Model saved with loss {best_loss:.6f}")

        epoch += 1
