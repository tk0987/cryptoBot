'''not ready yet'''
# data format: df[["timestamp", "open", "high", "low", "close", "volume"]]

import numpy as np
import tensorflow as tf
# from tensorflow.keras import tf.keras.layers
import random as r
from datetime import datetime
import os
from tqdm import tqdm
import pandas as pd
# Set random seed
r.seed(datetime.now().timestamp())

# Define global variables
window_size = 50
batch_size = 64


# +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++
#              Data Preparation
# +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++
print('Loading and preprocessing data...')
data=[]
# +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++
#                                      Declare The Thing (1982)
# +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++
with tf.device('/device:GPU:0'):
    def loss_trader(pred,pred_prev):
        # loss = pred-pred_prev

        return -tf.reduce_mean(pred - pred_prev)

    def inference(data,model):
        preds=model(data,training=False)
        
        idx=tf.argmax(preds)
        
        return data[idx]
    
    # def minmax_normalization(data):
    #     return (data - np.min(data)) / (np.max(data) -np.min(data))
    def tcn_layer(x, filters, dilation_rate):
        x1 = tf.keras.layers.Conv1D(filters, kernel_size=9, padding="causal",
                                    activation="relu", dilation_rate=dilation_rate)(x)
        x1=tf.keras.layers.Dense(filters,"elu")(x1)
        x2 = tf.keras.layers.Conv1D(filters, kernel_size=3, padding="causal",
                                    activation="relu", dilation_rate=dilation_rate)(x)
        return tf.keras.layers.Add()([x1, x2])
    def transformer_encoder(inputs, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        # Multi-head self-attention
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
        # attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output) # removed for inference compatibility
        out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

        # Feed-forward network
        ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(out1)
        ffn_output = tf.keras.layers.Dense(embed_dim)(ffn_output)
        # ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output) # removed for inference compatibility
        out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

        return out2
    def build_model(n, shape):
        inputs = tf.keras.Input(shape)
        inputs=transformer_encoder(inputs,shape[1],8,7,0.12)
        
        x1 = tcn_layer(inputs, 4*n, 4)
        
        x2 = tcn_layer(inputs, 4*n, 4)
        
        neck1 = tcn_layer(x1, n//5, 4)
        neck2 = tcn_layer(x2, n//5, 4)
        
        mix = tf.keras.layers.Add()([neck1, neck2])
        mix = transformer_encoder(mix,shape[1],8,7,0.12)
        mix = tf.keras.layers.Flatten()(mix)  # Ensure correct output shape

        out = tf.keras.layers.Dense(shape[1],'softmax')(mix)  # No activation for regression output

        return tf.keras.Model(inputs, out)

    # Build model
    net = build_model(64,(np.shape(data)[0],np.shape(data)[1],np.shape(data)[2]))
    net.summary()

    # +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++
    #                                       Compile the Model
    # +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++
    opti = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # loss_fn = tf.keras.losses.MeanSquaredError() # custom loss
    # net.compile(optimizer=opti, loss=loss_fn, metrics=["mae"])  # Using MAE as additional metric



    # print("Data shape:", np.shape(data))

    # +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++
    #                                           Training Loop
    # +++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++
    @tf.function
    def train_step(inp,target):
        
        with tf.GradientTape() as tape:
            preds = net(inp)
            loss = loss_trader(preds,target)  # Proper use of TF loss function
        grads = tape.gradient(loss, net.trainable_variables)
        # grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]  # Gradient clipping for stability
        opti.apply_gradients(zip(grads, net.trainable_variables))
        return loss

    epoch = 0
    best_loss = float('inf')
    prev_pred=0

    while True:
        epoch_loss = 0.0

        # Batch processing for efficiency
        for i in tqdm(range(0, len(data) - window_size, batch_size)):
            try:
                batch_inp = np.stack([data[j:j+window_size, :] for j in range(i, min(i+batch_size, len(data)-window_size))], axis=0)
                batch_target = np.stack([data[j+window_size:5+j+window_size, :] for j in range(i, min(i+batch_size, len(data)-window_size-5))], axis=0)  # Fix target shape
                # if np.shape(batch_inp)!=np.shape(batch_target):
                pred=inference(data,net)
                batch_loss = train_step(pred, prev_pred)
                epoch_loss += batch_loss.numpy()
                
                # else:
                #     print('SKIP')
                pred_prev=pred
            except ValueError as e:
                break 

        print(f"Epoch {epoch + 1}, Loss = {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            net.save("optimized_model.keras")
            print(f"Model saved with loss {best_loss:.6f}")

        epoch += 1
