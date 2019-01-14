import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# split data in 80%/10%/10% train/validation/test sets
valid_set_size_percentage = 10
test_set_size_percentage = 10

df = pd.read_csv('KOSPI.csv')
df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)
plt.plot(df['Open'], color='black', label='open')
plt.plot(df['Close'], color='green', label='close')
plt.plot(df['Low'], color='blue', label='low')
plt.plot(df['High'], color='red', label='high')
plt.title('stock price')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')

plt.subplot(1,2,2);
plt.plot(df['Volume'], color='black', label='volume')
plt.title('stock volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc='best');

plt.show()


# function for min-max normalization of stock
def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1, 1))
    df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1, 1))
    df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
    df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    df['Volume'] = min_max_scaler.fit_transform(df['Volume'].values.reshape(-1, 1))
    return df

def denormalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df[:, 0] = min_max_scaler.inverse_transform(df[:, 0].reshape(-1, 1))
    df[:, 1] = min_max_scaler.inverse_transform(df[:, 1].reshape(-1, 1))
    df[:, 2] = min_max_scaler.inverse_transform(df[:, 2].reshape(-1, 1))
    df[:, 3] = min_max_scaler.inverse_transform(df[:, 3].reshape(-1, 1))
    return df

# function to create train, validation, test data given stock data and sequence length
#
def load_data(stock, seq_len):
    data_raw = stock.as_matrix()  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index + seq_len])

    data = np.array(data);
    last_data = np.array([data[-1][1:, :]])



    valid_set_size = int(np.round(valid_set_size_percentage / 100 * data.shape[0]));
    test_set_size = int(np.round(test_set_size_percentage / 100 * data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_valid = data[train_set_size:train_set_size + valid_set_size, :-1, :]
    y_valid = data[train_set_size:train_set_size + valid_set_size, -1, :]

    x_test = data[train_set_size + valid_set_size:, :-1, :]
    y_test = data[train_set_size + valid_set_size:, -1, :]

    prediction_data = data[:, :-1, :]

    result = np.append(prediction_data, last_data, axis=0)

    return [x_train, y_train, x_valid, y_valid, x_test, y_test, result]


# choose one stock
df_stock = df.copy()
#df_stock.drop(['Volume'], 1, inplace=True)
df_stock.drop(['Date'], 1, inplace=True)

cols = list(df_stock.columns.values)
print('df_stock.columns.values = ', cols)

# normalize stock
df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm)

# create train, test data
seq_len = 20  # choose sequence length
x_train, y_train, x_valid, y_valid, x_test, y_test, prediction_data = load_data(df_stock_norm, seq_len)


print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ', x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)

plt.figure(figsize=(15, 5));
plt.plot(df_stock_norm.Open.values, color='red', label='open')
plt.plot(df_stock_norm.Close.values, color='green', label='low')
plt.plot(df_stock_norm.Low.values, color='blue', label='low')
plt.plot(df_stock_norm.High.values, color='black', label='high')
#plt.plot(df_stock_norm.volume.values, color='gray', label='volume')
plt.title('stock')
plt.xlabel('time [days]')
plt.ylabel('normalized price/volume')
plt.legend(loc='best')
plt.show()

## Basic Cell RNN in tensorflow

index_in_epoch = 0;
perm_array = np.arange(x_train.shape[0])
print(perm_array)
np.random.shuffle(perm_array)


# function to get the next batch
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array
    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array)  # shuffle permutation array
        start = 0  # start next epoch
        index_in_epoch = batch_size

    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]


# parameters
n_steps = seq_len - 1
n_inputs = 5
n_neurons = 200
n_outputs = 5
n_layers = 2
learning_rate = 0.001
batch_size = 50
n_epochs = 100
train_set_size = x_train.shape[0] # == 총 트레이닝 데이터 개수
test_set_size = x_test.shape[0] # == 총 테스트 데이터 개수

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="X")
y = tf.placeholder(tf.float32, [None, n_outputs], name="Y")

# use Basic RNN Cell
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
          for layer in range(n_layers)]

# use Basic LSTM Cell
# layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.relu)
#          for layer in range(n_layers)]

# use LSTM Cell with peephole connections
# layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons,
#                                  activation=tf.nn.leaky_relu, use_peepholes = True)
#          for layer in range(n_layers)]

# use GRU cell
# layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
#          for layer in range(n_layers)]

multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
print(rnn_outputs)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
print(stacked_rnn_outputs)
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
print(stacked_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs], name= "outputs")
print(outputs)
outputs = outputs[:, n_steps - 1, :]  # keep only last output of sequence
print(outputs)

loss = tf.reduce_mean(tf.square(outputs - y), name="loss")  # loss function = mean squared error
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer")
training_op = optimizer.minimize(loss,  name="training_op")

# run graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(int(n_epochs * train_set_size / batch_size)):
        x_batch, y_batch = get_next_batch(batch_size)  # fetch the next training batch
        sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
        if iteration % int(5 * train_set_size / batch_size) == 0:
            mse_train = loss.eval(feed_dict={X: x_train, y: y_train})
            mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid})
            print('%.2f epochs: MSE train/valid = %.6f/%.6f' % (
                iteration * batch_size / train_set_size, mse_train, mse_valid))

    y_train_pred = sess.run(outputs, feed_dict={X: x_train})
    y_valid_pred = sess.run(outputs, feed_dict={X: x_valid})
    y_test_pred = sess.run(outputs, feed_dict={X: x_test})
    y_real_prediction = sess.run(outputs, feed_dict={X: prediction_data})

ft = 0 # 0 = open, 1 = close, 2 = highest, 3 = lowest

## show predictions
plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);

plt.plot(np.arange(y_train.shape[0]), y_train[:,ft], color='blue', label='train target')

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_valid.shape[0]), y_valid[:,ft],
         color='gray', label='valid target')

plt.plot(np.arange(y_train.shape[0]+y_valid.shape[0],
                   y_train.shape[0]+y_test.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0]),y_train_pred[:,ft], color='red',
         label='train prediction')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_valid_pred.shape[0]),
         y_valid_pred[:,ft], color='orange', label='valid prediction')

plt.plot(np.arange(y_train_pred.shape[0]+y_valid_pred.shape[0],
                   y_train_pred.shape[0]+y_valid_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')

plt.title('past and future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best');

plt.subplot(1,2,2);

plt.plot(np.arange(y_train.shape[0] + y_valid.shape[0], y_train.shape[0] + y_valid.shape[0] + y_test.shape[0]),
         y_test[:, ft], color='black', label='test target')

plt.plot(np.arange(y_real_prediction.shape[0]),
         y_real_prediction[:, ft], color='green', label='test prediction')

plt.title('future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best');

corr_price_development_train = np.sum(np.equal(np.sign(y_train[:,3]-y_train[:,0]),
            np.sign(y_train_pred[:,3]-y_train_pred[:,0])).astype(int)) / y_train.shape[0]
corr_price_development_valid = np.sum(np.equal(np.sign(y_valid[:,3]-y_valid[:,0]),
            np.sign(y_valid_pred[:,3]-y_valid_pred[:,0])).astype(int)) / y_valid.shape[0]
corr_price_development_test = np.sum(np.equal(np.sign(y_test[:,3]-y_test[:,0]),
            np.sign(y_test_pred[:,3]-y_test_pred[:,0])).astype(int)) / y_test.shape[0]

print('correct sign prediction for close - open price for train/valid/test: %.2f/%.2f/%.2f'%(
    corr_price_development_train, corr_price_development_valid, corr_price_development_test))

plt.show()
#print(y_train_pred)
#print(denormalize_data(y_train_pred))
