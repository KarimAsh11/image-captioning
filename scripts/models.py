import tensorflow as tf


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc_i = tf.keras.layers.Dense(embedding_dim)
        self.fc_g = tf.keras.layers.Dense(embedding_dim)

    def call(self, a_i):
        a_g = tf.reduce_mean(a_i, 1, keepdims=True)
        v_i = tf.nn.relu(self.fc_i(a_i))
        v_g = tf.nn.relu(self.fc_g(a_g))

        return v_i, v_g


class Beta_Adaptive_Attention(tf.keras.Model):
  def __init__(self, units):
    super(Adaptive_Attention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.W3 = tf.keras.layers.Dense(units)

    self.V1 = tf.keras.layers.Dense(1)
    self.V2 = tf.keras.layers.Dense(1)

    self.B  = tf.Variable(tf.zeros(1))

  def call(self, v_i, hidden, st):
    hidden_with_time_axis = tf.expand_dims(hidden, 1)
    st_with_time_axis     = tf.expand_dims(st, 1)
    Beta                  = self.B
    spatial               = self.W1(v_i)+ self.W2(hidden_with_time_axis)
    z_t                   = self.V1(tf.nn.tanh(spatial)) 

    spatial_attention_weights = tf.nn.softmax(z_t, axis=1) 
    context_vector            = spatial_attention_weights * v_i
    context_vector            = tf.reduce_sum(context_vector, axis=1)
    c_hat = Beta*st + (1-Beta)*context_vector

    return c_hat, spatial_attention_weights, Beta



class Adaptive_Attention(tf.keras.Model):
  def __init__(self, units):
    super(Adaptive_Attention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.W3 = tf.keras.layers.Dense(units)

    self.V1 = tf.keras.layers.Dense(1)
    self.V2 = tf.keras.layers.Dense(1)

  def call(self, v_i, hidden, st):
    hidden_with_time_axis = tf.expand_dims(hidden, 1)
    st_with_time_axis     = tf.expand_dims(st, 1)

    spatial           = self.W1(v_i)+ self.W2(hidden_with_time_axis)
    z_t               = self.V1(tf.nn.tanh(spatial)) 
    y_t               = self.V2(tf.nn.tanh(self.W3(st_with_time_axis)  + self.W2(hidden_with_time_axis))) 
    z_hat             = tf.concat([z_t, y_t], axis=1)
    attention_weights = tf.nn.softmax(z_hat, axis=1) 

    concat_features   = tf.concat([v_i, st_with_time_axis], axis=1)
    context_vector    = attention_weights * concat_features
    context_vector    = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Spatial_Attention(tf.keras.Model):
  def __init__(self, units):
    super(Spatial_Attention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V1 = tf.keras.layers.Dense(1)

  def call(self, v_i, hidden, st):
    hidden_with_time_axis = tf.expand_dims(hidden, 1)
    z                     = self.W1(v_i)+ self.W2(hidden_with_time_axis)
    z_t                   = self.V1(tf.nn.tanh(z)) 
    attention_weights     = tf.nn.softmax(z_t, axis=1)
    context_vector        = attention_weights * v_i
    context_vector        = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size, adaptive):
    super(RNN_Decoder, self).__init__()
    self.units = units
    
    if adaptive == "adaptive": 
      self.attention = Adaptive_Attention(self.units)
      print("Adaptive attention model")
    elif adaptive == "spatial":  
      self.attention = Spatial_Attention(self.units)
      print("Spatial attention model")
    
    self.embedding     = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm_cell = tf.keras.layers.LSTM(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform' , name="Adaptive_LSTM")
    self.W_x       = tf.keras.layers.Dense(self.units)
    self.W_h       = tf.keras.layers.Dense(self.units)

    self.fc1       = tf.keras.layers.Dense(vocab_size)
 
  def call(self, word, v_i, v_g, h_old, m_old):
    w_t = self.embedding(word)
    x_t = tf.concat([w_t, v_g], axis=-1)
    
    decoder_state = [h_old, m_old]
    out, ht, mt   = self.lstm_cell(x_t, initial_state=decoder_state)
    sen_gate      = tf.nn.sigmoid(self.W_x(tf.squeeze(x_t, 1)) + self.W_h(h_old))
    st            =  tf.math.multiply(sen_gate, tf.nn.tanh(mt))

    context_vector, attention_weights = self.attention(v_i, ht, st)
    y = self.fc1(context_vector + ht)

    return y, ht, mt, attention_weights

  def reset_state(self, batch_size):

    return tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))
