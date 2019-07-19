import tensorflow as tf
import random
import numpy as np

def data_set(data_url):
  """process data input."""
  data = []
  word_count = []
  fin = open(data_url)
  while True:
    line = fin.readline()
    if not line:
      break
    id_freqs = line.split()
    doc = {}
    count = 0
    for id_freq in id_freqs[1:]:
      items = id_freq.split(':')
      # python starts from 0
      if int(items[0])-1<0:
        print('WARNING INDICES!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      doc[int(items[0])-1] = int(items[1])
      count += int(items[1])
    if count > 0:
      data.append(doc)
      word_count.append(count)
  fin.close()
  return data, word_count

def create_batches(data_size, batch_size, shuffle=True):
  """create index by batches."""
  batches = []
  ids = list(range(data_size))
  if shuffle:
    random.shuffle(ids)
  for i in range(int(data_size / batch_size)):
    start = i * batch_size
    end = (i + 1) * batch_size
    batches.append(ids[start:end])
  # the batch of which the length is less than batch_size
  rest = data_size % batch_size
  if rest > 0:
    batches.append(list(ids[-rest:]) + [-1] * (batch_size - rest))  # -1 as padding
  return batches

def fetch_data(data, count, idx_batch, vocab_size):
  """fetch input data by batch."""
  batch_size = len(idx_batch)
  data_batch = np.zeros((batch_size, vocab_size))
  count_batch = []
  mask = np.zeros(batch_size)
  indices = []
  values = []
  for i, doc_id in enumerate(idx_batch):
    if doc_id != -1:
      for word_id, freq in data[doc_id].items():
        data_batch[i, word_id] = freq
      count_batch.append(count[doc_id])
      mask[i]=1.0
    else:
      count_batch.append(0)
  return data_batch, count_batch, mask

def variable_parser(var_list, prefix):
  """return a subset of the all_variables by prefix."""
  ret_list = []
  for var in var_list:
    varname = var.name
    varprefix = varname.split('/')[0]
    if varprefix == prefix:
      ret_list.append(var)
    elif prefix in varname:
      ret_list.append(var)
  return ret_list

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

def linear_LDA(inputs,
           output_size,
           no_bias=False,
           bias_start_zero=False,
           matrix_start_zero=False,
           scope=None):
  """Define a linear connection."""
  with tf.variable_scope(scope or 'Linear'):
    if matrix_start_zero:
      matrix_initializer = tf.constant_initializer(0)
    else:
      matrix_initializer =  None
    if bias_start_zero:
      bias_initializer = tf.constant_initializer(0)
    else:
      bias_initializer = None
    input_size = inputs.get_shape()[1].value
    matrix = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.Variable(xavier_init(input_size, output_size))))
    
    output = tf.matmul(inputs, matrix)#no softmax on input, it should already be normalized
    if not no_bias:
      bias_term = tf.get_variable('Bias', [output_size], 
                                initializer=bias_initializer)
      output = output + bias_term
  return output


def linear(inputs,
           output_size,
           no_bias=False,
           bias_start_zero=False,
           matrix_start_zero=False,
           scope=None,
           weights=None):
  """Define a linear connection."""
  with tf.variable_scope(scope or 'Linear'):
    if matrix_start_zero:
      matrix_initializer = tf.constant_initializer(0)
    else:
      matrix_initializer =  tf.truncated_normal_initializer(mean = 0.0, stddev=0.01)
    if bias_start_zero:
      bias_initializer = tf.constant_initializer(0)
    else:
      bias_initializer = None
    input_size = inputs.get_shape()[1].value
   
    if weights is not None:
      matrix=weights
    else:
      matrix = tf.get_variable('Matrix', [input_size, output_size],initializer=matrix_initializer)
    
    output = tf.matmul(inputs, matrix)
    if not no_bias:
      bias_term = tf.get_variable('Bias', [output_size], 
                                initializer=bias_initializer)
      output = output + bias_term
  return output

def mlp(inputs, 
        mlp_hidden=[], 
        mlp_nonlinearity=tf.nn.tanh,
        scope=None):
  """Define an MLP."""
  with tf.variable_scope(scope or 'Linear'):
    mlp_layer = len(mlp_hidden)
    res = inputs
    for l in range(mlp_layer):
      res = mlp_nonlinearity(linear(res, mlp_hidden[l], scope='l'+str(l)))
    return res
    
    
def print_top_words(beta, feature_names, n_top_words=10,label_names=None,result_file=None):
    print('---------------Printing the Topics------------------')
    if result_file!=None:
      result_file.write('---------------Printing the Topics------------------\n')
    for i in range(len(beta)):
        topic_string = " ".join([feature_names[j]
            for j in beta[i].argsort()[:-n_top_words - 1:-1]])
        print(topic_string)
        if result_file!=None:
          result_file.write(topic_string+'\n')
    if result_file!=None:
      result_file.write('---------------End of Topics------------------\n')
    print('---------------End of Topics------------------')

def count_word_combination(dataset,combination):
  count = 0
  w1,w2 = combination
  for data in dataset:
    w1_found=False
    w2_found=False
    for word_id, freq in data.items():
      if not w1_found and word_id==w1:
        w1_found=True
      elif not w2_found and word_id==w2:
        w2_found=True
      if w1_found and w2_found:
        count+=1
        break
  return count
  
def count_word(dataset,word):
  count=0
  for data in dataset:
    for word_id, freq in data.items():
      if word_id==word:
        count+=1
        break
  return count      

def topic_coherence(dataset,beta, feature_names, n_top_words=10):
  word_counts={}
  word_combination_counts={}
  length = len(dataset)
  #go through dataset:
  #for each word combination:
    #\frac{log\frac{P(wi,wj)}{P(wi)*P(wj)}}{-logP(wi,wj)}
  coherence_sum=0.0
  coherence_count=0
  topic_coherence_sum=0.0
  
  for i in range(len(beta)):
    top_words = [j
            for j in beta[i].argsort()[:-n_top_words - 1:-1]]
    topic_coherence = 0
    topic_coherence_count=0.0
    for i,word in enumerate(top_words):
      if word not in word_counts:
        count = count_word(dataset,word)
        word_counts[word]=count
      for j in range(i):
        word2 = top_words[j]
        combination = (word,word2)
        if combination not in word_combination_counts:
          count = count_word_combination(dataset,combination)
          word_combination_counts[combination]=count
        #now calculate coherence
        wc1 = word_counts[word]/float(length)
        wc2 = word_counts[word2]/float(length)
        cc = (word_combination_counts[combination])/float(length)
        if cc>0:
          coherence = math.log(cc/float(wc1*wc2))/(-math.log(cc))
          topic_coherence+=coherence
          coherence_sum+=coherence
        coherence_count+=1
        topic_coherence_count+=1
    topic_coherence_sum+=topic_coherence/float(topic_coherence_count)
  return coherence_sum/float(coherence_count),topic_coherence_sum/float(len(beta))
