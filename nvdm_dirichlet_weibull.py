"""NVDM Tensorflow implementation by Yishu Miao, adapted to work with the Dirichlet distribution by Sophie Burkhardt"""
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math
import os
import utils as utils
import sys
import argparse
import pickle

np.random.seed(0)
tf.set_random_seed(0)

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 200, 'Batch size.')
flags.DEFINE_integer('n_hidden', 100, 'Size of each hidden layer.')
flags.DEFINE_boolean('test', True, 'Process test data.')
flags.DEFINE_string('non_linearity', 'relu', 'Non-linearity of the MLP.')
flags.DEFINE_string('summaries_dir','summaries','where to save the summaries')
FLAGS = flags.FLAGS

class NVDM(object):
    """ Neural Variational Document Model -- BOW VAE.
    """
    def __init__(self, 
                 vocab_size,
                 n_hidden,
                 n_topic,
                 learning_rate, 
                 batch_size,
                 non_linearity,
                 adam_beta1,
                 adam_beta2,
                 dir_prior):
        tf.reset_default_graph()
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self.n_sample = 1#n_sample
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        lda=False
        self.x = tf.placeholder(tf.float32, [None, vocab_size], name='input')
        self.mask = tf.placeholder(tf.float32, [None], name='mask')  # mask paddings
        self.warm_up = tf.placeholder(tf.float32, (), name='warm_up')  # warm up
        self.adam_beta1=adam_beta1
        self.adam_beta2=adam_beta2
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.min_alpha = tf.placeholder(tf.float32,(), name='min_alpha')
        # encoder
        with tf.variable_scope('encoder'): 
          self.enc_vec = utils.mlp(self.x, [self.n_hidden], self.non_linearity)
          self.enc_vec = tf.nn.dropout(self.enc_vec,self.keep_prob)
          self.k = tf.contrib.layers.batch_norm(utils.linear(self.enc_vec, self.n_topic, scope='k'))
          self.l = tf.contrib.layers.batch_norm(utils.linear(self.enc_vec, self.n_topic, scope='l'))#tf.contrib.layers.batch_norm()
          self.weibull_k = tf.math.softplus(self.k)
          self.weibull_k = tf.clip_by_value(self.weibull_k,0.1,1e10)
          self.weibull_l = tf.math.softplus(self.l)
          #Dirichlet prior alpha0
          self.prior = tf.ones((batch_size,self.n_topic), dtype=tf.float32, name='prior')*dir_prior
          self.analytical_kld = KL_GamWei_Paper(self.prior,1.,self.weibull_k,self.weibull_l)
        with tf.variable_scope('decoder'): 
          if self.n_sample ==1:  # single sample
            u = tf.random_uniform((batch_size,self.n_topic))
            with tf.variable_scope('prob'):
                #CDF transform
                self.doc_vec = self.weibull_l*tf.pow(-tf.log(1.-u),1./self.weibull_k)
                #normalize
                self.doc_vec = tf.div(self.doc_vec,tf.reshape(tf.reduce_sum(self.doc_vec,1), (-1, 1)))
                self.doc_vec.set_shape(self.weibull_l.get_shape())
            #reconstruction
            if lda:
              logits = tf.log(tf.clip_by_value(utils.linear_LDA(self.doc_vec, self.vocab_size, scope='projection',no_bias=True),1e-10,1.0))
            else:
              logits = tf.nn.log_softmax(tf.contrib.layers.batch_norm(utils.linear(self.doc_vec, self.vocab_size, scope='projection',no_bias=True)))
            self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.x), 1)
          # multiple samples
          #not implemented
       
        
        self.objective = self.recons_loss + self.warm_up*self.analytical_kld
        self.true_objective = self.recons_loss + self.analytical_kld
       
        self.analytical_objective = self.recons_loss+self.analytical_kld
       
        fullvars = tf.trainable_variables()

        enc_vars = utils.variable_parser(fullvars, 'encoder')
        dec_vars = utils.variable_parser(fullvars, 'decoder')
       
        #this is the standard gradient for the reconstruction network
        dec_grads = tf.gradients(self.objective, dec_vars)
        
        #####################################################
        #Now calculate the gradient for the encoding network#
        #####################################################
       
        
        kl_grad = tf.gradients(self.analytical_kld,enc_vars)
        
        g_rep = tf.gradients(self.recons_loss,enc_vars)
        
        enc_grads = [g_r+self.warm_up*g_e for g_r,g_e in zip(g_rep,kl_grad)]
        
       
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=self.adam_beta1,beta2=self.adam_beta2)
        self.optim_enc = optimizer.apply_gradients(zip(enc_grads, enc_vars))
        self.optim_dec = optimizer.apply_gradients(zip(dec_grads, dec_vars))
        self.optim_all = optimizer.apply_gradients(list(zip(enc_grads, enc_vars))+list(zip(dec_grads, dec_vars)))
        
def KL_GamWei(GamShape,GamScale,WeiShape,WeiScale):
  eulergamma=0.5772
  Out = eulergamma * (1.-1./WeiShape) + tf.log(WeiScale/WeiShape+1e-10) + 1. - tf.lgamma(GamShape) + (GamShape-1.)*(tf.log(WeiScale+1e-10)-eulergamma/WeiShape) - GamScale*WeiScale*tf.exp(tf.lgamma(1. + 1./WeiShape))
        
  return tf.reduce_sum(Out,axis=1)

def KL_GamWei_Paper(GamShape,GamScale,WeiShape,WeiScale):
  eulergamma=0.5772
  kld = GamShape*tf.log(WeiScale)-eulergamma*GamShape/WeiShape-tf.log(WeiShape)-WeiScale*tf.exp(tf.lgamma(1+1./WeiShape))+eulergamma+1.-tf.lgamma(GamShape)
  return tf.reduce_sum(-kld,axis=1)

def train(sess, model, 
          train_url, 
          test_url, 
          batch_size, 
          vocab_size,
          alternate_epochs=1,#10
          lexicon=[],
          result_file='test.txt',
          warm_up_period=100):
          
  """train nvdm model."""
  train_set, train_count = utils.data_set(train_url)
  test_set, test_count = utils.data_set(test_url)
  # hold-out development dataset
  train_size=len(train_set)
  validation_size=int(train_size*0.1)
  dev_set = train_set[:validation_size]
  dev_count = train_count[:validation_size]
  train_set = train_set[validation_size:]
  train_count = train_count[validation_size:]
  optimize_jointly = True
  dev_batches = utils.create_batches(len(dev_set), batch_size, shuffle=False)
  test_batches = utils.create_batches(len(test_set), batch_size, shuffle=False)
  warm_up = 0
  min_alpha = 0.00001#

  best_print_ana_ppx=1e10
  early_stopping_iters=30
  no_improvement_iters=0
  stopped=False
  epoch=-1
  #for epoch in range(training_epochs):
  while not stopped:
    epoch+=1
    train_batches = utils.create_batches(len(train_set), batch_size, shuffle=True)
    if warm_up<1.:
      warm_up += 1./warm_up_period
    else:
      warm_up=1.
   
    #-------------------------------
    # train
    #for switch in range(0, 2):
    if optimize_jointly:
      optim = model.optim_all
      print_mode = 'updating encoder and decoder'
    elif switch == 0:
      optim = model.optim_dec
      print_mode = 'updating decoder'
    else:
      optim = model.optim_enc
      print_mode = 'updating encoder'
    for i in range(alternate_epochs):
      loss_sum = 0.0
      ana_loss_sum = 0.0
      ppx_sum = 0.0
      kld_sum_train = 0.0
      ana_kld_sum_train = 0.0
      word_count = 0
      doc_count = 0
      recon_sum=0.0
      for idx_batch in train_batches:
        data_batch, count_batch, mask = utils.fetch_data(
        train_set, train_count, idx_batch, vocab_size)
        input_feed = {model.x.name: data_batch, model.mask.name: mask,model.keep_prob.name: 0.75,model.warm_up.name: warm_up,model.min_alpha.name:min_alpha}
        _, (loss,recon,ana_loss,ana_kld_train) = sess.run((optim, 
                                    [model.true_objective, model.recons_loss,model.analytical_objective,model.analytical_kld]),
                                    input_feed)
        loss_sum += np.sum(loss)
        ana_loss_sum += np.sum(ana_loss)
        kld_sum_train += np.sum(ana_kld_train) / np.sum(mask) 
        ana_kld_sum_train += np.sum(ana_kld_train) / np.sum(mask)
        word_count += np.sum(count_batch)
        # to avoid nan error
        count_batch = np.add(count_batch, 1e-12)
        # per document loss
        ppx_sum += np.sum(np.divide(loss, count_batch)) 
        doc_count += np.sum(mask)
        recon_sum+=np.sum(recon)
      print_loss = recon_sum/len(train_batches)
      dec_vars = utils.variable_parser(tf.trainable_variables(), 'decoder')
      phi = dec_vars[0]
      phi = sess.run(phi)
      utils.print_top_words(phi, lexicon,result_file=None)
      print_ppx = np.exp(loss_sum / word_count)
      print_ana_ppx = np.exp(ana_loss_sum / word_count)
      print_ppx_perdoc = np.exp(ppx_sum / doc_count)
      print_kld_train = kld_sum_train/len(train_batches)
      print_ana_kld_train = ana_kld_sum_train/len(train_batches)
      print('| Epoch train: {:d} |'.format(epoch+1), 
               print_mode, '{:d}'.format(i),
               '| Corpus ppx: {:.5f}'.format(print_ppx),  # perplexity for all docs
               '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),  # perplexity for per doc
               '| KLD: {:.5}'.format(print_kld_train),
               '| Loss: {:.5}'.format(print_loss),
               '| ppx anal.: {:.5f}'.format(print_ana_ppx),
               '|KLD anal.: {:.5f}'.format(print_ana_kld_train))
     
    
    #-------------------------------
    # dev
    loss_sum = 0.0
    kld_sum_dev = 0.0
    ppx_sum = 0.0
    word_count = 0
    doc_count = 0
    recon_sum=0.0
    print_ana_ppx = 0.0
    ana_loss_sum = 0.0
    for idx_batch in dev_batches:
      data_batch, count_batch, mask = utils.fetch_data(
          dev_set, dev_count, idx_batch, vocab_size)
      input_feed = {model.x.name: data_batch, model.mask.name: mask,model.keep_prob.name: 1.0,model.warm_up.name: 1.0,model.min_alpha.name:min_alpha}
      loss,recon,ana_kld,ana_loss = sess.run([model.objective, model.recons_loss, model.analytical_kld,model.analytical_objective],
                           input_feed)
      loss_sum += np.sum(loss)
      ana_loss_sum += np.sum(ana_loss)
      kld_sum_dev += np.sum(ana_kld) / np.sum(mask)  
      word_count += np.sum(count_batch)
      count_batch = np.add(count_batch, 1e-12)
      ppx_sum += np.sum(np.divide(loss, count_batch))
      doc_count += np.sum(mask) 
      recon_sum+=np.sum(recon)
    print_ana_ppx = np.exp(ana_loss_sum / word_count)
    print_ppx = np.exp(loss_sum / word_count)
    print_ppx_perdoc = np.exp(ppx_sum / doc_count)
    print_kld_dev = kld_sum_dev/len(dev_batches)
    print_loss = recon_sum/len(dev_batches)
    if print_ppx<best_print_ana_ppx:
      no_improvement_iters=0
      best_print_ana_ppx=print_ppx
      #check on validation set, if ppx better-> save improved model
      
      tf.train.Saver().save(sess, 'models/improved_model_weibull') 
      
    else:
      no_improvement_iters+=1
      print('no_improvement_iters',no_improvement_iters,'best ppx',best_print_ana_ppx)
      if no_improvement_iters>=early_stopping_iters:
          #if model has not improved for 30 iterations, stop training
          ###########STOP TRAINING############
          stopped=True
          print('stop training after',epoch,'iterations,no_improvement_iters',no_improvement_iters)
          ###########LOAD BEST MODEL##########
          print('load stored model')
          tf.train.Saver().restore(sess,'models/improved_model_weibull')
    print('| Epoch dev: {:d} |'.format(epoch+1), 
           '| Perplexity: {:.9f}'.format(print_ppx),
           '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
           '| KLD: {:.5}'.format(print_kld_dev)  ,
           '| Loss: {:.5}'.format(print_loss))  

    #-------------------------------
    # test
    if FLAGS.test:
      
      loss_sum = 0.0
      kld_sum_test = 0.0
      ppx_sum = 0.0
      word_count = 0
      doc_count = 0
      recon_sum = 0.0
      ana_loss_sum = 0.0
      ana_kld_sum_test = 0.0
      for idx_batch in test_batches:
        data_batch, count_batch, mask = utils.fetch_data(
          test_set, test_count, idx_batch, vocab_size)
        input_feed = {model.x.name: data_batch, model.mask.name: mask,model.keep_prob.name: 1.0,model.warm_up.name: 1.0,model.min_alpha.name:min_alpha}
        loss, recon,ana_loss,ana_kld_test = sess.run([model.objective, model.recons_loss,model.analytical_objective,model.analytical_kld],
                             input_feed)
        loss_sum += np.sum(loss)
        kld_sum_test += np.sum(ana_kld_test)/np.sum(mask) 
        ana_loss_sum += np.sum(ana_loss)
        ana_kld_sum_test += np.sum(ana_kld_test) / np.sum(mask)
        word_count += np.sum(count_batch)
        count_batch = np.add(count_batch, 1e-12)
        ppx_sum += np.sum(np.divide(loss, count_batch))
        doc_count += np.sum(mask) 
        recon_sum+=np.sum(recon)
      print_loss = recon_sum/len(test_batches)
      print_ppx = np.exp(loss_sum / word_count)
      print_ppx_perdoc = np.exp(ppx_sum / doc_count)
      print_kld_test = kld_sum_test/len(test_batches)
      print_ana_ppx = np.exp(ana_loss_sum / word_count)
      print_ana_kld_test = ana_kld_sum_test/len(train_batches)
      print('| Epoch test: {:d} |'.format(epoch+1), 
             '| Perplexity: {:.9f}'.format(print_ppx),
             '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
             '| KLD: {:.5}'.format(print_kld_test),
             '| Loss: {:.5}'.format(print_loss),
             '| ppx anal.: {:.5f}'.format(print_ana_ppx),
               '|KLD anal.: {:.5f}'.format(print_ana_kld_test)) 
      if stopped:#epoch==training_epochs-1:
        #only do it once in the end
        print('calculate topic coherence (might take a few minutes)')
        coherence=utils.topic_coherence(test_set,phi, lexicon)
        print('topic coherence',str(coherence))
  
  
def myrelu(features):
    return tf.maximum(features, 0.0)

def parseArgs():
    #get line from config file
    args = sys.argv
    linum = int(args[1])
    argstring=''
    configname = 'tfconfig'
    with open(configname,'r') as rf:
        for i,line in enumerate(rf):
            #print i,line
            argstring = line
            if i+1==linum:
                print(line)
                break
    argparser = argparse.ArgumentParser()
    #define arguments
    argparser.add_argument('--adam_beta1',default=0.9, type=float)
    argparser.add_argument('--adam_beta2',default=0.999, type=float)
    argparser.add_argument('--learning_rate',default=1e-3, type=float)
    argparser.add_argument('--dir_prior',default=0.1, type=float)
    argparser.add_argument('--n_topic',default=50, type=int)
    argparser.add_argument('--warm_up_period',default=100, type=int)
    argparser.add_argument('--data_dir',default='data/20news', type=str)
    return argparser.parse_args(argstring.split())

def main(argv=None):
    if FLAGS.non_linearity == 'tanh':
      non_linearity = tf.nn.tanh
    elif FLAGS.non_linearity == 'sigmoid':
      non_linearity = tf.nn.sigmoid
    else:
      non_linearity = myrelu#max(features, 1.1)#tf.nn.relu
    
    args = parseArgs()
    adam_beta1 = args.adam_beta1
    adam_beta2 = args.adam_beta2
    learning_rate = args.learning_rate
    dir_prior = args.dir_prior
    warm_up_period = args.warm_up_period
    n_topic = args.n_topic
    lexicon=[]
    vocab_path = os.path.join(args.data_dir, 'vocab.new')
    with open(vocab_path,'r') as rf:
        for line in rf:
            word = line.split()[0]
            lexicon.append(word)
    vocab_size=len(lexicon)
  
    nvdm = NVDM(vocab_size=vocab_size,
                n_hidden=FLAGS.n_hidden,
                n_topic=n_topic, 
                learning_rate=learning_rate, 
                batch_size=FLAGS.batch_size,
                non_linearity=non_linearity,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                dir_prior=dir_prior)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    result = sess.run(init)
    train_url = os.path.join(args.data_dir, 'train.feat')
    test_url = os.path.join(args.data_dir, 'test.feat')
    
    train(sess, nvdm, train_url, test_url, FLAGS.batch_size,vocab_size,lexicon=lexicon,
                result_file=None,
                warm_up_period = warm_up_period)

if __name__ == '__main__':
    tf.app.run()
