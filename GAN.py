import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import os
from IPython import display
from tensorflow.keras import layers
from sklearn.utils import shuffle
#import math

#python "C:\Users\bradley.tjandra\AppData\Local\Continuum\anaconda3\Lib\site-packages\tensorboard\main.py" --logdir="C:\Users\bradley.tjandra\Dropbox\2019\Machine Learning_2019\Code\Other Implementations\GAN Dashboard"
# http://GZ0XVT2:6006

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.

class GAN():
  
  def __init__(self, sess):
    self.tensors = {}
    self.holders = {}
    self.ops = {}
    self.sess = sess
    self.create_graph()
    
  def create_generator_graph(self, Z):

    with tf.variable_scope("gener"):
      layer = tf.layers.dense(
          Z 
          , units=128
          , activation=tf.nn.relu
          , name = "lay1"
          )
      layer = tf.layers.dense(
          layer
          , units=784
          , activation=tf.sigmoid
          , name = "lay3"
          )
      layer = tf.reshape(layer, shape=[-1, 28, 28])

    return(layer)
      
  def create_discriminator_graph(self):
    
    layers = []
    
    with tf.variable_scope("adver"):
      layers.append(tf.layers.Dense(
          units = 240
          , activation=tf.nn.relu
          , name = "adver/lay1"
          ))
      layers.append(tf.layers.Dense(
          units=1
          , kernel_initializer=tf.random_normal_initializer(-0.005,0.005)
          , name = "adver/lay3"
          ))
    return(layers)
  
  def apply_discriminator_graph(self, adver_layers, images):
    
    layer = tf.reshape(images, shape=[-1, 784])
    for L in adver_layers:
      layer = L(layer)
    preds = tf.nn.sigmoid(layer)
    return(preds, layer)
    
  def create_graph(self):
      
    Z = tf.placeholder(tf.float32, [None, 100], name="Z")
    real_images = tf.placeholder(tf.float32, [None, 28, 28])
    
    fake_images = self.create_generator_graph(Z)
    adver_layers = self.create_discriminator_graph()
    pred_on_real, logits_real  = self.apply_discriminator_graph(adver_layers, real_images)
    pred_on_fake, logits_fake = self.apply_discriminator_graph(adver_layers, fake_images)
    adver_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real)
                                                , logits=logits_real
                                               )
        + tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake)
                                                 , logits=logits_fake
                                                 )
    )
    gener_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake)
                                               , logits=logits_fake)
    )
    adver_optimizer = tf.train.AdamOptimizer(.001)
    gener_optimizer = tf.train.AdamOptimizer(.001)
    gener_grads = gener_optimizer.compute_gradients(gener_loss, fake_images)
    adver_train_op = adver_optimizer.minimize(adver_loss
                                              , var_list=tf.trainable_variables(scope="adver")
                                             )
    gener_train_op = gener_optimizer.minimize(gener_loss
                                              , var_list=tf.trainable_variables(scope="gener")
                                             )
    
    tf.summary.histogram("real_pred",tf.reduce_mean(pred_on_real))
    tf.summary.histogram("fake_pred",tf.reduce_mean(pred_on_fake))
    tf.summary.histogram("adversarial_loss",adver_loss)
    tf.summary.histogram("generator_loss",gener_loss)
    tf.summary.histogram("grad_gener", gener_grads)
    tf.summary.histogram("grad_gener_mean", tf.reduce_mean(gener_grads))
    tf.summary.histogram("grad_gener_max", tf.reduce_max(gener_grads))
    tf.summary.histogram("grad_gener_min", tf.reduce_min(gener_grads))
    merge = tf.summary.merge_all()
    
    self.holders['Z'] = Z
    self.holders['real_images'] = real_images
    self.tensors['merge'] = merge
    self.tensors['gener_grads'] = gener_grads
    self.tensors['pred_on_real'] = pred_on_real
    self.tensors['pred_on_fake'] = pred_on_fake
    self.tensors['fake_images'] = fake_images
    self.ops['adver'] = adver_train_op
    self.ops['gener'] = gener_train_op

  def prior(self, batch_size):
    return(np.random.normal(size=[batch_size,100]))
    
  def train_network(self, k, real_images):
    
    for i in range(k):
      Z = self.prior(real_images.shape[0])
      summary, _, pred_on_real, pred_on_fake = self.sess.run((
          self.tensors['merge']
          , self.ops['adver']
          , self.tensors['pred_on_real']
          , self.tensors['pred_on_fake']
        )
        , {
          self.holders['Z'] : Z
          , self.holders['real_images'] : real_images
          }
        )
  
    self.sess.run(self.ops['gener'], {
        self.holders['Z'] : Z
        })
  
    return(summary, np.mean(pred_on_real), 1-np.mean(pred_on_fake))  
  
  def generate_images(self, m=None, prior=None):
    
    if prior is None:
      prior = self.prior(m)
    images = self.sess.run(self.tensors['fake_images'],{
        self.holders['Z']: prior})
    return(np.reshape(images, (-1,28,28)))
  
  def see_grads(self, Z_test=None):
    
    if Z_test is None:
      Z_test = self.prior(1)
      
    grads = self.sess.run(
        self.tensors['gener_grads']
        , {
            self.holders['Z'] : Z_test
        })
    return(grads)
  
def sample_image(images, j, minibatch_size):
  if j == 0:
    images = shuffle(images)
  start = j * minibatch_size
  stop  = (j+1) * minibatch_size
  return(images[start:stop,])  

def generate_and_save_images(gan, epoch, Z_test):
  
  fakes = gan.generate_images(prior=Z_test)
  grads = gan.see_grads(Z_test)
  n_cols = 3
  n_rows = 4
  
  display.clear_output(wait=True)
  fig = plt.figure(figsize=(8,8))

  for i in range(n_rows):
      plt.subplot(n_rows,n_cols, n_cols*i+1)
      plt.imshow(fakes[i, :, :], cmap='gray_r')
      plt.axis('off')
      
      plt.subplot(n_rows,n_cols, n_cols*i+2)
      g = grads[0][0][i]
      if len(g.shape) == 1:
        g = np.expand_dims(g,-1)
      plt.imshow(g, cmap='gray')
      plt.colorbar()
      plt.axis('off')
      
      plt.subplot(n_rows,n_cols, n_cols*i+3)
      g = grads[0][1][i]
      if len(g.shape) == 1:
        g = np.expand_dims(g,-1)
      plt.imshow(g, cmap='gray_r')
      plt.colorbar()
      plt.axis('off')
      

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  
  print(np.max(grads[0][0]), np.min(grads[0][0]))
  
tf.reset_default_graph()  
sess = tf.Session()
gan = GAN(sess)  
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter(os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

Z_test = np.random.normal(size=[4,100])

for i in range(100):
  
  for j in range(int(x_train.shape[0]/128)):
  
    real_images = sample_image(x_train, j, 128)
    summary, real_acc, fake_acc = gan.train_network(1, real_images)
    if j % 50 == 0:
      train_writer.add_summary(summary, i)

    
  generate_and_save_images(gan, i, Z_test)
  print(real_acc, fake_acc)
    
    

