"""
Code to train the model
"""
import sys
sys.path.insert(0, '/content/gdrive/My Drive/Colab Notebooks')
sys.path.insert(0, '/content/gdrive/My Drive/Colab Notebooks/dataset.py')
import dataset
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
#from matplotlib.image import imread
# cv2:
# sys:
# argpars:
import cv2,sys,argparse


'''
  @brief  Initialzing the conv layer
  @pre    input, num_input_channels, filter_size, num_filters
'''
#####################################################
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters):        # Number of filters.

    # Shape of the filter-weights for the convolution.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    # Varaible: save tensor in memory
    # truncated_normal: 절단정규분포로부터 난수값을 반환
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    # Create new biases, one for each filter.
    # constant: 상수값 생성
    biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

    # 2d convolution layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 2, 2, 1],
                         padding='VALID')

    # A bias-value is added to each filter-channel.
    layer += biases

    return layer


'''
  @brief  Initailize the max pool layer
  @detail performs the max pooling on the input
  @pre    layer, ksize, strides
  @return return layer
'''
####################################################
def max_pool(layer,ksize,strides):
    # performs the max pooling on the input
    layer = tf.nn.max_pool(value=layer,
                           ksize=ksize,
                           strides = strides,
                           padding = 'VALID')
    return layer
    

'''
  @brief  fully connected layer
  @pre    input, num_inputs, num_outputs, use_relu
'''
####################################################
def new_fc_layer(input,           # The previous layer.
                 num_inputs,      # Num. inputs from prev. layer.
                 num_outputs,     # Num. outputs
                 use_relu=True):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights =tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
    
    #Include Drop-out as well to avoid overfitting
    # x_drop = tf.nn.dropout(input, keep_prob=keep_prob_input)
    
    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    # tf.matmul: multiply the matrixes
    layer = tf.matmul(input, weights) + biases

    # Use ReLU function
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer    


'''
  @detail flatten the layer
  @pre    layer
  @post   flatten the layer to [-1, num_features]
  @return layer_flat, num_features
'''  
####################################################
def flatten_layer(layer):
    # Get the shape of the input layer.
    # shape: the dimension of matrix
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # num_elements(): return the size of requested array variable
    num_features = layer_shape[1:4].num_elements()
    
    # reshpe the "layer" to [-1, num_features]
    # [-1] : 1-dimensional layer
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    return layer_flat, num_features


'''
  @brief  Model class
'''

####################################################
    
class Model:
    '''
    @brief  default constructor
    '''
    def __init__(self,in_dir,save_folder=None):
        # dataset
        dataset = dataset.load_cached(cache_path='gdrive/My Drive/Colab Notebooks/data/', in_dir=in_dir)
        # number of classes
        self.num_classes = dataset.num_classes

        # get training set
        image_paths_train, cls_train, self.labels_train = dataset.get_training_set()
        # get test set
        image_paths_test, self.cls_test, self.labels_test = dataset.get_test_set()
        
        ##############################IMAGE PARAMETERS#####################################
        self.img_size = 128
        self.num_channels = 3
        # batch size
        self.train_batch_size = 64
        self.test_batch_size = 64
        ###################################################################################
        # placeholder: setting the matrix(mapping)
        # @params dtype:  data type
        # @params shape:  shape of input data
        # @params name:   name of placeholder
        # x: feature, x_imange: reshape of feature
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size,self.img_size,self.num_channels], name='x')
        self.x_image = tf.reshape(self.x, [-1, self.img_size, self.img_size, self.num_channels])
        # placeholder: setting the matrix(mapping)
        # y true value
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        # tf.argmax: return the max value
        self.y_true_cls = tf.argmax(self.y_true, axis=1) #The True class Value

        self.keep_prob = tf.placeholder(tf.float32)
        self.keep_prob_2 = tf.placeholder(tf.float32)
        self.y_pred_cls = None
        # train images
        self.train_images= self.load_images(image_paths_train)
        # test images
        self.test_images= self.load_images(image_paths_test)
        self.save_folder=save_folder
        self.optimizer,self.accuracy = self.define_model()        
        
    '''
      @brief load the images form disk
    '''
    def load_images(self,image_paths):
        # Load the images from disk.
        # cv2.imread(): read the image file
        images = [cv2.imread(path,1) for path in image_paths]
        
        # Convert to a numpy array and return it in the form of [num_images,size,size,channel]
        #print(np.asarray(images[0]).shape)
        return np.asarray(images)
    
    '''
      @brief  define the model
    '''
    def define_model(self):
        #Convolution Layer 1
        filter_size1 = 10          # Convolution filters are 10 x 10 
        num_filters1 = 24         # There are 24 of these filters.

        # Convolutional Layer 2
        filter_size2 = 7          # Convolution filters are 7 x 7 
        num_filters2 = 48         # There are 48 of these filters.
        
        # Convolutional Layer 3
        filter_size3 = 11          # Convolution filters are 11 x 11 
        num_filters3 = 96         # There are 96 of these filters.
        
        # Fully-connected layer
        fc_size = 96 
        
        # layer_conv1 = new_conv_layer
        layer_conv1 = new_conv_layer(input=self.x_image,
                                     num_input_channels=self.num_channels,
                                     filter_size=filter_size1,
                                     num_filters=num_filters1)
        #Max Pool Layer
        ksize1 = [1,4,4,1] # ??????????????????????/
        strides1 = [1,2,2,1]
        layer_max_pool1 = max_pool(layer_conv1,ksize1,strides1)
        
        #Convolutional Layer 2
        layer_conv2 = new_conv_layer(input=layer_max_pool1,
                                     num_input_channels=num_filters1,
                                     filter_size=filter_size2,
                                     num_filters=num_filters2)
        #Max Pool Layer
        ksize2 = [1,2,2,1]
        strides2 = [1,1,1,1]
        layer_max_pool2 = max_pool(layer_conv2,ksize2,strides2)
        
        #Convolutional Layer 3
        layer_conv3 = new_conv_layer(input=layer_max_pool2,
                                     num_input_channels=num_filters2,
                                     filter_size=filter_size3,
                                     num_filters=num_filters3)
        
        #Flatten
        layer_flat, num_features = flatten_layer(layer_conv3)
        #Use the Relu function
        #Relu Layer
        layer_relu = tf.nn.relu(layer_flat)


        #Fully-Connected Layer1
        layer_fc1 = new_fc_layer(input=layer_relu,
                                 num_inputs=num_features,
                                 num_outputs=fc_size,
                                 use_relu=True)
        
        #Fully-Connected Layer2
        layer_fc2 = new_fc_layer(input=layer_fc1,
                                 num_inputs=fc_size,
                                 num_outputs=self.num_classes,
                                 use_relu=False)
        
        #Predict the class
        #softmax function
        y_pred = tf.nn.softmax(layer_fc2)
        # argmax: return the max value
        self.y_pred_cls = tf.argmax(y_pred, dimension=1,name="predictions")
    
        #Cost Function
        #tf.nn.softmax_cross_entropy_with_logits()
        #   : computes softmax cross entropy between logits and labels
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=self.y_true)
        #comput the mean value
        cost = tf.reduce_mean(cross_entropy)

        #optimzer: Adam
        #learning rate = 1e-4
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
#===================================== 191121 02:33 ==========================
        #Predict
        #correct_prediction: same of two vlaue
        correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
        #tf.reduce_mean(): compute the mean value
        #tf.cast(): change a tensor to a new shape
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return optimizer, accuracy
        
    '''
      @brief  use the random batch
      @pre    None
      @post   x_batch, y_batch
      @return return the random batch(x_batch, y_batch)
    '''
    def random_batch(self):
        # Number of images in the training-set.
        num_images = len(self.train_images)
        
        # Create a random index.
        idx = np.random.choice(num_images,
                               size=self.train_batch_size,
                               replace=False)
        
        # Use the random index to select random x and y-values.
        x_batch = self.train_images[idx]
        y_batch = self.labels_train[idx]
        
        return x_batch, y_batch
    
    '''
      @brief  print test accuracy
    '''
    def print_test_accuracy(self,sess):
    
        # Number of images in the test-set.
        num_test = len(self.test_images)
    
        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        #fill zero to the "num_test" shape
        cls_pred = np.zeros(shape=num_test, dtype=np.int)
    
        i = 0
    
        #loop when i is smaller than num_test
        while i < num_test:
            # The ending index for the next batch is denoted j.
            j = min(i + self.test_batch_size, num_test)

            # test images
            images = self.test_images[i:j]
            # test labels
            labels = self.labels_test[i:j]
    
            # Create a feed-dictionary with these images and labels.
            feed_dict = {self.x: images,
                 self.y_true: labels,
                 self.keep_prob: 1,
                 self.keep_prob: 1}

            # run the session
            cls_pred[i:j] = sess.run(self.y_pred_cls, feed_dict=feed_dict)
    
            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j
    
        # Create a boolean array whether each image is correctly classified.
        correct = (self.cls_test == cls_pred)

        # Classification accuracy
        # Classification accuracy is the number of correctly classified
        # images divided by the total number of images in the test-set.
        acc = float(correct.sum()) / num_test
    
        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, correct.sum(), num_test))
        
#===================================== 191121 10:14 ==========================

    '''
      @brief  optimize the function
      @pre    number of iterations
      @post   print and save the accuracy and the time
    '''
    def optimize(self, num_iterations):
        # Ensure we update the global variable rather than a local copy.
        global total_iterations
        total_iterations = 0
        # tf.train.Saver API: save and restore the model and parameters
        saver = tf.train.Saver()
        # Start-time used for printing time-usage below.
        start_time = time.time()
        with tf.Session() as sess:
            #global_step_int = tf.train.get_global_step(sess.graph)
            # initialize the global variables
            sess.run(tf.global_variables_initializer())
            
            for i in range(total_iterations,
                           total_iterations + num_iterations):
                
                # Get a batch of training examples.
                # x_batch now holds a batch of images and
                # y_true_batch are the true labels for those images.
                x_batch, y_true_batch = self.random_batch()
                
                # dictionary for training
                feed_dict_train = {self.x: x_batch,
                                   self.y_true: y_true_batch}
                                   #self.keep_prob: 0.5,
                                   #self.keep_prob: 0.5}
    
                #session: class for TensorFlow calculation
                sess.run([self.optimizer], feed_dict=feed_dict_train)
                
                # Print status every 100 iterations.
                if i % 100 == 0:
                    # Calculate the accuracy on the training-set.
                    feed_dict_acc = {self.x: x_batch,
                                     self.y_true: y_true_batch}
                                     #self.keep_prob: 1,
                                     #self.keep_prob: 1}
                    acc = sess.run(self.accuracy, feed_dict=feed_dict_acc)
                    
                    # Message for printing.
                    msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
                    
                    # Print it.
                    print(msg.format(i + 1, acc))
                    
                    # Update the total number of iterations performed.
                    total_iterations += num_iterations
                    
                    # Ending time.
                    end_time = time.time()
                    
                #Calculate the accuracy on the test set every 100 iterations
                if i%100 == 0:
                    self.print_test_accuracy(sess)
                                   
                #Saves every 500 iterations
                if i%500 == 0:
                    #Change this according to your convenience
                    saver.save(sess, os.path.join(self.save_folder,'model')) 
    
            # Difference between start and end-times.
            time_dif = end_time - start_time
            self.print_test_accuracy(sess)
            # Print the time-usage.
            # timedelta: datime instance 차이를 마이크로초 해사동로 나타내는 기간
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            saver.save(sess, os.path.join(self.save_folder,'model_complete'))

'''
  @brief  parse the arguments using argparse
  @return return the parsed argument
'''    
def parse_arguments():
    # create the ArgumentParser object
    parser = argparse.ArgumentParser(description='Training Network')
    # add the argument
    parser.add_argument('--in_dir',dest='in_dir',type=str,default='cracky')
    parser.add_argument('--iter',dest='num_iterations',type=int,default=1500)
    parser.add_argument('--save_folder',dest='save_folder',type=str,default=os.getcwd())
    return parser.parse_args()
            
'''
  @brief  main function
  @pre    argument
  @post   
'''
def main(args):
    args=parse_arguments()
    num_iterations = args.num_iterations
    
    model = Model(args.in_dir, args.save_folder)
    model.optimize(num_iterations)
    
if __name__ == '__main__':
    main(sys.argv)