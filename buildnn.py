import tensorflow as tf
def neural_net_image_input(image_shape):
	return tf.placeholder(tf.float32, shape = [None, image_shape[0], image_shape[1], image_shape[2]], name = 'x')

def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, shape = [None, n_classes], name = 'y')

def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name = 'keep_prob')
	
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
	weight = tf.Variable(tf.truncated_normal([conv_ksize[0],\
                                            conv_ksize[1],\
                                            x_tensor.get_shape().as_list()[-1],\
                                            conv_num_outputs],stddev = 0.1))    
    print(x_tensor.get_shape().as_list())
    bias = tf.Variable(tf.zeros(conv_num_outputs, dtype = tf.float32))
    conv_layer = tf.nn.conv2d(x_tensor, weight, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    conv_layer = tf.nn.relu(conv_layer)
    conv_layer = tf.nn.max_pool(conv_layer, ksize = [1,*pool_ksize, 1], strides = [1, *pool_strides, 1], padding = 'SAME')
    return conv_layer
	
def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    flattened_size = x_tensor.shape[1]*x_tensor.shape[2]*x_tensor.shape[3]
    return tf.reshape(x_tensor, [-1, flattened_size.value])
	
def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    num_features = x_tensor.shape[1].value
    weights = tf.Variable(tf.random_normal([num_features, num_outputs], stddev=0.1))
    biases = tf.Variable(tf.zeros([num_outputs]))
    fc = tf.add(tf.matmul(x_tensor, weights), biases)
    fc = tf.nn.relu(fc)
    return fc
	
def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
	"""
	weights = tf.Variable(tf.random_normal([x_tensor.shape[1].value, num_outputs]))
    bias = tf.Variable(tf.zeros([num_outputs]))
    out_layer = tf.add(tf.matmul(x_tensor, weights), bias)
    return out_layer
	
def conv_net(x, keep_prob):
	x = conv2d_maxpool(x, 8, (3, 3), (1, 1), (2, 2), (2, 2))
    x = conv2d_maxpool(x, 16, (3, 3), (1, 1), (2, 2), (2, 2))
    x = conv2d_maxpool(x, 32, (3, 3), (1, 1), (2, 2), (2, 2))
	x = flatten(x)
	x = fully_conn(x, 1024)
    x = tf.nn.dropout(x, keep_prob=keep_prob)
    x = fully_conn(x, 1024)
    x = tf.nn.dropout(x, keep_prob=keep_prob)
	x = output(x, 10)
	return x

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    session.run(optimizer, feed_dict = {x:feature_batch, y: label_batch, keep_prob: keep_probability})
	
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    loss = session.run(cost, feed_dict={x:feature_batch, 
                                        y:label_batch,
                                        keep_prob:1.0}) 
    acc = session.run(accuracy, 
                feed_dict={x:valid_features, 
                           y:valid_labels, 
                           keep_prob:1.0})
    print('Loss={0} ValidationAccuracy={1}'.format(loss, acc))

epochs = 40
batch_size = 2048
keep_probability = 0.9

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)
	
def main():
	print('Checking the Training on a Single Batch...')
	save_model_path = './image_classification'
	
	print('Training...')
	with tf.Session() as sess:
		# Initializing the variables
		sess.run(tf.global_variables_initializer())
		
		# Training cycle
		for epoch in range(epochs):
			# Loop over all batches
			n_batches = 5
			for batch_i in range(1, n_batches + 1):
				for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
					train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
				print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
				print_stats(sess, batch_features, batch_labels, cost, accuracy)
	saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
			
if __name__ == '__main__':
  main()