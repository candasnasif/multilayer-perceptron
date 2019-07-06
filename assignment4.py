import sys
import random
import numpy as np
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf

#This function converts a dictionary to list
def dict_to_list(dict):
    list = []
    for x in dict:
        list.append(dict[x])
    return list
#Multilayer perceptron function take x(template of input), weight and biases
def multilayer_perceptron(x, weights, biases):


    # Hidden layer with ReLU activation
    hidden_layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    hidden_layer_1 = tf.nn.relu(hidden_layer_1)
    # Hidden layer with ReLU activation
    hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, weights['h2']), biases['b2'])
    hidden_layer_2 = tf.nn.relu(hidden_layer_2)
    # Output layer with ReLu activation
    output_layer = tf.add(tf.matmul(hidden_layer_2, weights['out']), biases['out'])
    output_layer = tf.nn.relu(output_layer)
    #This line apply linear activation for get higher accuracy
    #output_layer = tf.matmul(hidden_layer_2, weights['out']) + biases['out']
    return output_layer

#This function Calculate vectors for input layer and return them into a dictionary
def input_datas(vectors, train):
    input_data = dict()
    for line in train:
        sum = np.zeros(shape=(1,200),dtype=float)
        words = re.split(' +',line)
        for word in words:
            if word in vectors:
                y= np.array(vectors[word],dtype=float)
                #Addition operation between vectors of words into the sentence with numpy array
                sum = np.add(sum,y)

        input_data[line] = sum


    return input_data

#Read positive sentences from file
fo = open(str(sys.argv[1]), encoding='utf-8')
pos = fo.readlines()
fo.close()
#Read negative sentences from file
fo = open(str(sys.argv[2]), encoding='utf-8')
neg = fo.readlines()
fo.close()
#Read word2vec vectors from vectors file
fo = open(str(sys.argv[3]), encoding='utf-8')
vector_file = fo.readlines()
fo.close()

pos_labels = []
neg_labels = []
#Assign [1,0] positive labels
for i in range(len(pos)):
    pos_labels.insert(i,[[1,0]])
#Assign [0,1] negative labels
for i in range(len(neg)):
    neg_labels.insert(i,[[0,1]])
#combine all sentences
sentences = pos+neg
#combine all labels
labels = pos_labels+neg_labels
#sharing list for same suffle order
c = list(zip(sentences, labels))
#shuffle in same order
random.shuffle(c)
sentences, labels = zip(*c)
#size of training set
train_size = (len(sentences)*int(str(sys.argv[4])))/100
#training sentences
train = sentences[:int(train_size)]
#labels of training sentences
train_label = labels[:int(train_size)]
#test sentences
test = sentences[int(train_size):]
#label of test sentences
test_label = labels[int(train_size):]
#vectors dictionary vectors of words key = word value = vector
vectors = dict()
for line in vector_file:
    str = re.split(':| +',line)
    vectors[str[0]] = str[1:]

#input datas coming from input_datas function
input_data = input_datas(vectors = vectors,train = train)
#convert input_data dictionary to list
input_list = dict_to_list(dict = input_data)
#test_datas coming from input_datas function
test_data = input_datas(vectors = vectors, train = test)
#convert test_data dictionary to list
test_list = dict_to_list(dict = test_data)
#convert input labels list to numpy array
np_input_label = np.array(train_label)
#convert input vectors list to numpy array
np_input = np.array(input_list)
#convert test vectors list to numpy array
np_test = np.array(test_list)
#convert test labels vector list to numpy array
np_test_label = np.array(test_label)


learning_rate = 0.001
n_hidden_1 = 100 # 1st hidden layer number of neurons
n_hidden_2 = 100 # 2nd hidden layer number of neurons
n_input = 200   #dimension of vectors
n_classes = 2   # positive and negative classes
training_epochs = 15
display_step = 1

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
# Store layers weight
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
# Construct model
prediction = multilayer_perceptron(x, weights, biases)

#cost calculation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
#optimizer model for cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

fo = open('output.txt', 'w')

#initialize tf.session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        tempEpoch = epoch
        avg_cost = 0.

        # Loop over all inputs
        for i in range(len(np_input)):
            #training input vectors and their labels
            batch_xs = np_input[i]
            batch_ys = np_input_label[i]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,y: batch_ys})

            # Compute average loss
            avg_cost += c / len(np_input)
       # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", avg_cost)
            fo.write("Epoch: 000{} cost= {}\n".format((epoch+1),avg_cost))
    print("Optimization Finished!")
    fo.write("Optimization Finished!\n")
    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction, 1,), tf.argmax(y, 1))

    #Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    acc = 0.0
    for i in range(len(np_test)):
        acc = acc + accuracy.eval({x: np_test[i], y: np_test_label[i]})
    fo.write("Accuracy: {}\n".format((acc / len(np_test))))
    print("Accuracy:", acc / len(np_test))
fo.close()