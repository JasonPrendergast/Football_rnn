import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.contrib import rnn
lemmatizer = WordNetLemmatizer()
train_x,train_y,test_x,test_y = pickle.load(open("football_set.pickle","rb"))

hm_epochs =100
n_classes = 2
batch_size = 1
chunk_size = 168
n_chunks = 1
rnn_size = 128
total_batches = int(len(train_x)/batch_size)

x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

def recurrent_neural_network(x):
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']
    return output


saver = tf.train.Saver()
#saver = tf.train.import_meta_graph('./model.ckpt.meta')
tf_log = 'tf.log'

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        try:
            epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
            print('STARTING:',epoch)
        except:
            epoch = 1
            print('STARTING:',epoch)
        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess,"./model.ckpt")
            epoch_loss = 1
            i=0
            batches_run = 0
            while i < int(total_batches*batch_size):
                batch_x = []
                batch_y = []
                #batches_run = 0
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                if len(batch_x) >= batch_size:
                    
                        batch_x=np.array(batch_x)
                        batch_x = batch_x.reshape((batch_size,n_chunks,chunk_size))
                    
                        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                                  y: np.array(batch_y)})
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run +=1
                        i+=batch_size
                        #print('Batch run:',batches_run,'/',total_batches,'| Epoch:',epoch,'| Batch Loss:',c,)

            i=0
            #print(len(train_x))
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            #print(correct)
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            #print(accuracy)

            #testy_x=[]
            #testy_x=np.array(test_x)
            #test_x=test_x.reshape((batch_size,n_chunks,chunk_size))
            totaltestbatch = int(len(test_x)/batch_size)
            #print(totaltestbatch)
            
            while i < int(totaltestbatch*batch_size):
                testbatch_x = []
                testbatch_y = []
                #print('inthis')
               
                #batches_run = 0
                start = i
                end = i+batch_size
                testbatch_x = np.array(test_x[start:end])
                testbatch_y = np.array(test_y[start:end])
                if len(testbatch_x) >= batch_size:
                    
                        testbatch_x=np.array(testbatch_x)
                        testbatch_x = testbatch_x.reshape((batch_size,n_chunks,chunk_size))
                        i+=batch_size

            print(len(testbatch_x))
            #print(len(testbatch_y))
            #print(len(test_x))

            print('Accuracy:',accuracy.eval({x:testbatch_x, y:np.array(testbatch_y)}))                      

                
            
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            saver.save(sess, "./model.ckpt")
            #print(len(train_x))
            #print()
            #print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            with open(tf_log,'a') as f:
                f.write(str(epoch)+'\n') 
            epoch +=1
            
    

train_neural_network(x)

def test_neural_network():
    prediction = recurrent_neural_network(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            try:
                saver.restore(sess,"./model.ckpt")
            except Exception as e:
                print(str(e))
            epoch_loss = 0
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
      #  feature_sets = []
      #  labels = []
      #  counter = 0
       # with open('processed-test-set.csv', buffering=20000) as f:
       #     for line in f:
        #        try:
        #            features = list(eval(line.split('::')[0]))
       #             label = list(eval(line.split('::')[1]))
       #             feature_sets.append(features)
        #            labels.append(label)
        #            counter += 1
        #        except:
        #            pass
       # print('Tested',counter,'samples.')
       # test_x = np.array(feature_sets)
       # test_y = np.array(labels)
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))
print ('here')
#x = tf.placeholder('float', [None, n_chunks,chunk_size])
#test_neural_network()
