#Start of the DeepLearning
### Train your model here.
### Feel free to use as many code cells as needed.

from tensorflow.contrib.layers import flatten
import tensorflow as tf

def Preprocess(images):
    #Convert to Grayscale
    images_gray = np.array([cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in images])
    #Normalize Gray
    images_gray_norm = (images_gray- 127.5)/255.0
    #Convert to Tensor
    images_tensor = np.expand_dims(images_gray_norm,axis=3)
    return images_tensor

#Modify input and output layers as input is color, output has 43 labels
def LeNet(x,convdkp,fcdkp):    
    # Hyperparameters
    mu = 0
    sigma = 0.02
    relu_shift = 0.01
    
    L1_filtnum = 32
    L2_filtnum = 64
    
    #fc1_neurnum =
    #fc2_neurnum = 

    #conv0_W = tf.Variable(tf.truncated_normal(shape=(1,1, 3, 16 ),mean = mu, stddev = sigma))
    #conv0_b = tf.Variable(tf.zeros(3))

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 32x32x16.
    #decreasing initial layer filter size
    conv1_W = tf.Variable(tf.truncated_normal(shape=(3,3, 1, L1_filtnum ),\
                                              mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(L1_filtnum))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1 + relu_shift)
    conv1 = tf.nn.dropout(x=conv1, keep_prob=convdkp)

    # SOLUTION: Pooling. Input = 32x32x16. Output = 16x16x16.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 16x16x64.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(7, 7, L1_filtnum, L2_filtnum), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(L2_filtnum))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2 +relu_shift)
    conv2 = tf.nn.dropout(x=conv2, keep_prob=convdkp)

    # SOLUTION: Pooling. Input = 16x16x64 Output = 8x8x64.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # SOLUTION: Flatten. Input = 8x8x64. Output = 4096.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 4096 Output = 1024.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(4096, 1024), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(1024))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1+relu_shift) #Adding small bias as per CS231n
    
    #add dropout
    fc1 = tf.nn.dropout(x=fc1, keep_prob=fcdkp)

    # SOLUTION: Layer 4: Fully Connected. Input = 4096. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(1024, 512), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(512))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2 +relu_shift) #Adding small bias as per CS231n  
    #add dropout
    fc2 = tf.nn.dropout(x=fc2, keep_prob=fcdkp)

    # SOLUTION: Layer 5: Fully Connected. Input = 512. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(512, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits#Accuracy Evaluation Function
def evaluate(X_data, y_data,batchSize=100):
    num_examples = len(X_data)
    print("Input Dataset: ",num_examples)
    total_accuracy = 0
    sess = tf.get_default_session()
    
    BATCH_SIZE = batchSize
    
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def evaluate(X_data, y_data,batch_fcdkp=1.0,batch_convdkp=1.0, bn_train = False, batchSize = 100):
    num_examples = len(X_data)
    print("Input Dataset: ",num_examples)
    total_accuracy = 0
    
    BATCH_SIZE = batchSize
    
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, fcdkp:batch_fcdkp,\
                                                           convdkp:batch_convdkp,to_train: bn_train})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def GetEvalMetrics(network_ops, X_data, y_data, batch_fcdkp=1.0,batch_convdkp=1.0,get_labels =False, bn_train = False,batchSize=100):
    num_examples = len(X_data)
    #print("Input Dataset: ",num_examples)
    total_accuracy = 0
    total_loss =0
    
    BATCH_SIZE = batchSize
    
    sess = tf.get_default_session()
    total_results = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        results = sess.run(network_ops,\
                                       feed_dict={x: batch_x, y: batch_y, fcdkp:batch_fcdkp,convdkp:batch_convdkp
                                                 ,to_train: bn_train})
            
        total_results +=results

    return total_results/num_examples

def GetEvalMetricswithLabels(X_data, y_data,batch_fcdkp=1.0,batch_convdkp=1.0,bn_train = False,batchSize=100):
    num_examples = len(X_data)
    #print("Input Dataset: ",num_examples)
    total_accuracy = 0.0
    total_loss =0.0
    total_predicted_labels =[]
    
    BATCH_SIZE = batchSize

    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        mean_loss, accuracy,out_labels = sess.run([loss_value, accuracy_operation, predicted_labels], \
                                                  feed_dict={x: batch_x, y: batch_y, fcdkp:batch_fcdkp,convdkp:batch_convdkp
                                                            ,to_train: bn_train})          
        total_accuracy += (accuracy * len(batch_x))
        total_loss+= (mean_loss*len(batch_x))
        total_predicted_labels.append(out_labels.flatten().tolist())

    return total_accuracy / num_examples, total_loss/ num_examples,total_predicted_labels


def vis_softmax(tf_model,feed_vals,top_nprobs = 3, category_names =None):
    import matplotlib.pyplot as plt
    import pandas as pd
    #Import the model
    if tf_model == None:
        print("Error: Empty Model")
        return
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(tf_model)
        loader.restore(sess, tf.train.latest_checkpoint('.'))
        
    #Get softmax probabilities
        #feed_vals ={x:input_img, fcdkp:1.0,convdkp:1.0,to_train:0}
        score_vals,label_nums = sess.run(tf.nn.top_k(tf.nn.softmax(logits),k=top_nprobs), feed_dict= feed_vals)
        scores = score_vals[0]
        scores_labels = label_nums[0]
    #Plot softmax
        y_pos = list(range(top_nprobs))
        plt.barh(y_pos,scores,align='center')
        plt.ylabel('Categories')
        plt.xlabel('Probabilities')
        if len(category_names)>0:
            label_names = [category_names[i] for i in scores_labels] #Get names of top 5 labels
            plt.yticks(y_pos,label_names)
        #print(scores)
    #Build Pandas Datafrae for output
        #print(scores)
        #print(label_names)
        df = pd.DataFrame(scores, index = label_names, columns =['Probability'])
    return df
import numpy as np
import cv2
def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    
    A Random uniform distribution is used to generate different parameters for transformation
    
    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    shear_M = cv2.getAffineTransform(pts1,pts2)
    
    #Perform Transformations
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    return img
def AugumentImageData(X_data,y_data):
    from collections import defaultdict
    import random
    import numpy as np
    
    label_indices =defaultdict(list)
    #Append label indices into a dict
    for index,label in enumerate(y_data):
        label_indices[label].append(index)

    label_sizes =[len(label_indices[label]) for label in label_indices.keys()]
    max_label_size = max(label_sizes)
    total_images_add = (max_label_size * 43) - sum(label_sizes)
    print("total images to add: ", total_images_add)
    X_augument = []
    y_augument = []
    index =0
    for j,label in enumerate(label_indices.keys()):
        current_label_size = len(label_indices[label])
        num_images_add = max_label_size- current_label_size
        #num_images_add = 1
        for i in range(num_images_add):
            #get random image from label list
            random_index = random.choice(label_indices[label])
            #skew,rotate image
            X_augument.append(transform_image(X_data[random_index], \
                                                        5,5,5))
            y_augument.append(label)
            #append to list
            index +=1
            #print(label)

    X_out = np.concatenate((X_data, X_augument), axis=0)
    y_out = np.concatenate((y_data,y_augument), axis=0)
    print("Number of Images Initially:", len(X_train_raw))
    print("Number of Images Augumented:", len(X_augument))
    print("Total Number of Images finally:", len(X_train_total))
    
    return X_out,y_out