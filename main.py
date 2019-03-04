import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CnnModel import CnnModel
from DenseModel import DenseModel
import helper as helper
from sklearn.model_selection import train_test_split
import json

### hyperparameters ###
config = {'model': 'dense',  # cnn or dense
          'n_hidden': 20, # for hidden
          'filters': 32,  # for cnn
          'kernel_size': [5,5], #for cnn
          'n_classes': 10,
          'learning_rate': 0.001,
          'batch_size': 32,
          'n_steps': 1000,
          'json_file': 'test.json'}

# create dataframes
train_df = pd.read_csv('data/fashion-mnist_train.csv',sep=',')
test_df = pd.read_csv('data/fashion-mnist_test.csv', sep = ',')

print("Train size: {}".format(len(train_df)))
print("Test size: {}".format(len(test_df)))

# first col: label,
# 784 cols: pixel 28*28
print(test_df.head())

# numpy arrays for tensorflow
train_data = np.array(train_df, dtype = 'float32')
test_data = np.array(test_df, dtype = 'float32')



### Preprocessing ###

train_labels = np.array(train_data[:,0], dtype= 'int8')
test_labels = np.array(test_data[:,0], dtype= 'int8')

# Apply normalization and visualize
train_data = helper.normalize_data(train_data[:,1:])
test_data = helper.normalize_data(test_data[:,1:])

# split training data in training and validation set
x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=5000, random_state=42)

# explore the dataset
helper.plot_image_index(101, train_data, train_labels)
helper.expore(train_data, train_labels)


### Define the Model ###
if config['model'] == 'cnn':
    model = CnnModel(config)
else:
    model = DenseModel(config)
model.build_model()



# iterator to get batch of training examples for training
iterator = helper.create_tf_Dataset(x_train, y_train, config)
next_element = iterator.get_next()

# start tensorflow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run([init, iterator.initializer])


### Train the model ####

eval_dict = {'config': config, 'training_loss': [], 'training_acc': [], 'validation_loss': [],
             'validation_acc': [], 'steps': [], 'final_score': []}

for step in range(config['n_steps']):
    try:
        batch_x, batch_y = sess.run(next_element)
    except tf.errors.OutOfRangeError:
        print("End of dataset")
        break

    sess.run(model.train, feed_dict={model.x: batch_x, model.y: batch_y})

    if step % 50 == 0:
        acc_value_train = sess.run([model.accuracy, model.loss], feed_dict={model.x: batch_x, model.y: batch_y})
        acc_value_val = sess.run([model.accuracy, model.loss], feed_dict={model.x: x_val, model.y: y_val})
        print("Validation accuracy on the {}th step: {}".format(step, acc_value_val))
        # save statistics. item() necessary for json
        eval_dict['training_loss'].append(acc_value_train[1].item())
        eval_dict['training_acc'].append(acc_value_train[0].item())
        eval_dict['validation_loss'].append(acc_value_val[1].item())
        eval_dict['validation_acc'].append(acc_value_val[0].item())
        eval_dict['steps'].append(step)

# now evaluate
acc, loss = sess.run([model.accuracy, model.loss],
                                    feed_dict={model.x: test_data, model.y: test_labels})
eval_dict['final_score'] = (acc.item(), loss.item())
print("Final test accuracy: {}".format((acc, loss)))

# save results
with open('experiments/'+ config['json_file'], 'w') as fp:
    json.dump(eval_dict, fp)


