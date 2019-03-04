import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def normalize_data(images):
    images = images.transpose()
    # Subtract mean of each image from its individual values
    mean = images.mean(axis=0)
    images = images - mean

    # Truncate to +/- 3 standard deviations and scale to -1 and +1
    pstd = 3 * images.std()
    images = np.maximum(np.minimum(images, pstd), -pstd) / pstd
    # Rescale from [-1,+1] to [0,1]
    images = (1 + images)/2.

    return images.transpose()

def create_tf_Dataset(x_train, y_train, config):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(buffer_size=10000) # make random
    dataset = dataset.batch(batch_size=config['batch_size']) # batch size for training the model
    dataset = dataset.repeat(1+int(config['n_steps']/(50000/config['batch_size'])))
    iterator = dataset.make_initializable_iterator()
    return iterator

def plot_many(images, labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i,:].reshape((28,28)), cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])

def plot_image_index(index, train_data, train_labels):
    print ("Showing digit in train set with index : " + str(index))

    image = train_data[index, :].reshape((28, 28))
    plt.imshow(image)
    print ("y=" + str(class_names[train_labels[index]]))

    plt.show()

def expore(train_data, train_labels):
    print("plot histogram of train labels")
    plt.hist(train_labels, rwidth=0.5)
    plt.show()
    plot_many(train_data, train_labels)
    plt.show()