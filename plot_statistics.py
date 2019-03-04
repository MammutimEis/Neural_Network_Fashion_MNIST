import json
import matplotlib.pyplot as plt

plt.close("all")
def plot_single_file(path):
    '''
    plots the statistics of a single json test file.
    :param path: path to file
    '''
    with open(path, 'r') as f:
        data = json.load(f)

    steps = data['steps']
    training_loss = data['training_loss']
    validation_loss = data['validation_loss']
    training_acc = [i*100 for i in data['training_acc']]
    validation_acc = [i*100 for i in data['validation_acc']]

    plt.plot(steps, training_loss, c='b', label="train loss")
    plt.plot(steps, validation_loss, c='g', label="val loss")
    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    plt.plot(steps, training_acc, c='b', label="train acc")
    plt.plot(steps, validation_acc,c='g', label="val acc")
    plt.xlabel("training steps")
    plt.ylabel("accuracy in %")
    plt.yticks([i*10 for i in range(11)])
    plt.legend()
    plt.show()

def plot_multiple_files(path):
    '''Plots all files in the directory. filenames must be numeric because they get sorted!
    :param path: directory of test files
    '''
    from os import listdir
    from os.path import isfile, join
    plot_files = [f for f in listdir(path) if isfile(join(path, f))]
    final_acc = []
    final_loss = []
    nodes = []

    sorted_files = sorted(plot_files, key=lambda x: float(x))
    print(sorted_files)

    for file in sorted_files:
        with open(path+str(file), 'r') as f:
            data = json.load(f)
        steps = data['steps']
        training_loss = data['training_loss']
        validation_loss = data['validation_loss']
        training_acc = [i * 100 for i in data['training_acc']]
        validation_acc = [i * 100 for i in data['validation_acc']]
        final_acc.append(data['validation_acc'][-1]*100)
        final_loss.append(data['validation_loss'][-1])
        nodes.append(data['config']['n_hidden'])


        plt.figure(1, figsize=(10, 6), dpi=150)#, facecolor='w', edgecolor='k')
        #plt.plot(steps, training_loss, c='b', label="train loss")
        plt.semilogy(steps, validation_loss, label=str(file))
        plt.xlabel("training steps")
        plt.ylabel("loss")
        plt.legend()

        plt.figure(2, figsize=(10, 6), dpi=150)
        #plt.plot(steps, training_acc, c='b', label="train acc")
        plt.plot(steps, validation_acc, label=str(file))
        plt.xlabel("training steps")
        plt.ylabel("accuracy in %")
        plt.yticks([i * 10 for i in range(11)])
        plt.legend()

    plt.figure(3, figsize=(10, 6), dpi=150)
    plt.plot(nodes, final_acc, 'o')
    plt.xlabel("number of nodes")
    plt.yticks([i * 10 for i in range(11)])
    plt.ylabel("Final accuracy on validation set")

    plt.show()

plot_single_file("experiments/test.json")
#plot_multiple_files("experiments/")