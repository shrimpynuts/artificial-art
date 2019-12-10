import pickle
from matplotlib import pyplot as plt


def parse_outputs(path):
    x_batch = []
    g_loss = []
    d_loss = []
    accuracy = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'G loss' in line and 'D loss' in line:
                tokens = line.split(' ')
                idx = tokens[0]
                tokens2 = line.split(': ')
                d_l = tokens2[1].split(',')[0]
                acc = tokens2[2].split(']')[0]
                g_l = tokens2[3].split(']')[0]
                x_batch.append(idx)
                g_loss.append(g_l)
                d_loss.append(d_l)
                accuracy.append(acc)
    parsed_file = open('parsed_outputs', 'wb')
    dic = {}
    dic["x_batch"] = x_batch
    dic["d_loss"] = d_loss
    dic["g_loss"] = g_loss
    dic["accuracy"] = accuracy
    pickle.dump(dic, parsed_file)

    print(x_batch)
    print(d_loss)
    print(g_loss)
    print(accuracy)


def plot_losses(file):
    loss_file = open(file, 'rb')
    dic = pickle.load(loss_file)
    d_loss = dic["d_loss"]
    g_loss = dic["g_loss"]
    idx = dic["x_batch"]
    d_loss = d_loss[0:10000]
    g_loss = g_loss[0:10000]
    idx = idx[0:10000]

    print("Plotting losses...hold on...")
    plt.plot(idx, d_loss, label="discriminator loss")
    plt.plot(idx, g_loss, label="generator loss")
    plt.title("Generator and discriminator losses")
    plt.savefig('loss_fig')


if __name__ == '__main__':
    parse_outputs('outputs.txt')
    plot_losses('parsed_outputs')
    print("Done")