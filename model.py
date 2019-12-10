from keras.models import *
from keras.layers import *
from keras.optimizers import *
import argparse
import numpy as np
from skimage import io
from datetime import datetime
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import backend as K
# import tensorflow.python.keras.backend as K
import keras
import time
from keras.utils.generic_utils import get_custom_objects
# from tensorflow.keras.optimizers import Adam


logdir = "logs/scalar/"
generator_logdir = logdir +"generator/" + datetime.now().strftime("%m/%d-%H:%M")
discriminator_logdir = logdir+"discriminator/" + datetime.now().strftime("%m/%d-%H%:%M")
generator_callback = keras.callbacks.TensorBoard(log_dir=generator_logdir, write_images=True, write_graph=True)
discriminator_callback = keras.callbacks.TensorBoard(log_dir=discriminator_logdir, write_images=True, write_graph=True)

# https://gist.github.com/joelthchao/ef6caa586b647c3c032a4f84d52e3a11
def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

# https://gist.github.com/erenon/91f526302cd8e9d21b73f24c0f9c4bb8
def named_logs(model, logs):
  result = {}
  for l in zip(model.metrics_names, logs):
    result[l[0]] = l[1]
  return result


def custom_activation(x):
    return 1.0 * x / 2.0
get_custom_objects().update({'custom_activation': Activation(custom_activation)})


# https://stackoverflow.com/questions/51140950/how-to-obtain-the-gradients-in-keras
def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    print("grads")
    print(grads)
    print(len(grads))
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    print("symb_inputs")
    print(symb_inputs)
    print(len(symb_inputs))
    print(symb_inputs[0])
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad

def get_layer_output_grad(model, inputs, outputs, layer=-1):
    """ Gets gradient a layer output for given inputs and outputs"""
    grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


class GAN():
    def __init__(self, args):
        print("initialization---")
        self.img_rows = args.img_rows
        self.img_cols = args.img_columns
        self.channels = args.num_channel
        self.img_shape = (self.img_rows, self.img_cols, self.channels)


        # config = tf.compat.v1.ConfigProto()
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.999
        # session = tf.compat.v1.Session(config=config)
        # tf.compat.v1.keras.backend.set_session(session)


        optimizer_g = Adam(0.0001, 0.5)
        optimizer_d = Adam(0.0001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer_d,
            metrics=['accuracy'])

        discriminator_callback.set_model(self.discriminator)

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer_g)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_g, metrics=['accuracy'])

        generator_callback.set_model(self.combined)

    # def custom_activation(x):
    #     return 1.0 * x / 2.0
    #
    # get_custom_objects().update({'custom_activation': Activation(custom_activation)})


    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

        ###################
        model.add(Dense(128 * 16 * 16, activation='relu', input_shape=noise_shape))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Reshape((16, 16, 128)))

        model.add(Conv2D(128, kernel_size=5, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        # hid = Dropout(0.5)(hid)
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2DTranspose(128, 4, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(128, kernel_size=5, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        # hid = Dropout(0.5)(hid)
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(128, kernel_size=5, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(3, kernel_size=5, strides=1, padding="same"))
        model.add(Activation("tanh"))
        model.add(Activation(custom_activation, name='SpecialActivation'))

        ###################

        # model.add(Dense(256 * 4 * 4, input_shape=noise_shape))
        # model.add(Reshape((4, 4, 256)))
        #
        # model.add(Conv2DTranspose(128, kernel_size=8, strides=2, padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        # model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        # # model.add(Conv2DTranspose(256, kernel_size=3, padding='same'))
        # # model.add(LeakyReLU(alpha=0.2))
        # # model.add(BatchNormalization(momentum=0.8))
        # #
        # # model.add(Conv2DTranspose(128, kernel_size=3, padding='same'))
        # # model.add(LeakyReLU(alpha=0.2))
        # # model.add(BatchNormalization(momentum=0.8))
        # #
        # # model.add(Conv2DTranspose(64, kernel_size=3, padding='same'))
        # # model.add(LeakyReLU(alpha=0.2))
        # # model.add(BatchNormalization(momentum=0.8))
        #
        #
        # model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        # # model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        # # print(self.img_shape)
        # # model.summary()
        #
        # model.add(Flatten())
        # model.add(Activation('sigmoid'))
        # # model.add(Dense((2048), activation='tanh'))
        # # model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        # # print("--")
        # # print(self.img_shape)
        # # for layer in model.layers:
        # #     print(layer.output_shape)
        # # print("")
        # model.add(Reshape(self.img_shape))


        ###################

        # model.add(Dense(256, input_shape=noise_shape))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(np.prod(self.img_shape), activation='sigmoid'))
        # model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)


    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()


        #######################

        model.add(Conv2D(128, kernel_size=3, strides=1, padding='same', input_shape=img_shape))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(64, kernel_size=4, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(32, kernel_size=4, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.1))

        # model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
        # model.add(BatchNormalization(momentum=0.9))
        # model.add(LeakyReLU(alpha=0.1))

        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        ##################

        # model.add(Conv2D(2, (8, 8), padding='same', input_shape=img_shape))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(4, (4, 4), padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # # model.add(Conv2D(128, (4, 4), padding='same'))
        # # model.add(LeakyReLU(alpha=0.2))
        # # model.add(Conv2D(256, (4, 4), padding='same'))
        # # model.add(LeakyReLU(alpha=0.2))
        # # model.add(Conv2D(512, (4, 4), padding='same'))
        # # model.add(LeakyReLU(alpha=0.1))
        #
        # # model.add(Conv2D(512, (4, 4), padding='same'))
        # # model.add(LeakyReLU(alpha=0.1))
        # model.add(Flatten())
        # # model.add(Dense(1024, activation='relu'))
        #
        # # # with tf.device('/gpu:2'):
        # # model.add(Dense(64))
        # # model.add(LeakyReLU(alpha=0.2))
        # # model.add(Dense(64))
        # # model.add(LeakyReLU(alpha=0.2))
        # # model.add(Dense(16))
        # # model.add(LeakyReLU(alpha=0.2))
        # # model.add(Dense(16))
        # # model.add(LeakyReLU(alpha=0.2))
        # # model.add(LeakyReLU(alpha=0.2))
        # # model.add(Dense(16))
        # # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(1, activation='sigmoid'))


        ###########################

        # model.add(Flatten(input_shape=img_shape))
        # model.add(Dense(256))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(128))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(1, activation='sigmoid'))



        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)


    def train(self, epochs, batch_size=128, save_interval=50):

        generator_gradient = []
        discriminator_gradient_real = []
        discriminator_gradient_fake = []
        epochs_index = []


        start = time.time()
        print("Beginning training")

        # figure out directory to save images to
        now = datetime.now()
        save_dir = "out/%d-%d %d:%d" % (now.month, now.day, now.hour, now.minute)
        if not os.path.exists('out'):
            os.mkdir('out', 0o777)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir, 0o777)

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        (X_train, _) = buildData(args.image_dir, args.img_rows, args.img_columns, self.channels)

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # if epoch > 3000:
            # Train the discriminator
            # if epoch < 30:
            #     d_loss_real = self.discriminator.train_on_batch(imgs, np.zeros((half_batch, 1)))
            #     d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.ones((half_batch, 1)))
            # else:
            # print(np.zeros((half_batch, 1)).shape)
            # print((np.random.random_sample((half_batch,))*0.1).shape)
            d_real_labels = np.zeros((half_batch, 1))+np.random.random_sample((half_batch,1))*0.1
            d_fake_labels = np.ones((half_batch, 1))-np.random.random_sample((half_batch,1))*0.1

            if epoch % 4 == 0:
                d_loss_real = self.discriminator.train_on_batch(imgs, d_real_labels)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, d_fake_labels)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            if epoch % 1 == 0:
                # print("--------loss--------")
                # print("     real loss:")
                # print("         %f" % d_loss_real[0])
                # print("         acc: %f" % d_loss_real[1])
                # print("     fake loss:")
                # print("         %f" % d_loss_fake[0])
                # print("         acc: %f" % d_loss_fake[1])
                # print(d_loss)
                # print(d_loss[1])
                # print("")
                print(
                            "                                                     Real: loss: %f, acc: %f   Fake: loss: %f, acc: %f" % (
                    d_loss_real[0], d_loss_real[1], d_loss_fake[0], d_loss_fake[1]))

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            # if epoch < 30:
            #     valid_y = np.array([0] * batch_size)
            # else:
            valid_y = np.array([0] * batch_size)

            # Train the generator
            # with tf.device('/gpu:3'):
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # if epoch > 3000:
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))
            # else:
            #     print ("%d  [G loss: %f]" % (epoch, g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                print("Currently at %.4f minutes" % (time.time() - start / 60))
                self.save_imgs(save_dir, epoch)

            if epoch % 20 == 0:
                # print("--------")
                # print(type(d_loss))
                # print(len(d_loss))
                # print(type(g_loss))
                # print(noise.shape)
                generator_callback.on_epoch_end(epoch, named_logs(self.generator, g_loss))
                # if epoch > 3000:
                discriminator_callback.on_epoch_end(epoch, named_logs(self.discriminator, d_loss))

            # record gradient
            # if epoch % 1 == 0:
            #     generator_weight_grads = get_weight_grad(self.combined, noise, valid_y)
            #     discriminator_weight_grads_real = get_weight_grad(self.discriminator, imgs, d_real_labels)
            #     discriminator_weight_grads_fake = get_weight_grad(self.discriminator, gen_imgs, d_fake_labels)
            #     print("-----GRADIENTS")
            #     print(generator_weight_grads)
            #     print("")
            #     generator_gradient.append(generator_weight_grads)
            #     discriminator_gradient_real.append(discriminator_weight_grads_real)
            #     discriminator_gradient_fake.append(discriminator_weight_grads_fake)
            #     epochs_index.append(epoch)


        print("training took: %.4f minutes" % (time.time() - start / 60))


    def save_imgs(self, save_dir, epoch):
        num_image_samples = 10
        noise = np.random.normal(0, 1, (num_image_samples, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        # print("images??????? %d" % epoch)
        # print(gen_imgs)
        # print("")

        for i in range(num_image_samples):
            img = gen_imgs[i, :, :, :]
            # print("----- %d: %d" % (epoch, i))
            # print(img)
            # print("")
            plt.imsave(save_dir + "/_%d-%d.png" % (epoch, i), img)

        # r, c = 5, 5
        # noise = np.random.normal(0, 1, (r * c, 100))
        # gen_imgs = self.generator.predict(noise)
        #
        # # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5
        #
        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i, j].imshow(gen_imgs[cnt, :, :, 0])
        #         axs[i,j].axis('off')
        #         cnt += 1
        # fig.savefig(save_dir + "/_%d.png" % epoch)
        # plt.close()


def buildData(dir_path, img_rows, img_columns, num_channels):
    """
    Pre-process the images in the directory. Arrange them in arrays to be fed into the model.
    :param dir_path:
    :return:
    """
    images = io.imread_collection(dir_path + "/*.jpg")
    n = len(images.files)
    print(n)
    print("--")
    train_data = np.zeros([n, img_rows, img_columns, num_channels], dtype=float)
    label = np.zeros(n, dtype=float)
    k = 0
    for image_file in images.files:
        # print(image_file)
        a = io.imread(image_file)
        # print(a.shape)
        # print("")
        if (len(a.shape) == 3):
            train_data[k,:,:] = a
            label[k] = 1
            k+=1
    return (train_data, label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument('--image-dir', action='store', type=str, metavar='N',
                        help='The the image directory')
    parser.add_argument('--num-channel', action='store', type=int, metavar='N',
                        help='Number of channels in convolutional layers')
    parser.add_argument('--img-rows', action='store', type=int, metavar='N',
                        help='Number of rows of pixels')
    parser.add_argument('--img-columns', action='store', type=int, metavar='N',
                        help='Number of columns of pixels')
    parser.add_argument('--batch-size', action='store', type=int, metavar='N',
                        help='Batch size')
    #parser.add_argument('--num-train', action='store', type=int, metavar='N',
    #                    help='Number of training data')
    # parser.add_argument('--num-test', action='store', type=int, metavar='N',
    #                     help='Number of testing data')
    args = parser.parse_args()

    # with tf.device(['/gpu:0','/gpu:1','/gpu:2','/gpu:3','/gpu:4','/gpu:5','/gpu:6','/gpu:7']):
    with tf.device('/gpu:0'):
        gan = GAN(args)
        gan.train(epochs=10000, batch_size=args.batch_size, save_interval=20)
