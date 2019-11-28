from keras.models import *
from keras.layers import *
from keras.optimizers import *
import argparse
import numpy as np
from skimage import io
from datetime import datetime
import os
from matplotlib import pyplot as plt
import keras


logdir = "logs/scalar/"
generator_logdir = logdir +"generator/" + datetime.now().strftime("%m/%d-%H:%M")
discriminator_logdir = logdir+"discriminator/" + datetime.now().strftime("%m/%d-%H%:%M")
generator_callback = keras.callbacks.TensorBoard(log_dir=generator_logdir, write_images=True)
discriminator_callback = keras.callbacks.TensorBoard(log_dir=discriminator_logdir, write_images=True)

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


class GAN():
    def __init__(self, args):
        self.img_rows = args.img_rows
        self.img_cols = args.img_columns
        self.channels = args.num_channel
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['mse', 'accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        discriminator_callback.set_model(self.discriminator)


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
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['mse', 'accuracy'])

        generator_callback.set_model(self.combined)


    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

        # model.add(Dense(2048 * 4 * 4, input_dim=noise_shape))
        # model.add(Reshape((4, 4, 2048)))
        #
        # model.add(Conv2DTranspose(1024, kernel_size=3, strides=2, padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        # model.add(Conv2DTranspose(512, kernel_size=3, strides=2, padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        # model.add(Conv2DTranspose(256, kernel_size=3, strides=2, padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        # model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        # model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        #
        # model.add(Conv2DTranspose(3, kernel_size=3, strides=2, padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        #
        # model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        # model.add(Reshape(self.img_shape))

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)


    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        # model.add(Conv2D(32, (4, 4), padding='same', input_shape=img_shape))
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(Conv2D(64, (4, 4), padding='same'))
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(Conv2D(128, (4, 4), padding='same'))
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(Conv2D(256, (4, 4), padding='same'))
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(Conv2D(512, (4, 4), padding='same'))
        # model.add(LeakyReLU(alpha=0.1))
        # # model.add(Conv2D(512, (4, 4), padding='same'))
        # # model.add(LeakyReLU(alpha=0.1))
        # model.add(Flatten())
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(512, activation='relu'))
        # model.add(Dense(1, activation='relu'))

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)


    def train(self, epochs, batch_size=128, save_interval=50, log_interval=10):

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

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(save_dir, epoch)

            if epoch % log_interval  == 0:
                generator_callback.on_epoch_end(epoch, named_logs(self.generator, g_loss))
                discriminator_callback.on_epoch_end(epoch, named_logs(self.discriminator, d_loss))


    def save_imgs(self, save_dir, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                img = gen_imgs[cnt, :, :, :]
                axs[i, j].imshow(img, cmap='viridis')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(save_dir + "/_%d.png" % epoch)
        plt.close()


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
        a = io.imread(image_file)
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
    args = parser.parse_args()

    gan = GAN(args)
    gan.train(epochs=1000, batch_size=args.batch_size, save_interval=10)
