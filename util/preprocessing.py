import argparse
from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import img_as_ubyte
import time
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--indir", help="which directory to take images from", default=None, type=str, required=True)
parser.add_argument("--outdir", help="which directory to output processed images to", type=str, default=None, required=True)
parser.add_argument("--width", help="width to scale images to", default=None, type=int, required=True)
parser.add_argument("--height", help="height to scale images to", default=None, type=int, required=True)

def preprocessing(indir, outdir, width, height, plot_dim=False):
    print("Beginning processing!")
    print("Rescaling all images in %s to %dx%d and outputting to %s" % (indir, width, height, outdir))
    start_time = time.time()

    x = io.imread_collection(indir + "/*.jpg")

    # PLOTTING THE DIMENSIONS OF ALL IMAGES
    if plot_dim:
        shapes = []
        for img_file in x.files:
            a = io.imread(img_file)
            if len(a.shape) != 3:
                print(img_file, a.shape)
            shapes.append(a.shape)

        widths = [x[1] for x in shapes]
        heights = [x[0] for x in shapes]
        plt.figure(figsize=(20, 20))
        plt.plot(widths, heights, 'o')

    def scale_image(img, new_width, new_height):
        return resize(img, (new_width, new_height), anti_aliasing=True)

    # MAKE OUT DIRECTORY
    if not os.path.exists(outdir):
        os.mkdir(outdir, 0o777)

    # setup toolbar
    toolbar_width = 40
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['

    # OUTPUT ALL OF THE RESIZED IMAGES
    for i in range(len(x.files)):
        img_file = x.files[i]

        if i % (len(x.files) // toolbar_width) == 0:
            sys.stdout.write("-")
            sys.stdout.flush()

        a = io.imread(img_file)
        if (len(a.shape) == 3):  # filter out gray-scale images
            a_resized = scale_image(a, width, height)
            a_normalized = img_as_ubyte(a_resized)
            out_file = outdir + "/" + img_file.split("/")[2]
            io.imsave(out_file, a_normalized)

    sys.stdout.write("]\n")  # this ends the progress bar

    end_time = time.time()
    print("Processing finished in " + str((end_time - start_time) / 60.0) + " minutes")


if __name__ == '__main__':
    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir 
    width = args.width
    height = args.height
    preprocessing(indir, outdir, width, height)