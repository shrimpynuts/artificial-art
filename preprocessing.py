import argparse
from skimage.io import imread_collection

parser = argparse.ArgumentParser()
parser.add_argument("--indir", help="which directory to take images from", default=None)
parser.add_argument("--outdir", help="which directory to output processed images to", default=None)

def preprocessing(indir, outdir):
    print("processing!", indir, outdir)
    x = imread_collection(indir + "/*.jpg")
    

if __name__ == '__main__':
    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir 
    preprocessing(indir, outdir)