import numpy,scipy, os, array, scipy.misc
import sys


data_dir = sys.argv[1]
image_dir = '/home/lyvd/FinalDataset/gray_images'
for (dirpath, dirnames, filenames) in os.walk(data_dir):
    for fi in filenames:
        print(fi)
        filename = os.path.join(dirpath, fi)
        print(filename)
        f = open(filename,'rb');
        ln = os.path.getsize(filename); # length of file in bytes
        width = 256;
        rem = ln%width;

        a = array.array("B"); # uint8 array
        a.fromfile(f,ln-rem);
        f.close();

        g = numpy.reshape(a,(len(a)/width,width));
        g = numpy.uint8(g);
        dest = os.path.join(image_dir, fi + '.png')
        scipy.misc.imsave(dest,g); # save the image
