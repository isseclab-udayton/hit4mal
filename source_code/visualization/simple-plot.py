import requests
import os
import sys
from PIL import Image, ImageDraw
import numpy as np


OUTPUT_DIR = os.path.join(os.getcwd(), 'compare_outputs')

location = lambda x: os.path.join(os.path.join(os.getcwd(), 'tmp'), x)


def get_outputs(sample_file):
    # url = 'http://localhost:5000/layer/max_pooling2d_3/{}'.format(sample_file)
    url = 'http://localhost:5000/predict/{}'.format(sample_file)
    res = requests.get(url)

    url = 'http://localhost:5000/layer/conv2d_1/{}'.format(sample_file)
    return requests.get(url).json()


def build_img(benign, malicious):
    compare_outpus = zip(get_outputs(benign), get_outputs(malicious))
    for index, (b, m) in enumerate(compare_outpus):
        b, m = location(b), location(m)
        list_im = [b, m]
        imgs = [Image.open(i) for i in list_im]
        # pick the image which is the smallest, and resize the others to match
        # it (can be arbitrary image shape here)
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

        # save that beautiful picture
        imgs_comb = Image.fromarray(imgs_comb)

        # width, height = imgs_comb.size
        # draw = ImageDraw.Draw(imgs_comb)
        # draw.line((0,0, height,width), fill=128)
        filename = os.path.join(OUTPUT_DIR, '{}.png'.format(index))
        imgs_comb.save(filename)

        # imgs_comb = PIL.Image.fromarray( imgs_comb)
        # new_im.save(filename)

        print(b, m)


if __name__ == '__main__':
    benign = '003a27a2e8dcfe385e7603eaabde1d60865e765f20b1c6b3a8370d0f9ebbedfe.png'
    malicious = '0063d7dad57ca78c3dce6a2e7d4ff7a47dbbbbaa33f92aef747d8102e055d1aa.png'

    # print(get_outputs(benign))
    build_img(benign, malicious)
