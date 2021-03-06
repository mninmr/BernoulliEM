import os
import struct
from array import array
import numpy as np

class MNIST(object):
    def __init__(self, path='./data/'):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                         os.path.join(self.path, self.test_lbl_fname))

        self.test_images = ims
        self.test_labels = labels

        return ims, labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                         os.path.join(self.path, self.train_lbl_fname))

        self.train_images = ims
        self.train_labels = labels

        return ims, labels

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                    'got %d' % magic)

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                    'got %d' % magic)

            image_data = array("B", file.read())

        images = []
        for i in xrange(size):
            images.append([0]*rows*cols)

        for i in xrange(size):
            images[i][:] = image_data[i*rows*cols : (i+1)*rows*cols]

        return images, labels

    def test(self):
        test_img, test_label = self.load_testing()
        train_img, train_label = self.load_training()
        assert len(test_img) == len(test_label)
        assert len(test_img) == 10000
        assert len(train_img) == len(train_label)
        assert len(train_img) == 60000
        print 'Showing num:%d' % train_label[2]
        #print self.display(train_img[0])
        self.displayPlot(train_img[2])
        return True

    def conv2binary(self, img):
        for ii in range(len(img)):
            if img[ii] > 200:
                img[ii] = 1
            else:
                img[ii] = 0
        return img

    def write2file(self, imgs):
        with open('./data/train', 'w+') as ff:
            for img in imgs[0:99]:
                ff.write(str(self.conv2binary(img)))
                ff.write('\n')

    @classmethod
    def display(cls, img, width=28):
        render = ''
        for i in range(len(img)):
            if i % width == 0: render += '\n'
            if img[i] > 200:
                render += '1'
            else:
                render += '0'
        return render

    @classmethod
    def displayPlot(cls, img, width=28):
        import matplotlib.pyplot as plt
        grid = []
        for ii in range(0, width):
            grid.insert(0, img[ii*width:ii*width+width-1])
        input_grid = np.array(grid, np.float)
        plt.pcolor(input_grid, cmap=plt.cm.GnBu)
        plt.yticks(np.arange(0,29))
        plt.xticks(np.arange(0,29))
        plt.show()


if __name__ == "__main__":
    print 'Testing'
    mn = MNIST('./data/')
    if mn.test():
        print 'Passed'
