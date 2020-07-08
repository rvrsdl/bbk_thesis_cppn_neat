import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import matplotlib.animation as animation
from PIL import Image

from genome import Genome
from nnet import NNFF
from netviz import netviz


CHANNELS = 3 # 3 for RGB

def create_image(net, imsize):
    size_x = imsize[0]
    size_y = imsize[1]
    pixels = size_x*size_y
    # Generate the coordinate pairs for inputs (normalised)
    xcoords = np.tile(range(size_x), (size_y,1))
    ycoords = np.tile(np.array([range(size_y)]).transpose(), (1,size_x))
    xinp = (xcoords - np.mean(xcoords)) / np.std(xcoords)
    yinp = (ycoords - np.mean(ycoords)) / np.std(ycoords)
    dinp = np.sqrt(xinp**2 + yinp**2)
    inp_zip = list(zip(xinp.ravel(), yinp.ravel(), dinp.ravel(), np.tile(1, pixels)))
    img_raw = np.array([net.feedforward(inp) for inp in inp_zip])
    if CHANNELS==1:
        img_square = img_raw.reshape(size_x, size_y)
    else:
        img_square = img_raw.reshape(size_x, size_y, CHANNELS)
    return img_square

def create_image2(net, imsize, bias=1):
    """
    This is way faster. Do all the pixels at once rather than looping
    Didn't even need to make any change to the nnet code!
    """
    size_x = imsize[0]
    size_y = imsize[1]
    pixels = size_x*size_y
    # Generate the coordinate pairs for inputs (normalised)
    xcoords = np.tile(range(size_x), (size_y,1))
    ycoords = np.tile(np.array([range(size_y)]).transpose(), (1,size_x))
    xinp = (xcoords - np.mean(xcoords)) / np.std(xcoords)
    yinp = (ycoords - np.mean(ycoords)) / np.std(ycoords)
    dinp = np.sqrt(xinp**2 + yinp**2)
    img_raw = np.array(net.feedforward((xinp.ravel(), yinp.ravel(), dinp.ravel(), np.tile(bias, pixels))))
    if CHANNELS==1:
        img_square = img_raw.T.reshape(size_x, size_y)
    else:
        img_square = img_raw.T.reshape(size_x, size_y, CHANNELS)
    return img_square

def show_image(img):
    if CHANNELS==1:
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(img, vmin=0, vmax=1)
    plt.show()
    
def create_genome():
    G = Genome(4, CHANNELS)
    # Make it more complex
    for i in range(30):
        G.add_node()
        for j in range(i//2):
            G.add_connection()
    G.randomise()
    return G

for i in range(13,25):
    print("Creating image {:d}".format(i))
    G = create_genome()
    net = NNFF(G)
    imsize = 64
    img = create_image2(net, (imsize,imsize))
    #show_image(img)
    stem_name = "./output/img64c_{:d}".format(i)
    imsave(stem_name+".png", img, vmin=0, vmax=1, cmap='gray')
    G.save(stem_name+".json")

def upscale(img_num, imsize=512):
    G = Genome.load("./output/img64c_{:d}.json".format(img_num))
    net = NNFF(G)
    img = create_image2(net, (imsize,imsize))
    show_image(img)
    imsave("./output/img64c_{:d}_to{:d}.png".format(img_num, imsize), img, vmin=0, vmax=1)
    
def animate(net, imsize=128):
    frames = []
    for bias in np.arange(-3,3.5,0.5):
        img_dat = create_image2(net, (imsize,imsize), bias)
        frames.append(Image.fromarray(np.uint8(img_dat*255)))
    loop = []
    loop.extend(frames[1:]) # to go in one direction add all after the first frame
    loop.extend(frames[-2:0:-1]) # then to come back append in reverse direction
    frames[0].save('test.gif', format='GIF', append_images=loop, save_all=True, duration=100, loop=0)
    
    