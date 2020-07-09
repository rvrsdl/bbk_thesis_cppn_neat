import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
#import matplotlib.animation as animation
from PIL import Image

from genome import Genome
from nnet import NNFF
from netviz import netviz


CHANNELS = 3 # 3 for RGB

def create_image(net, imsize):
    """
    Creates image by looping through pixel coordinates.
    Obsolete: use create_image2 instead.
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

def do_run(num=10, size=64):
    """
    Do a run of eg. 10 images, saving the images (.png) 
    and the corresponding networks (.json) in the output folder.
    """
    for i in range(num):
        print("Creating image {:d}".format(i))
        G = create_genome()
        net = NNFF(G)
        img = create_image2(net, (size,size))
        #show_image(img)
        stem_name = "./output/e{}".format(get_epoch_str())
        imsave("{}_{:d}.png".format(stem_name, size), img, vmin=0, vmax=1)
        G.save(stem_name+".json")

def upscale_saved(epoch_str, imsize=512):
    """
    Load one of the saved networks and produce a hi res image.
    """
    if type(imsize)==int:
        str_imsize = imsize
        imsize = (imsize, imsize)
    else:
        str_imsize = "{:d}x{:d}".format(imsize[0],imsize[1])
    stem_name = "./output/{}".format(epoch_str)
    G = Genome.load(stem_name+".json")
    net = NNFF(G)
    img = create_image2(net, imsize)
    show_image(img)
    imsave("{}_{}.png".format(stem_name, str_imsize), img, vmin=0, vmax=1)
    
def animate(net, imsize=128, filename='test.gif'):
    """
    By varying the bias input we can alter the image in smooth steps.
    This function makes a GIF of this.
    """
    frames = []
    for bias in np.arange(-3,3.5,0.5):
        img_dat = create_image2(net, (imsize,imsize), bias)
        frames.append(Image.fromarray(np.uint8(img_dat*255)))
    loop = []
    loop.extend(frames[1:]) # to go in one direction add all after the first frame
    loop.extend(frames[-2:0:-1]) # then to come back append in reverse direction
    frames[0].save(filename, format='GIF', append_images=loop, save_all=True, duration=100, loop=0)
    
def animate_saved(epoch_str, imsize=128):
    """
    Make a GIF from a saved network.
    """
    stem_name = "./output/{}".format(epoch_str)
    G = Genome.load(stem_name+".json")
    net = NNFF(G)
    gif_name = "{}_{:d}.gif".format(stem_name, imsize)
    animate(net, imsize=imsize, filename=gif_name)
    
def crossover_saved(estr1, estr2, imsize=64):
    """
    Crossover two saved genomes and save the child and resulting image.
    N.B. This won't work unless the two parents are from the same
    population (because they need to have the smae global innovation
    ids)
    """
    G1 = Genome.load("./output/{}.json".format(estr1))
    G2 = Genome.load("./output/{}.json".format(estr2))
    G3 = G1.crossover(G2)
    net = NNFF(G3)
    img = create_image2(net, (imsize,imsize))
    stem_name = "./output/e{}".format(get_epoch_str)
    imsave("{}_{:d}.png".format(stem_name, imsize), img, vmin=0, vmax=1)
    G3.save(stem_name+".json")
    print("Saved as: " + stem_name)
    

def get_epoch_str():
    """
    For filenames - the Unix epoch can be used as an identifier.
    eg. 
    e1594252310.json
    e1594252310_img64.png
    e1594252310_img256.gif
    """
    return str(int(time.time()))