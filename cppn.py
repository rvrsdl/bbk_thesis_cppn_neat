import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
#import matplotlib.animation as animation
from PIL import Image

from genome import Genome
from nnet import NNFF
from netviz import netviz

import fourier


CHANNELS = 3 # 3 for RGB
FFEATS = 10 # NUmber of fourier features (if using)
B = fourier.initialize_fourier_mapping_vector(n_features=FFEATS)
# ^ should probably belong to the genome so we
# dn't get randomness (although the feature generation
# does feel like part of the image-making process
# rather than the genetic process...)

def get_coords(imsize=(64,64)):
    x = np.linspace(-1, 1, imsize[0])
    y = np.linspace(-1, 1, imsize[1])
    xx, yy = np.meshgrid(x, y)
    coords = np.stack([xx,yy], axis=-1)
    return coords

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
    # size_x = imsize[0]
    # size_y = imsize[1]
    # pixels = size_x*size_y
    # # Generate the coordinate pairs for inputs (normalised)
    # xcoords = np.tile(range(size_x), (size_y,1))
    # ycoords = np.tile(np.array([range(size_y)]).transpose(), (1,size_x))
    # xinp = (xcoords - np.mean(xcoords)) / np.std(xcoords)
    # yinp = (ycoords - np.mean(ycoords)) / np.std(ycoords)
    # dinp = np.sqrt(xinp**2 + yinp**2)
    # img_raw = np.array(net.feedforward((xinp.ravel(), yinp.ravel(), dinp.ravel(), np.tile(bias, pixels))))
    coords = get_coords(imsize=imsize)
    size_x = imsize[0]
    size_y = imsize[1]
    pixels = size_x * size_y
    xcoords= coords[:,:,0].ravel()
    ycoords= coords[:,:,1].ravel()
    dcoords = np.sqrt(xcoords**2 + ycoords**2)
    bias_tile = np.tile(bias, pixels)
    img_raw = np.array(net.feedforward((xcoords, ycoords, dcoords, bias_tile)))
    if CHANNELS==1:
        img_square = img_raw.T.reshape(size_x, size_y)
    else:
        img_square = img_raw.T.reshape(size_x, size_y, CHANNELS)
    return img_square

def create_image3(net, imsize, bias=[0.2,0.4,0.6,0.8]):
    """
    For use with a bias/noise vector of arbitrary length
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
    noise = np.tile(bias, (pixels,1))
    img_raw = np.array(net.feedforward((xinp.ravel(), yinp.ravel(), dinp.ravel(), *noise.T)))
    if CHANNELS==1:
        img_square = img_raw.T.reshape(size_x, size_y)
    else:
        img_square = img_raw.T.reshape(size_x, size_y, CHANNELS)
    return img_square

def create_image_fourier(net, imsize, fourier_map_vec, bias=1):
    coords = get_coords(imsize=imsize)
    size_x, size_y = imsize
    pixels = size_x * size_y
    feats = fourier.fourier_mapping(coords, fourier_map_vec)
    n_ffeats = fourier_map_vec.shape[0]
    feats = feats.reshape(pixels, n_ffeats*2)
    bias_tile = np.tile(bias, pixels)
    img_raw = np.array(net.feedforward((*feats.T, bias_tile)))
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
    
def create_genome(input_nodes=4):
    G = Genome(input_nodes, CHANNELS)
    # Make it more complex
    for i in range(20): # 30 seems about the minimum to get interesting
        G.add_node()
        for j in range(i//2):
            G.add_connection()
    G.randomise_weights()
    return G

def do_run(num=10, imsize=128, fourier=False, ffeats=10):
    """
    Do a run of eg. 10 images, saving the images (.png) 
    and the corresponding networks (.json) in the output folder.
    """
    if fourier:
        in_nodes = (ffeats*2)+1 # pls 1 for bias input
    else:
        in_nodes = 4 #
    for i in range(num):
        print("Creating image {:d}".format(i))
        stem_name = "./output/e{}".format(get_epoch_str())
        G = create_genome(input_nodes=in_nodes)
        G.save(stem_name+".json")
        net = NNFF(G)
        if fourier:
            img = create_image_fourier(net, (imsize,imsize), n_features=ffeats)
        else:
            img = create_image2(net, (imsize,imsize))
        #show_image(img)
        imsave("{}_{:d}.png".format(stem_name, imsize), img, vmin=0, vmax=1, cmap='binary')
        

def upscale_saved(epoch_str, imsize=512, save=True):
    """
    Load one of the saved networks and produce a hi res image.
    (Either save it or return as an array.)
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
    if save:
        save_name = "{}_{}.png".format(stem_name, str_imsize)
        imsave(save_name, img, vmin=0, vmax=1, cmap='binary')
        return save_name
    else:
        return img
    
def animate(net, imsize=128, filename='test.gif'):
    """
    By varying the bias input we can alter the image in smooth steps.
    This function makes a GIF of this.
    """
    frames = []
    #np.arange(-1.8,2,0.2):
    for bias in np.arange(-0.8,0.8,0.2):
        img_dat = create_image2(net, (imsize,imsize), bias)
        frames.append(Image.fromarray(np.uint8(img_dat*255)))
    loop = []
    loop.extend(frames[1:]) # to go in one direction add all after the first frame
    loop.extend(frames[-2:0:-1]) # then to come back append in reverse direction
    frames[0].save(filename, format='GIF', append_images=loop, duration=120, loop=0)
    
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
    
def netviz_saved(epoch_str):
    """
    Saves a PNG of the visualisation of a saved (json) network.
    PROBLEM: Currently saves two files, one of which is empty.
    """
    G = Genome.load("./output/{}.json".format(epoch_str))
    nn = netviz(G)
    nn.format = 'png'
    out_fn = "./output/{}_net.png".format(epoch_str)
    nn.render(filename=out_fn, format='png')
    print("Saved as: {}".format(out_fn))
    
    
def get_epoch_str():
    """
    For filenames - the Unix epoch can be used as an identifier.
    eg. 
    e1594252310.json
    e1594252310_img64.png
    e1594252310_img256.gif
    """
    return str(int(time.time()*100))