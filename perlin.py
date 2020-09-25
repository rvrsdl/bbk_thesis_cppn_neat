# Inspired by: https://medium.com/@yvanscher/playing-with-perlin-noise-generating-realistic-archipelagos-b59f004d8401
# And: https://stackoverflow.com/questions/60350598/perlin-noise-in-pythons-noise-library
import noise
import numpy as np
#from PIL import Image
#import matplotlib.pyplot as plt

def get_perlin_noise(shape=(128,128), seed=np.random.randint(0,100)):
    scale = 0.5
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
    
    world = np.zeros(shape)
    
    # make coordinate grid on [0,1]^2
    x_idx = np.linspace(0, 1, shape[0])
    y_idx = np.linspace(0, 1, shape[1])
    world_x, world_y = np.meshgrid(x_idx, y_idx)
    
    # apply perlin noise, instead of np.vectorize, consider using itertools.starmap()
    world = np.vectorize(noise.pnoise2)(world_x/scale,
                            world_y/scale,
                            octaves=octaves,
                            persistence=persistence,
                            lacunarity=lacunarity,
                            repeatx=1024,
                            repeaty=1024,
                            base=seed)
    
    return world+0.5

# img = np.floor((world + .5) * 255).astype(np.uint8) # <- Normalize world first
# plt.imshow(img)
# #Image.fromarray(img, mode='L').show()