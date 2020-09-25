from image_cppn import Image, CPPN

imsize = (128,128)
x_coords, y_coords, r_coords, phi_coords = CPPN.get_coords(imsize)

imc = phi_coords - np.min(phi_coords)
imc = imc / np.max(imc)
im = Image(imc)
im.show()

import funcs
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 128)
y = 1-funcs.gaussian(x*2)
plt.plot(x,y)
plt.show()

axs[0,0].set_xlim([-1,1])
axs[0,0].set_ylim([0,1])
axs[0,0].set_title('y = sigmoid(5x)')
axs[0,0].set_ylabel('(a)', y=1, fontweight='bold', labelpad=10, rotation=0)

import noise
