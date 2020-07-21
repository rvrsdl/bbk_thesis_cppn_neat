"""
Using Tkinter to display/select images.
"""
import glob
import random
import re

import tkinter as tk
from PIL import ImageTk, Image
import numpy as np

import cppn


class ImgGrid(object):
    
    def __init__(self, path_or_arrays, n_imgs=35, nrows=5, ncols=7, title="Image Grid"):
        """
        Produces a grid of images from saved PNG files.
        Pass in a path name (with wildcards eg. output/*.png)
        to display a batch of images.
        """
        self.root = tk.Tk()
        self.nrows = nrows
        self.ncols = ncols
        self.imgs = [] # to hold ImageTk.PhotoImage
        self.labels = [] # to hold tk.Label
        self.sliders = [] # to hold tk.Scale
        self.slider_visible = [] # True if image is showing
        self.scores = []
        self.root.title(title)
        
        if type(path_or_arrays)==str:
            from_saved = True
            # Use glob to turn wildcard string into list of paths.
            filenames = glob.glob(path_or_arrays)[:n_imgs]
            random.shuffle(filenames)
            n_imgs = min([n_imgs, len(filenames)])
        elif type(path_or_arrays)==list:
            from_saved = False
            n_imgs = min([n_imgs, len(path_or_arrays)])
        else:
            raise ValueError("Unknown input")
        self.n_imgs = n_imgs
        for i in range(n_imgs):
            r, c = divmod(i, self.ncols)
            if from_saved:
                f = filenames[i]
                self.imgs.append( ImageTk.PhotoImage(Image.open(f)) ) #, master=root
            else:
                a = path_or_arrays[i]
                self.imgs.append( ImageTk.PhotoImage(image=Image.fromarray(np.uint8(a*255),mode='RGB')) ) #, master=root
            #lab = tk.Label(root, image = imgs[i], borderwidth=2, relief='solid')
            lab = tk.Label(self.root, image = self.imgs[i])
            lab.grid(row = r, column = c, padx = 0, pady = 0)
            self.labels.append(lab)
            self.sliders.append(tk.Scale(self.root))
            self.slider_visible.append(False) # All sliders start off hidden
            #estr = re.findall('e\d*',f)[0]
            #lab.bind("<Button-1>", func = lambda e,r=r,c=c: print("Row={}, Col={}".format(r,c)) )
            #lab.bind("<Button-1>", func = lambda e,l=lab: l.focus_set() )
            #lab.bind("<Button-1>", func = lambda e,l=lab: l.config(text = 'hello') )
            if from_saved:
                lab.bind("<Button-1>", func = lambda e,f=f: self.show_hi_res(f) )
            else:
                lab.bind("<Button-1>", func = lambda e,l=lab: self.toggle_border(l), add='+' )
                lab.bind("<Button-1>", func = lambda e,s=self.sliders[i]: s.set(s.get()+10), add='+' )
            lab.bind("<Button-3>", func = lambda e,r=r,c=c: self.toggle_slider(r,c) )
        # Now add some buttons at the bottom
        b1 = tk.Button(self.root, text="Show Scores", command=self.show_all_sliders)
        b1.grid(row=nrows+1, column=ncols-3)
        b2 = tk.Button(self.root, text="Hide Scores", command=self.hide_all_sliders)
        b2.grid(row=nrows+1, column=ncols-2)
        b3 = tk.Button(self.root, text="Submit", command=self.submit_scores)
        b3.grid(row=nrows+1, column=ncols-1)
            
    def run(self):
        """
        Shows the grid.
        When the user clicks submit the window is closed and 
        the scores are returned.
        """
        self.root.mainloop()
        #input('Press enter...')
        return self.scores
        
        
    def toggle_slider(self, row, col):
        index = np.ravel_multi_index((row, col), (self.nrows, self.ncols))
        if self.slider_visible[index]:
            self.sliders[index].grid_forget()
            self.slider_visible[index] = False
        else:
            self.sliders[index].grid(row=row,column=col)
            self.slider_visible[index] = True
            
    def show_all_sliders(self):
        for i in range(self.n_imgs):
            r, c = divmod(i, self.ncols)
            self.sliders[i].grid(row=r,column=c)
            self.slider_visible[i] = True
            
    def hide_all_sliders(self):
        for i in range(self.n_imgs):
            r, c = divmod(i, self.ncols)
            self.sliders[i].grid_forget()
            self.slider_visible[i] = False
        
    def toggle_border(self, lab):
        """
        Has no practical use but just toggles a border on/off on the clicked image.
        Maybe useful for marking ones you want to come back to?
        """
        current_border = lab.cget('borderwidth')
        if type(current_border)!=int:
            current_border = int(lab.cget('borderwidth').string)
        if current_border==2:
            lab.config(borderwidth=0)
        else:
            lab.config(borderwidth=2, relief='solid')
            # To remov image and set text do:  image='', text='hello'

    def submit_scores(self):
        """
        Puts all the slider values in self.scores and closes
        the window (ending the main loop which was launched in
        self.run() )
        """
        for s in self.sliders:
            self.scores.append(s.get())
        self.root.destroy() # Closes the window.
        
    def show_hi_res(self, img_filename):
        estr = re.findall('e\d*',img_filename)[0]
        img_array = cppn.upscale_saved(estr, save=False)
        img = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(img_array*255),mode='RGB'))
        window = tk.Toplevel(self.root)
        window.title(estr)
        lab = tk.Label(window, image=img)
        lab.image = img # Annoying tkinter thing. See here: http://effbot.org/pyfaq/why-do-my-tkinter-images-not-appear.htm
        lab.pack()

        
        
        