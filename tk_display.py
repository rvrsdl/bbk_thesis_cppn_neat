"""
Using Tkinter to display/select images.
"""
import glob
import random
import re
from typing import List, Union

import tkinter as tk
from PIL import ImageTk, Image
from image_cppn import Image as MyImage
import numpy as np

import cppn

aborted = False

class ImgGrid(object):
    
    def __init__(self, path_or_arrays: Union[str, List[MyImage]], text: List[str] = None,
                 n_imgs: int = 35, nrows: int = 5, ncols: int = 7, title: str = "Image Grid",
                 default_scores: List[int] = None):
        """
        Produces a grid of images from saved PNG files.
        Pass in a path name (with wildcards eg. output/*.png)
        to display a batch of images.
        """
        self.root: tk.Tk = tk.Tk()
        self.nrows: int = nrows
        self.ncols: int = ncols
        self.imgs_rw: List[MyImage] = []
        self.imgs_tk: List[ImageTk.PhotoImage] = []
        self.labels: List[tk.Label] = []
        self.sliders: List[tk.scale] = []
        self.slider_visible: List[bool] = []
        self.scores: List[float] = []
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        if type(path_or_arrays) == str:
            from_saved = True
            # Use glob to turn wildcard string into list of paths.
            filenames = glob.glob(path_or_arrays)[:n_imgs]
            random.shuffle(filenames)
            n_imgs = min([n_imgs, len(filenames)])
        elif type(path_or_arrays) == list:
            from_saved = False
            n_imgs = min([n_imgs, len(path_or_arrays)])
        else:
            raise ValueError("Unknown input")
        self.n_imgs = n_imgs
        if text is None:
            text = np.tile(None, n_imgs) # So that we can subscript it.
        for i in range(n_imgs):
            r, c = divmod(i, self.ncols)
            if from_saved:
                f = filenames[i]
                this_img = ImageTk.PhotoImage(Image.open(f))  #, master=root
            else:
                im = path_or_arrays[i]
                self.imgs_rw.append(im)
                this_img = self.to_image_tk(im)
            #lab = tk.Label(root, image = imgs[i], borderwidth=2, relief='solid')
            self.imgs_tk.append(this_img)  # TKinter requires us to keep a list of images (even though we don't refer to it in the code).
            lab = tk.Label(self.root, image=this_img, text=text[i], compound='top')
            lab.grid(row=r, column=c, padx=1, pady=1)
            self.labels.append(lab)
            slider = tk.Scale(self.root)
            if default_scores is not None:
                slider.set(int(round(default_scores[i]))) # Round because it must be an int.
            self.sliders.append(slider)
            self.slider_visible.append(False) # All sliders start off hidden
            #estr = re.findall('e\d*',f)[0]
            #lab.bind("<Button-1>", func = lambda e,r=r,c=c: print("Row={}, Col={}".format(r,c)) )
            #lab.bind("<Button-1>", func = lambda e,l=lab: l.focus_set() )
            #lab.bind("<Button-1>", func = lambda e,l=lab: l.config(text = 'hello') )
            if from_saved:
                lab.bind("<Button-1>", func=lambda e, f=f: self.show_hi_res(f)) # TODO
            else:
                #lab.bind("<Button-1>", func = lambda e,l=lab: self.toggle_border(l), add='+' )
                lab.bind("<Button-1>", func=lambda e, s=self.sliders[i]: s.set(s.get()+10), add='+')
                lab.bind("<Button-3>", func=lambda e, s=self.sliders[i]: s.set(s.get()-10), add='+')
                lab.bind("<Double-Button-1>", func=lambda e, k=self.imgs_rw[i]: self.show_hi_res(k))
        # Now add some buttons at the bottom
        b4 = tk.Button(self.root, text="Zero Scores", command=self.zero_scores)
        b4.grid(row=nrows+1, column=ncols-4)
        b3 = tk.Button(self.root, text="Show Scores", command=self.show_all_sliders)
        b3.grid(row=nrows+1, column=ncols-3)
        b2 = tk.Button(self.root, text="Hide Scores", command=self.hide_all_sliders)
        b2.grid(row=nrows+1, column=ncols-2)
        b1 = tk.Button(self.root, text="Submit", command=self.submit_scores)
        b1.grid(row=nrows+1, column=ncols-1)
        self.check_scores_apply_border(repeat=False)
            
    def to_image_tk(self, img: MyImage) -> ImageTk:
        mode = 'RGB' if img.channels == 3 else 'L'
        return ImageTk.PhotoImage(image=Image.fromarray(np.uint8(img.data*255), mode=mode))
        
    def run(self):
        """
        Shows the grid.
        When the user clicks submit the window is closed and 
        the scores are returned.
        """
        self.root.after(500, self.check_scores_apply_border)
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
            
    def zero_scores(self):
        """
        Reset all scores to zero.
        """
        for s in self.sliders:
            s.set(0)
            
    def check_scores_apply_border(self, thresh=10, repeat=True):
        """
        This will be called during mainloop every second
        to check if a score has been raised above the threshold
        (10) by the user, and if so to run on the border.
        """
        for s, l in zip(self.sliders, self.labels):
            if s.get() >= thresh:
                l.config(borderwidth=2, relief='solid')
            else:
                l.config(borderwidth=0)
        if repeat:
            self.root.after(500, self.check_scores_apply_border)
        
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
            # To remove image and set text do:  image='', text='hello'

    def submit_scores(self):
        """
        Puts all the slider values in self.scores and closes
        the window (ending the main loop which was launched in
        self.run() )
        """
        self.scores = [s.get() for s in self.sliders]
        self.root.destroy() # Closes the window.

    def show_hi_res(self, img: MyImage):
        """
        Opens a new window showing an image in high resolution.
        """
        img_large = img.change_resolution(512)  # TODO: Get high res size from config file
        imgtk = self.to_image_tk(img_large)
        window = tk.Toplevel(self.root)
        window.title('High Resolution')
        lab = tk.Label(window, image=imgtk)
        lab.image = imgtk # Annoying tkinter thing. See here: http://effbot.org/pyfaq/why-do-my-tkinter-images-not-appear.htm
        save_button = tk.Button(window, text="Save", command=lambda: save_cmd())
        lab.pack()
        save_button.pack()

        def save_cmd():
            saved_path = img_large.save()
            save_button['state'] = tk.DISABLED
            lab['text'] = 'Saved here:\n{}'.format(saved_path)
            lab['compound'] = 'top'

    def on_closing(self):
        """
        This method gets called if the user closes the window.
        We want to end the whole iterative selction process
        """
        global aborted
        aborted = True
        self.root.destroy()
