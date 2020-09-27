# -*- coding: utf-8 -*-

"""
Contains various fitness evaluators that can be passed to
a Population object.
"""
import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
# Annoying to have these inputs here as not needed for everything
import tensorflow as tf
import tensorflow_hub as hub

from src.genome import Genome
from src.imaging import ImageCreator, Image
from src import tk_display as td

# typing definitions
Genomes = List[Genome]
Images = List[Image]


class AbstractEvaluator(ABC):

    def __init__(self, image_creator: ImageCreator, visible: bool = False,
                 breed_method: str = None, thresh: int = None) -> None:
        self._image_creator = image_creator
        self.visible = visible
        self.breed_method = breed_method
        self.thresh = thresh
        self.gameover = False  # We can set this to true to break out of caller loop
        self.show_gens = 10

    @abstractmethod
    def run(self, genomes: Genomes, gen_num: int) -> None:
        pass
    
    def show_grid(self, imgs: Images, text: np.array = None,
                  default_scores: np.array = None, gen_num: int = 0) -> np.array:
        grd = td.ImgGrid(imgs, text=text, n_imgs=28, nrows=4,
                         title="Generation {}".format(gen_num),
                         default_scores=default_scores,
                         select_method=self.breed_method, thresh=self.thresh)
        ratings = grd.run()
        if td.aborted:
            self.gameover = True
        return ratings
    
    def _create_images(self, genomes: Genomes) -> Images:
        """
        Any evaluator will need to generate images from the
        genomes so this method can be shared between them.
        """
        return [self._image_creator.create_image(g) for g in genomes]


class InteractiveEvaluator(AbstractEvaluator):
    
    def __init__(self, image_creator: ImageCreator, **kwargs) -> None:
        # Pass through args to the base class constructor but with show_grid=True
        super().__init__(image_creator, visible=True, **kwargs)
    
    def run(self, genomes: Genomes, gen_num: int) -> None:
        imgs = self._create_images(genomes)
        default_scores = np.array([g.get_fitness() for g in genomes])
        ratings = self.show_grid(imgs, default_scores=default_scores, gen_num=gen_num)
        # Now set the genome fitnesses according to the ratings
        for g, r in zip(genomes, ratings):
            g.metadata['fitness'] = r


class PixelDiffEvaluator(AbstractEvaluator):
    
    def __init__(self, image_creator: ImageCreator, target_img: Image = None, **kwargs) -> None:
        super().__init__(image_creator, **kwargs)
        if target_img is None:
            self.target_img = self.get_default_target()
        else:
            self.target_img = target_img
        self.channels = self.target_img.channels
        # Could do something to see how close RGB colours are
        self.pixels = self.target_img.size[0] * self.target_img.size[0] * 1

    def run(self, genomes: Genomes, gen_num: int) -> None:
        imgs = self._create_images(genomes)
        ratings = [self.dist_rating(img) for img in imgs]
        if self.visible or (gen_num % self.show_gens==0):
            user_ratings = self.show_grid(imgs, default_scores=ratings, gen_num=gen_num)
            if user_ratings:
                ratings = user_ratings  # WARNING: this allows user to modify default ratings
        # Now set the genome fitnesses according to the ratings
        for g, r in zip(genomes, ratings):
            g.metadata['fitness'] = r
            
    def dist_rating(self, img: Image) -> float:
        """
        L2 Pixel distance
        """
        # squares for each image first. ie. effectively
        # lower the resolution.
        if self.channels == 1:
            # grayscale so we can just do squared diff of brightness for each pixel
            l2_dist = np.sqrt(np.sum((self.target_img.data.ravel() - img.data.ravel())**2))
            rating = 1 - (l2_dist / self.pixels)
        elif self.channels == 3:
            # TODO 
            pass
        return rating*100

    @staticmethod
    def get_default_target() -> Image:
        """ 
        Returns a quartered checkerboard image.
        """
        ones = np.tile(1, (64, 64))
        zeros = np.tile(0, (64, 64))
        left = np.concatenate((ones, zeros), axis=0)
        right = np.concatenate((zeros, ones), axis=0)
        out = np.concatenate((left, right), axis=1)
        return Image(out)


class PixelPctEvaluator(PixelDiffEvaluator):
    
    def pct_ok(self, img: Image) -> float:
        """
        Overriding base class method to find percent of
        pixels within a certain shade of target.
        The point is to avoid getting all grey when
        target is checkerboard.
        """
        thresh = 0.05 # pixels have to be within 5% of the target value
        # TODO: thresh should get lower as learning goes on.
        # maybe based on generation number? or something more advanced.
        if self.channels == 1:
            num_ok = np.sum(np.abs(self.target_img.data.ravel() - img.data.ravel()) <= thresh)
            rating = num_ok / self.pixels
        elif self.channels == 3:
            # TODO
            pass
        return rating
    
    def diff_pct_ok(self, img: Image) -> float:
        """
        Just check that we get the major tone changes.
        """
        #thresh = 0.95 # Now thresh is the majorness of the tone changes we care about
        # ie. 1 is complete black to white or vice versa.
        thresh = np.percentile(self.target_img.diff().data, 99) # threshold at 95th percentile. TODO: needs more thought!
        if self.channels == 1:
            # How many of the major changes in the target pic are also major changes in the candidate
            tgt_major_diff_loc = self.target_img.diff().data >= thresh
            img_major_diff_loc = img.diff().data >= thresh
            # Jaccard coeff ie. intersection over union
            rating = jaccard(tgt_major_diff_loc, img_major_diff_loc)
        elif self.channels == 3:
            # TODO
            pass
        return rating
    
    def dist_rating(self, img: Image) -> float:
        """
        Just using 50/50 normal and diff pct
        """
        # NB rating is expected to be between 0 and 100
        return (self.pct_ok(img)*50) + (self.diff_pct_ok(img)*50)


class ImageNetEvaluator(AbstractEvaluator):
    """
    Introducing a "fade factor". Images where the most likely 
    category has been seen before get their rating downgraded
    by multiplying by the fade factro (eg. 0.98). The idea is
    to improve novelty search - if we have seen loads of "jellyfish"
    they begin to become less desirable. This should stop us going
    down a jellyfish rabbithole early on. (NB fade factors compound
    so if we have seen 10 jellyfish we will be multiplying by 0.98^10)
    """

    def __init__(self, image_creator: ImageCreator, fade_factor: float = 1, **kwargs) -> None:
        super().__init__(image_creator, **kwargs)
        self._fade_factor = fade_factor
        self._class_seen = dict()  # To keep track of how many times we have seen each class
        self._model = tf.keras.Sequential([
            hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/classification/4")
            ])
        self._model.build([None, 128, 128, self._image_creator.channels])  # Batch input shape.
        # load class labels
        labels_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ImageNetLabels.txt'))
        with open(labels_file, "r") as f:
            self._class_labels = np.array([line.strip() for line in f.readlines()])
        
    def run(self, genomes: Genomes, gen_num: int) -> None:
        imgs = self._create_images(genomes)
        model_input = self.imgs2tensor(imgs)  # turn it into a tensor
        logits = self._model(model_input) # use the model
        probs = np.array(tf.nn.softmax(logits))
        max_idx = np.argmax(probs, axis=1)
        best_probs = probs[np.arange(len(probs)), max_idx]
        best_labels = self._class_labels[max_idx]
        for lab in best_labels:
            # Each time we see a winning label, compound the fade factor for that class
            # (class fades are initialised to 1 if not already in dictionary)
            self._class_seen[lab] = self._class_seen.setdefault(lab, 0) + 1
        multiplier = np.array([self._fade_factor ** self._class_seen[l] for l in best_labels])
        ratings = best_probs * multiplier * 100  # *100 because ratings need to be in range 0-100
        if self.visible or (gen_num % self.show_gens==0):
            text = ['{:.12}: {:.0f}%'.format(l, p*100) for (l, p) in zip(best_labels, best_probs)]
            user_ratings = self.show_grid(imgs, text=text, default_scores=ratings, gen_num=gen_num)
            #user_ratings = self.show_grid(imgs, text=text, default_scores=ratings, gen_num=gen_num)
            if user_ratings:
                ratings = user_ratings # WARNING: this allows user to modify default ratings
        # Now set the genome fitnesses according to the ratings
        for g, fit, raw, spec in zip(genomes, ratings, best_probs, best_labels):
            g.metadata['fitness'] = fit
            g.metadata['raw_fitness'] = raw * 100
            g.metadata['species'] = spec
        
    def imgs2tensor(self, imgs: Images): # -> tf.python.framework.ops.EagerTensor:
        """
        Converts a list of Images generated by CPPN object
        into a "batch" of tensorflow images ready to be put
        through a tensorflow model.
        The shape will be [batch_size, height, width, 3] for 
        an RGB image.
        """
        return tf.stack([i.data for i in imgs])


def jaccard(a, b):
    """
    Returns Jaccard coefficient of two boolean numpy arrays 
    of the same size ie. sum(a & b) / sum(a | b )
    """
    return np.sum(a & b) / np.sum(a | b)


def hhi(a: np.ndarray, axis: int) -> np.ndarray:
    """
    Calculates the Herfindahl-Hirschman Index (a measure of concentration)
    along the axis specified.
    """
    return np.square(a/a.sum(axis=axis)).sum(axis=axis)