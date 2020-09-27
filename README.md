# A System for Generating Artistic Images using an Interactive Evolutionary Algorithm
## Birkbeck MSc Computer Science Project

## Intro
This system uses K. Stanley's CPPN-NEAT algorithm to allow the user to evolve images interactively.
A simple user interface is provided, allowing selection and saving of the images.
The system can also be run in "automatic" mode where a pretrained ImageNet ANN selects the images to be evolved.

Example results can be viewed at: https://www.instagram.com/evolved_art_neat/

## Setup Instructions
* Clone Repo
```
$ git clone https://github.com/Birkbeck/msc-computer-science-project-2019-20-files-rhonro01.git
```
  * Then cd into the repo directory
* Set up a virtual environment and install dependencies.
   * Using conda:
```
$ conda env create -f environment.yml
```
   * Or using virtualenv and pip:
```
$ virtualenv -p python3.8 venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
* To view graph diagrams it may be necessary to install graphviz if you don't already have it.
    * First check whether you already have it with the command `dot -V` . If the output is something like `dot - graphviz version 2.40.1` then you already have it,
    otherwise you should install it if you want to view the graph diagrams (although it is not necessary for the core functionality of evolving images). 
    * On a Mac using the Homebrew package manager Graphviz can be installed using the command `brew install graphviz`
    * On Ubuntu it can be installed with the command `sudo apt-get install graphviz`

# Run
The entry point to the program is main.py. A config file must be provided as a command line argument
(it will expect this to be in the `./configurations/` folder).

To get started using one of the preset configuration files eg.:
```
$ python main.py config_manual.yml
```

# Contents
Contents:
- `main.py`: The entry point for the program.
- `configurations/*.yml`: A few different configuration files.
- `src/population.py`: Contains the `Population` class which manages the evolutionary process.
- `src/genome.py`: contains the `Genome` class which handles mutation etc.
- `src/imaging.py`: Contains the `ImageCreator` class, used to create images from `Genome` objects; and the `Image` class used to hold image data. 
- `src/cppn.py`: Contains the `CPPN` class, which turns a `Genome` into a Compositional Pattern Producing Network.
- `src/evaluators.py`: Contains several Evaluator classes which are used to assess the fitness of images.
- `src/userinterface.py`: Contains the ImgGrid class which manages Tk GUI windows.   
- `src/visualise`: Contains functions to allow rendering of `Genome` objects as graph diagrams.
- `src/funcs`: Contains activation functions used by the CPPN.
- `src/perlin`: Contains functions for creating Perlin noise.
- `src/fourier`: Contains functions to map Cartesian coordinates to higher dimensional space using Fourier features.
- `tests/`: Contains unit tests used by pytest.
- `conftest.py`: Allows pytest to discover tests.
