# About

Installable version of Charles McGuinness' `focusstack` project (see *Original description* below), also translated from python 2 to python 3.

To install: `cd` to root of package (where *setup.py* is), and
```bash
pip install .
```
or, for an editable install:
```bash
pip install -e .
```

Then, for example
```python
from focusstack import stack

# process all files in current directory
stack()

# process all .jpg files in the specified directory (case-sensitive)
stack('img/to_stack/', pattern='*.jpg')

# Specify iterable of files instead
stack(files=['IMG_3331.JPG', 'IMG_3332.JPG'])

# save processed image in a different directory
stack(savepath='img/processed')

# use ORB image matching algorithm (default: SIFT, can also be SURF but only
# for users with a license to use SURF)
stack(algo='orb')

# Change kernel and blur sizes (default: 5 for both)
stack(kernel_size=11, blur_size=11)
```

It is also possible to use the module from the console (e.g. bash) directly:
```bash
python -m focusstack
```
(process all files in current directory with default arguments).

# Original description

Simple Focus Stacking in Python

This project implements a simple focus stacking algorithm in Python.

Focus stacking is useful when your depth of field is shallower than
all the objects you wish to capture (in macro photography, this is
very common).  Instead, you take a series of pictures where you focus
progressively throughout the range of images, so that you have a series
of images that, in aggregate, have everything in focus.  Focus stacking,
then, merges these images by taking the best focused region from all
of them to build up the final composite image.

The focus stacking logic is contained in FocusStack.py.  There is a
sample driver program in main.py.  It assumes that there is a subdirectory
called "input" which contains the source images.  It generates an output
called "merged.png".

I have also included some sample images in the input directory to allow
you to experiment with the code without having to shoot your own set of images.

This project is Copyright 2015 Charles McGuinness, and is released under the
Apache 2.0 license (see license file for precise details).


# Testing

With pytest : cd into root of package, and
```bash
pytest
```


# Contributors

Olivier Vincent
(ovinc.py@gmail.com)