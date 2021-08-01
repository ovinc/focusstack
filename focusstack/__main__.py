"""Use module as a main file (python -m focusstack)."""

import argparse
from focusstack import stack

msg = "Focus Stacking of Images."

# Below, prog= is to have correct name of module in help and not __main__.py
# (see e.g. https://bugs.python.org/issue22240)

parser = argparse.ArgumentParser(description=msg,
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 prog='python -m focusstack')

# Arguments to interact with Stemmer camera directly (stemmer module) --------

parser.add_argument('--path', type=str, default=None,
                    help='str or Path object in which image files are located')

parser.add_argument('--pattern', type=str, default=None,
                    help='glob pattern of files consider (e.g. "*.jpg")')

parser.add_argument('--files', type=str, default=None,
                    help='if specified, take these specific files instead')

parser.add_argument('--savepath', type=str, default=None,
                    help='str or Path object; where to save the final, stacked image')

parser.add_argument('--kernel_size', type=int, default=None,
                    help='size of the Laplacian window')

parser.add_argument('--blur_size', type=int, default=None,
                    help='size of the kernel used for gaussian blur')

parser.add_argument('--algo', type=str, default=None,
                    help="algorithm used to find image features: 'sift', 'surf', 'orb'")

args = parser.parse_args()

kwargs = {k: v for k, v in vars(args).items() if v is not None}

stack(**kwargs)
