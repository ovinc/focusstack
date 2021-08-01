"""Simple Focus Stacker

    Author:     Charles McGuinness (charles@mcguinness.us)
    Copyright:  Copyright 2015 Charles McGuinness
    License:    Apache License 2.0


This code will take a series of images and merge them so that each
pixel is taken from the image with the sharpest focus at that location.

The logic is roughly the following:

1.  Align the images.  Changing the focus on a lens, even
    if the camera remains fixed, causes a mild zooming on the images.
    We need to correct the images so they line up perfectly on top
    of each other.

2.  Perform a gaussian blur on all images

3.  Compute the laplacian on the blurred image to generate a gradient map

4.  Create a blank output image with the same size as the original input
    images

4.  For each pixel [x,y] in the output image, copy the pixel [x,y] from
    the input image which has the largest gradient [x,y]


This algorithm was inspired by the high-level description given at

http://stackoverflow.com/questions/15911783/what-are-some-common-focus-stacking-algorithms
"""

from pathlib import Path
import numpy as np
import cv2


def find_homography(image_1_kp, image_2_kp, matches):
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

    for i in range(0, len(matches)):
        image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
        image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

    homography, mask = cv2.findHomography(image_1_points, image_2_points,
                                          cv2.RANSAC, ransacReprojThreshold=2.0)

    return homography


def align_images(images, use_sift=True, output=False):
    """Align images so they overlap properly.

    Parameters
    ----------
    - use_sift: bool (default: True)
        SIFT generally produces better results, but it is not FOSS, so chose
        the feature detector that suits the needs of your project. ORB does OK.

    - output: bool (default: False)
        If you find that there's a large amount of ghosting, it may be because
        one or more of the input images gets misaligned. Outputting the
        aligned images may help diagnose that.
    """
    outimages = []

    if use_sift:
        detector = cv2.xfeatures2d.SIFT_create()
    else:
        detector = cv2.ORB_create(1000)

    #   We assume that image 0 is the "base" image and align everything to it
    print("Detecting features of base image")
    outimages.append(images[0])
    image1gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    image_1_kp, image_1_desc = detector.detectAndCompute(image1gray, None)

    for i in range(1, len(images)):
        print("Aligning image {}".format(i))
        image_i_kp, image_i_desc = detector.detectAndCompute(images[i], None)

        if use_sift:
            bf = cv2.BFMatcher()
            # This returns the top two matches for each feature point (list of list)
            pairMatches = bf.knnMatch(image_i_desc, image_1_desc, k=2)
            rawMatches = []
            for m, n in pairMatches:
                if m.distance < 0.7 * n.distance:
                    rawMatches.append(m)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            rawMatches = bf.match(image_i_desc, image_1_desc)

        sortMatches = sorted(rawMatches, key=lambda x: x.distance)
        matches = sortMatches[0:128]

        hom = find_homography(image_i_kp, image_1_kp, matches)
        newimage = cv2.warpPerspective(images[i], hom,
                                       (images[i].shape[1], images[i].shape[0]),
                                       flags=cv2.INTER_LINEAR)

        outimages.append(newimage)

        if output:
            cv2.imwrite("aligned{}.png".format(i), newimage)

    return outimages


def do_lap(image, kernel_size=5, blur_size=5):
    """Compute the gradient map of the image.

    Parameters
    ----------
    YOU SHOULD TUNE THESE VALUES TO SUIT YOUR NEEDS:
    - kernel_size: Size of the laplacian window
    - blur_size: How big of a kernal to use for the gaussian blur
    Generally, keeping these two values the same or very close works well.
    Also, odd numbers, please...
    """
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)


def focus_stack(unimages, use_sift=True):
    """Find the points of best focus in all images and produce a merged result.

    Parameters
    ----------
    - use_sift: book (default False): see align_images()
    """
    images = align_images(unimages, use_sift=use_sift)

    print("Computing the laplacian of the blurred images")
    laps = []
    for i in range(len(images)):
        print("Lap {}".format(i))
        laps.append(do_lap(cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)))

    laps = np.asarray(laps)
    print("Shape of array of laplacians = {}".format(laps.shape))

    output = np.zeros(shape=images[0].shape, dtype=images[0].dtype)

    abs_laps = np.absolute(laps)
    maxima = abs_laps.max(axis=0)
    bool_mask = abs_laps == maxima
    mask = bool_mask.astype(np.uint8)
    for i in range(0, len(images)):
        output = cv2.bitwise_not(images[i], output, mask=mask[i])

    return 255 - output


def stack(path='.', savepath='.', pattern=None, use_sift=True):
    """Perform focus stacking for images in specified path.

    Parameters
    ----------
    path: str or Path object in which image files are located
    extension (default None): extension of img files to consider (e.g. '.jpg')
    """
    pattern = '*.*' if pattern is None else pattern
    image_files = Path(path).glob(pattern)

    focusimages = []

    for image_file in image_files:
        print(f"Reading in file {image_file.name}")
        img = cv2.imread(str(image_file.resolve()))
        focusimages.append(img)

    merged_image = focus_stack(focusimages, use_sift=use_sift)
    merged_file = str((Path(savepath) / "merged.png").resolve())

    cv2.imwrite(merged_file, merged_image)
