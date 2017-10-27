# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mtplt


def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")  # Single input, single output
    print(sys.argv[
              0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")  # Two inputs, three outputs
    print(sys.argv[
              0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")  # Two inputs, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):
    # Write histogram equalization here
    # Display the result
    # Histogram equalization result

    #Reference tutorial: http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html

    color = ('b', 'g', 'r')
    #split the image into b,g,r channels and process each channel separately
    images = cv2.split(img_in)

    for col, i in zip(range(len(images)), ('b', 'g', 'r')):
        hist = cv2.calcHist([images[col]], [0], None, [256], [0, 256])
        plt.plot(hist, color=i)
        plt.xlim([0, 256])

        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        #save the final image for each channel
        images[col] = cdf[images[col]]
    plt.show()
    #merge to get the final image
    img_equalized = cv2.merge((images[0], images[1], images[2]))
    color = ('b', 'g', 'r')
    for i, col in enumerate((color)):
        hist = cv2.calcHist([img_equalized], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()

    # Code to verify calcHist using equalizeHist
    b, g, r = cv2.split(img_in)
    equb = cv2.equalizeHist(b)
    equg = cv2.equalizeHist(g)
    equr = cv2.equalizeHist(r)
    img_equalizeHist = cv2.merge((equb, equg, equr))
    res = np.hstack((img_equalized, img_equalizeHist))  # stacking images side-by-side

    cv2.imshow('image comparison', res)
    cv2.waitKey(0)

    img_out = img_equalized
    return True, img_out


def Question1():
    # Read in input images
    input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
    # Histogram equalization
    succeed, output_image = histogram_equalization(input_image)

    # Write out the result
    output_name = sys.argv[3] + "1.jpg"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def low_pass_filter(img_in):
    # Write low pass filter here
    # convert bgr to grayscale
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY);
    #take DFT
    dft = cv2.dft(np.float32(img_in), flags=cv2.DFT_COMPLEX_OUTPUT)
    #shift it to the middle region
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img_in.shape
    crow, ccol = rows / 2, cols / 2
    # create a mask first, center square is 1, remaining all zeros to pass low frequencies
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 1

    # apply mask and inverse DF
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    img_out = img_back  # Low pass filter result
    return True, img_out


def high_pass_filter(img_in):
    # Write high pass filter here
    # convert bgr to grayscale
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY);
    dft = cv2.dft(np.float32(img_in), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = img_in.shape
    crow, ccol = rows / 2, cols / 2
    # create a mask first, center square is 0, remaining all ones to block low frequencies
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0

    # apply mask and inverse DF
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    img_out = img_back  # High pass filter result
    return True, img_out


def deconvolution(img_in):
    # Write deconvolution codes here
    gk = cv2.getGaussianKernel(21, 5)
    #take transpose to get a square matrix
    gk = gk * gk.T

    def ft(im, newsize=None):
        dft = np.fft.fft2(np.float32(im), newsize)
        return np.fft.fftshift(dft)

    def ift(shift):
        f_ishift = np.fft.ifftshift(shift)
        img_back = np.fft.ifft2(f_ishift)
        return np.abs(img_back)

    imf = ft(img_in, (img_in.shape[0], img_in.shape[1]))  # make sure sizes match
    gkf = ft(gk, (img_in.shape[0], img_in.shape[1]))  # so we can multiple easily
    # divide the FT of the blurred image by the FT of the kernel for deconvolution
    imconvf = imf / gkf

    # now for example we can reconstruct the blurred image from its FT
    blurred = ift(imconvf)
    # Multiply by 255 to get in the range of [0,255] from [0,1]
    blurred = blurred * 255
    img_out = blurred  # Deconvolution result

    return True, img_out


def Question2():
    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Low and high pass filter
    succeed1, output_image1 = low_pass_filter(input_image1)
    succeed2, output_image2 = high_pass_filter(input_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(input_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "2.jpg"
    output_name2 = sys.argv[4] + "3.jpg"
    output_name3 = sys.argv[4] + "4.jpg"

    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)

    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):
    # Write laplacian pyramid blending codes here
    # make images rectangular
    img_in1 = img_in1[:, :img_in1.shape[0]]
    img_in2 = img_in2[:img_in2.shape[0], :img_in2.shape[0]]
    # generate Gaussian pyramid for A
    G = img_in1.copy()
    gpA = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = img_in2.copy()
    gpB = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpB.append(G)
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]] #store last gaussian to add to the HP filter while generating the image back
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)
    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1, 6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return True, ls_


def Question3():
    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
    # input_image1 = cv2.cvtColor(cv2.imread(sys.argv[2]), cv2.COLOR_BGR2RGB)
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
    # input_image2 = cv2.cvtColor(cv2.imread(sys.argv[3]), cv2.COLOR_BGR2RGB)

    # Laplacian pyramid blending
    succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)

    # Write out the result
    output_name = sys.argv[4] + "5.jpg"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

    if (question_number == 1 and not (len(sys.argv) == 4)):
        print 'sys.argvs', sys.argv[0], sys.argv[1], sys.argv[2]
        help_message()
        sys.exit()
    if (question_number == 2 and not (len(sys.argv) == 5)):
        print 'sys.argvs', sys.argv[0], sys.argv[1], sys.argv[2]
        help_message()
        sys.exit()
    if (question_number == 3 and not (len(sys.argv) == 5)):
        print 'sys.argvs', sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
        help_message()
        sys.exit()
if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
    print("Input parameters out of bound ...")
    sys.exit()

function_launch = {
    1: Question1,
    2: Question2,
    3: Question3,
}

# Call the function
function_launch[question_number]()
