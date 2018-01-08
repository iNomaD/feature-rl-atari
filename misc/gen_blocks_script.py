import numpy
import scipy.misc

i = 1
for width in range(8, 144+8, 8):
    a = numpy.zeros((6, width, 3), dtype=numpy.uint8)
    a[...,0] = 200
    a[...,1] = 72
    a[...,2] = 72
    scipy.misc.imsave('Breakout-v0-c8-'+str(i)+'.bmp', a)
    i = i + 1