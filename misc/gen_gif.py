import os
import imageio


frames = []
filenames = os.listdir('frames')
filenames = sorted(filenames,key=lambda x: int(os.path.splitext(x)[0]))
for file in filenames:
    frames.append(imageio.imread('frames/'+file))

imageio.mimsave('gif.gif', frames, fps=60, subrectangles=True)