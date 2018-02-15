## The aim of this project is the development of intelligent agents which are able to play atari games.

### Technologies
* Python3
* OpenCV
* Tensorflow
* OpenAI Gym - Atari

### Models after 48 hours of training
![Pong](https://github.com/iNomaD/feature-rl-atari/blob/master/res/animation/Pong.gif)
![Breakout](https://github.com/iNomaD/feature-rl-atari/blob/master/res/animation/Breakout.gif)
![SpaceInvaders](https://github.com/iNomaD/feature-rl-atari/blob/master/res/animation/SpaceInvaders.gif)
![MsPacman](https://github.com/iNomaD/feature-rl-atari/blob/master/res/animation/MsPacman.gif)

### Installation
* Get Python3. If you are using Anaconda simply type `conda create -n atari36 python=3.6`)
* Activate virtual env `source activate atari36`
* Install dependencies `pip install numpy scipy tensorflow opencv-python gym atari-py Pillow PyOpenGL`
* For Windows refer here `https://github.com/j8lp/atari-py` to install atari environment
* On Linux you can get Intel optimized tensorflow wheel:

        pip install https://anaconda.org/intel/tensorflow/1.4.0/download/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl

### Usage
To train the model:

    python start_atari_dqn.py -g Pong-v0
    
The model is saved to `%GAME_ID%/my_dqn.ckpt` by default. To view it in action, run:

    python start_atari_dqn.py -r -t

For more options:

    python tiny_dqn.py --help

### Links
* https://github.com/ageron/tiny-dqn