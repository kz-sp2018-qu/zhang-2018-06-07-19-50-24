## Reinforcement Learning: Cartpole

The experiment shows a simple implementation of Deep Q-learning and how to apply it to play cartpole. In addition, one can see how to pack the whole thing in  a docker image.

## Prerequisite
### Create environment 
The environment for this experiment will be build by [Keras](https://keras.io/) and [OpenAI Gym](https://github.com/openai/gym). The docker image is based on Ubuntu 16.04.

To create a new docker image with enviroment for this project, please view ``` build_image_instruction.txt```


## Cartpole demo

Stay in the docker container, and now we can conduct the experiment with the instruction as follow:
1. play cartpole game and view scores:
```
cd ReinforcementLearning/
```
```
python3.5 dqn_cartpole.py
```
2. view saved video in Jupyter notebook

Lauch Jupyter notebook:

* if you use docker image to jupyter notebook:
```
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```
* if you build environment and run this project in AMI:

```
jupyter notebook --no-browser --port=8888 --ip="0.0.0.0" --NotebookApp.password=sha1:8739d1c94de7:83fb85ccaa70ba0666e146e9502634f97369dc98 --NotebookApp.tornado_settings="{'headers':{'Content-Security-Policy':'frame-ancestors * \'self\''}}"
```
To open Jupyter notebook, copy the website address showed in the terminal after running the above command.

You can review the video by entering the folder named **cartpole_video** or use  **view_cartpole_video.ipynb**.

## Important Links
* [QuantUniversity](http://www.quantuniversity.com/)
