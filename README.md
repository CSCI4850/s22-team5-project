# s22-team5-project

This project was coded in python and uses the OpenAI gym as an environment.
If you would like to run our demo follow these steps as listed:

## Demo just using PPO
1. Download this repo in a location on your computer where you can reference later.
2. Go to [this link](https://towardsdatascience.com/how-to-set-up-anaconda-and-jupyter-notebook-the-right-way-de3b7623ea4a) to install anaconda and get jupyter-lab installed.
3. Then within jupyter lab, navigate to this repository on your computer and enter the demo folder
4. Double click one of the files ending in .ipynb
5. Run each cell one by one by hitting "shift + enter" on your keyboard

## Demo with the Autoencoder
1. 

If you would like to take our main code and run it elsewhere, make sure that you have at least python 3.5 installed on your computer and have gym installed using pip.
For instructions on how to do that, check out [this link](https://gym.openai.com/docs/).
Then check out the [vizdoom github](https://github.com/mwydmuch/ViZDoom) and install vizdoom for OpenAI gym.

To train the autoencoder, run
	pythom ./gym/doom_trainer.py
