# Arc for Anyones usage in FRI

This repository is designed for people to freely modify rewards and train models in any way they prefer using **Stable Baselines 3** in robosuite. If any question on how to use please contact the author's since we'd be happy to help and explain.  
## Fri Requirements 
You need Sudo for the installation, since UT has no default enviorments on the lab machines that has all dependencies. You are recomended to dual boot and work on ubuntu on your own machine since you have more freedom and its not that computationally heavy. Use a vertiual enviorment if you want to work on the FRI lab machines. If working on your own machine you have free rain to not use vertual enviorments but it's still recomend that you make a vitual enviorment.

## Problems we had 
- The installation tends to be half the battle
- There is a huge issue with robosuite on windows, there is documentation on their website to fix it and you'll have to do it if you want to work on a windows machine
- Please refrain from using magic numbers, they are temp fixes that **WILL** fail given enough time

## Current Issues that need to be addressed for future use
- The requirements.txt is basically bloatware since they have a lot of things not in use
- There are comments that are no longer Relavent 
- better tutorial needs to be added 



## Features

- Easily change the reward structure in Stable baseline 3
- Flexibility to train models with various configurations.
- Built using Stable Baselines 3 for reinforcement learning.

## Requirements

- Python 3 or higher 
- Stable Baselines 3 (`pip install stable-baselines3`)
- Other dependencies as specified in `requirements.txt`.
-

## Usage

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
