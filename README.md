This repository contains a variety of programs used in the paper **What Makes Freezing Layers in Deep Neural Networks Effective? A Linear Separability Perspective**. In this work, we investigate what causes freezing layers in deep neural networks to cause networks to (1) generalize better and (2) learn more quickly. We focus on Cover's theorem, commonly cited as driving the effects of freezing layers, which relates to increased linear separability of data coming from nonlinear random transformations. 

# Setup
### Step 1: Set Up a Conda Environment
It is recommended to use Python 3.12.1 for this project to ensure package compatability. Otherwise, additional effort will need to be done to resolve dependency issues. To set and activate up a conda environment, run the following commands:

```bash
conda create -n freezing python=3.12.1
conda activate freezing
```

### Step 2: Run the Setup Script
After setting up the conda environment and installing the necessary dependencies, navigate to the root directory of this repository and run the following command:

```bash
pip install -v -e .
```
This command will install all the required packages listed in the requirements.txt file.

# Usage
Each experiment can be replicated using the provided Python scripts. In addition to providing the scripts to run the analysis from scratch, we provide the results of our experiments in the `Results` folder. 

# Paper and Citation
This paper is currently under review. As such, we refrain from adding a citation or link to the draft to avoid compromising the blind review process. We will provide both once the paper is accepted. 
