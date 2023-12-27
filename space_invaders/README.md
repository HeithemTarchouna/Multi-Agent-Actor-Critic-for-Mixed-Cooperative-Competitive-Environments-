

## Introduction
This repository contains a tutorial to show how to train an MADDPG agent on the space invaders atari environment.
. The source code and methodologies are based on the tutorial available at [PettingZoo Tutorial](https://pettingzoo.farama.org/main/tutorials/agilerl/MADDPG/).

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.6 or higher
- pip (Python package manager)

## Installation
To set up your environment to run the code, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone [URL of your GitHub repository]
   cd [repository-name]
   ```

2. **Create and Activate a Virtual Environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Unix or MacOS
   venv\Scripts\activate  # For Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Here's how you can run the code:
- python train.py
- python evaluate.py

## Fixing the ParallelAtariEnv Error
If you encounter the following error:

> AttributeError: 'ParallelAtariEnv' object has no attribute 'np_random'

Please apply this fix in the code:

Locate the line where the error occurs and replace it with:

```python
num_skips = int(np.random.randint(low, high + 1))
```

This should resolve the issue by correctly accessing the `np.random` module.

