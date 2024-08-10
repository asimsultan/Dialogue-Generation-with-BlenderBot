
# Dialogue Generation with BlenderBot

Welcome to the Dialogue Generation with BlenderBot project! This project focuses on generating dialogues using the BlenderBot model.

## Introduction

Dialogue generation involves creating conversational responses based on input text. In this project, we leverage BlenderBot to generate responses using a dataset of dialogues.

## Dataset

For this project, we will use a custom dataset of input texts and their corresponding responses. You can create your own dataset and place it in the `data/dialogue_data.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Datasets
- Pandas

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/asimsultan/dialogue_generation_blenderbot.git
cd dialogue_generation_blenderbot

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes input texts and their corresponding responses. Place these files in the data/ directory.
# The data should be in a CSV file with two columns: input_text and response.

# To fine-tune the BlenderBot model for dialogue generation, run the following command:
python scripts/train.py --data_path data/dialogue_data.csv

# To evaluate the performance of the fine-tuned model, run:
python scripts/evaluate.py --model_path models/ --data_path data/dialogue_data.csv
