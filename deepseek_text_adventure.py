import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import sqlite3
import numpy as np
import psutil
import requests

# Load tokenizer and model for DeepSeek
model_name = "deepseek-ai/DeepSeek-R1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# (Full code for the main game as displayed in the previous output)
if __name__ == "__main__":
    main()
