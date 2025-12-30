# Qwen2VL-fine-tuning-and-related-agents
This is a presentation of the fine-tuning projects and agent results for qwen2vl-2B and qwen2.5vl-7B. 
## Introduction
Given an image containing a car, run "main.py", enter "your_figure_path.jpg" in the prompt to select the nearest car / the leftmost car / the rightmost car. Then, the following image will be obtained.
<div style="display: flex; gap: 20px;">
  <img width="400" alt="功能截图1" src="https://github.com/user-attachments/assets/31d53487-3ab4-461b-97d8-40312a992132" />
  <img width="400" alt="功能截图2" src="https://github.com/user-attachments/assets/4c2119d0-43ca-4da2-8225-6f60a7160458" />
</div>
example2
<div style="display: flex; gap: 20px; justify-content: center;">
  <img width="400" alt="系统架构图" src="https://github.com/user-attachments/assets/6126b094-8ceb-4ab2-a0a2-c17d88aab9e2" />
  <img width="400" alt="功能演示图" src="https://github.com/user-attachments/assets/eab0dae8-d111-4b39-8d3c-2254971bac01" />
</div>
When you ask other questions, the agent will respond using the un-tuned qwen2.5VL model.



## Requirements
```
torch>=2.0.0
transformers>=4.37.0
peft>=0.7.0
accelerate>=0.21.0
```
## Installation
```
git clone https://github.com/yutianyuan-6/Qwen2VL-and-related-agents.git
cd Qwen2VL-and-related-agents
pip install -r requirements.txt
```
## Usage
1、download model to ./models/
```
pip install huggingface-hub
# Qwen2.5-VL-7B
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./Qwen/Qwen2.5-VL-7B-Instruct
# Qwen2-VL-2B
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir ./Qwen/Qwen2-VL-2B-Instruct
```
2、Place the query image in the root directory.

3、Run main.py




