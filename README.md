To reproduce results only the baseline.py and evaluate.py are required. 

**Step 1 :**
Use the baseline.py script to instantiate a model instance, decompose it as per the required budget using the chosen dataset and save the model. (No GPU required)

**Step 2:**
Using the evaluate.py script, employ the LLM-Harness to benchmark the model saved the in previous step. Result will saved in the log path. 

**Comments :**
The implementation of the proposed method is present in layers.py. 
