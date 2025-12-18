from datasets import load_dataset

ds = load_dataset("openai_humaneval")

ds.save_to_disk("./workflow_data/openai_humaneval")