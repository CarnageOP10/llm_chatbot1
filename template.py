import os
from pathlib import Path

list_of_files = [
    "src/__init__.py",
    "src/main.py",
    "src/utils.py",
    "experiment",
    "src/prompts.py",
    "app.py"
]

for filepath in list_of_files:
   filepath = Path(filepath)
   filedir, filename = os.path.split(filepath)

   if filedir !="":
      os.makedirs(filedir, exist_ok=True)

   if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
      with open(filepath, 'w') as f:
         pass

   else:
        print(f"File already exists: {filepath}")
        continue