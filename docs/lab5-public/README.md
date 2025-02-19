
## Robot Control 24/25 - Homework 1.

In order to run code for a particular task, run

```bash
pip install -r requirements.txt
python script.py <task_number>
```
What code belongs to which task can be backtracked from the main function located at the end of the file.

I used the following command to find matching points between images:

```bash
python .\match_pairs.py --input_dir .\my_images --output_dir .\output_dir --input_pairs my_pairs.txt --viz --resize -1
```

"undistorted" directory contains undistored images from stiching directory (task1) and partial outputs for multiple image stitching (task 7)

"produced_images" directory contains images required as tasks solutions (tasks 5,6,7)

Python version: Python 3.10.3