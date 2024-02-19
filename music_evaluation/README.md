# Music Evaluation

Adapted from this GitHub repository [mgeval](https://github.com/RichardYang40148/mgeval)

Deleted all packages in mgeval using python 2.

# Packages:
scipy, numpy, seaborn, pretty_midi, scikit-learn, python 3

# Usage:

```
python music_evaluator.py --set1dir /path/to/your/ground-truth/data/ --set2dir /path/to/your/generated-sample/ --outdir output-dir --num_sample number-of-samples-to-evaluate
```


# Output
All outputs are in the output-dir directory in the current folder, including plots and statistics.txt. 

Check out the result folder for an example.

You can either run music_evaluator.py or demo.ipynb for evaluation.