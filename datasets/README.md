# Creating data representation for symbolic music

This directory contains instructions and scripts for creating training dataset. 
Notice that you do not need to prepare dataset if you want to generate music and have target rule labels in mind. 
You will need to prepare dataset if you want to train a model, or you want to extract rule labels from existing music excerpts.

We train our diffusion model on three datasets: [Maestro](https://magenta.tensorflow.org/datasets/maestro#v300) (classical piano performance), Muscore (crawled from the Muscore website), and Pop ([Pop1k7](https://drive.google.com/file/d/1qw_tVUntblIg4lW16vbpjLXVndkVtgDe/view) and [Pop909](https://github.com/music-x-lab/POP909-Dataset)).
You can download the data and put the midi files into the corresponding folder: `maestro`, `muscore` and `pop`.

Then run `piano_roll_all.py` to create piano roll excerpts from the dataset.

The above script creates piano roll excerpts of 1.28s. 
To create music of 10.24s for training, return to the main folder and run `rearrange_pr_data.py` to concat shorter piano rolls to longer ones.
The processed data will be saved in `datasets/all-len-40-gap-16-no-empty` by default, and along with the data, there will be two csv files:
`all-len-40-gap-16-no-empty_train.csv` and `all-len-40-gap-16-no-empty_test.csv` that list the filenames.

If you want to extract rule label from the piano rolls and condition on a specific dataset, you need to create csv file for each dataset using:
```
python filter_class.py --file_path all-len-40-gap-16-no-empty_test.csv --class_label <class label>
```
