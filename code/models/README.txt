DONT RUN preprocessing without downloading the dataset correctly. Skip to 6. if preprocessed data already exists. 

1. Download Dataset at https://visualqa.org/download.html
(There is example.txt files with the correct name in each folder, just delete the txt files and replace with the correct files from the dataset with the correct name)
2. Put annotations in dataset\ann\ --> into train and val annotations respective. Call the files "ann_'whicheverset'.json". (Whicheverset = 'train' or 'val').
3. Put questions in dataset\qst\ --> into test, train and val questions respective. Call the files "multi_qst_'whicheverset'.json" and "open_qst_'whicheverset'.json". (Whicheverset = 'test', 'train' or 'val').
4. Put images in dataset\ann\ --> into test, train and val images respective. (No need to change names.) 
5. Run "preprocessing.ipynb"

Otherwise if the preprocessed data already exists -->
6. Run "training.ipynb"