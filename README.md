## Project Overview
This project, titled DATA3888_image_4, focuses on cell image analysis and model development for classification tasks. The project involves several steps, including data collection, initial data analysis, data cleaning and preprocessing, model development and evaluation, integrating model outputs with other omics data, and deployment of Shiny Apps.

## To Reproduce report ONLY
Run the "DATAT3888.Rmd" files with all the files and folder in Github downloaded in the same hierachical file structure

## To deploy Shiny Apps ONLY 
To deploy the Shiny Apps successfully, ensure that you have the "Shiny_Apps.Rmd" files in the same directory as the "shiny_output" folder. The "shiny_output" folder contains the output files of the eight models. This setup will enable the Shiny Apps to access and utilize the model outputs.


# NOTE FOR BELOW: To completely reproduce the whole project, in addition to all files in Github, you will need to obtain the tif.file from https://canvas.sydney.edu.au/courses/47736/pages/biotechnology-data-science-image-analysis, under Resources : Biotechnology data bundle ~400MB. The file path to the tif file : "Biotechnology/data_raw/morphology_focus.tif" and move it to the same file path as where you found it within our "Biotechnology" folder in Github. 

## To generate all cell images 
- Head to "Biotechnology/script/DATA3888_Biotechnology_generateImages_2023.Rmd" and follow the instructions to run the code. A folder with 28 subfolders in it will be created with different number of images in each folder. 

## To clean data and preprocessing data
- Head to "Data_Preprocessing.Rmd" to run the code, and 4 files will be created , which consist of 2 files for train and test for raw data, and 2 files for train and test for clean data

## Model Development
Codes to run each models are stored in the "Model_codes" folder which contains 8 ipynb format files, each corresponding to a trained model. This code will require either the train_raw and test_raw folder or the train_raw and test_raw folder depending on the model you intend to run. Read the provided ipynb files to access the code and documentation for model development and evaluation. These files contain the necessary information to understand the models and their performance. Each of this files will create a confusion matrix with hierachical clustering, an output containing prediction details and the test accuracy and test loss in csv file format, and a training process csv, and grad-cam/SAAV depending on whether the model is transformers or cnn. The output files are saved in a folder named "Model_result"

## Integrate data for Shiny Apps deployment 
Run the "Data_Integration.Rmd" file to integrate output from model to the final usable output for Shiny Apps deployment. You will need to have the "Model_result" folder as well as other folders in this Github


For more detailed instructions and code explanations, refer to the individual files and documentation provided within this repository.
