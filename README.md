# Project Overview
This project, titled DATA3888_image_4, focuses on cell image analysis and model development for classification tasks. The project involves several steps, including data collection, initial data analysis, data cleaning and preprocessing, model development and evaluation, integrating model outputs with other omics data, and deployment of Shiny Apps.

# Reproduce report 
Run the "DATA3888" files with all the files and folder in Github downloaded in the same hierachical file structure

# Deployment of Shiny Apps
To deploy the Shiny Apps successfully, ensure that you have the "Shiny_Apps.Rmd" files in the same directory as the "shiny_output" folder. The "shiny_output" folder contains the output files of the eight models. This setup will enable the Shiny Apps to access and utilize the model outputs.

# Generate csv files used for shiny apps 
Run the "Data Integration.Rmd" files within the zip file 

# Model Development
Codes to run each models are stored in the "Model_codes" folder which contains 8 ipynb format files, each corresponding to a trained model. 
Read the provided ipynb files to access the code and documentation for model development and evaluation. These files contain the necessary information to understand the models and their performance. Each of this files will create a confusion matrix with hierachical clustering, an output containing prediction details and the test accyuracy and test loss in csv file format, and a training process csv, and grad-cam/SAAV depending on whether the model is transformers or cnn. The output files are saved in a folder named "Model_result"

# Integrating Model Outputs with omics data and 
Run the "Data_Integration.Rmd" file to integrate output from model to the final usable output for Shiny Apps deployment. You will need to have the "Model_result" folder as well as other folders in this Github


For more detailed instructions and code explanations, refer to the individual files and documentation provided within this repository.
