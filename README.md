# Project Overview
This project, titled DATA3888_image_4, focuses on cell image analysis and model development for classification tasks. The project involves several steps, including data collection, initial data analysis, data cleaning and preprocessing, model development and evaluation, integrating model outputs with other omics data, and deployment of Shiny Apps.

# Reproduce report 
Download the "Whole Project" zip file, extract all files and run the DATAT3888 RMD files 

# Deployment of Shiny Apps
To deploy the Shiny Apps successfully, ensure that you have the "Shiny_Apps.Rmd" files in the same directory as the "shiny_output" folder. The "shiny_output" folder contains the output files of the eight models. This setup will enable the Shiny Apps to access and utilize the model outputs.

# Generate csv files used for shiny apps 
Run the "Data Integration.Rmd" files within the zip file 







# Generating Cell Images
To generate cell images, navigate to `Biotechnology/scripts/DATA3888_Biotechnology_generateImages_2023.Rmd` to extract the images. This script will create cell images and organise them in a hierarchical folder structure.

# Data Cleaning and Preprocessing & Splitting data into Train-validation and test in 80:20 ratio
Using "Data_Preprocessing.Rmd", and make sure this file is in same directory path as the Biotechnology folder

# Model Development
Codes to run each models are stored in the "Model_codes" folder which contains 8 ipynb format files, each corresponding to a trained model. 
Read the provided ipynb files to access the code and documentation for model development and evaluation. These files contain the necessary information to understand the models and their performance. Each of this files will create a confusion matrix with hierachical clustering, an output containing prediction details and the test accyuracy and test loss in csv file format, and a training process csv, and grad-cam/SAAV depending on whether the model is transformers or cnn. The output files are saved in a folder named "Model_result" under 

# Evaluate 
Read the "Model_evaluation.Rmd" files to read the training process history for each model and the accuracy plots comparison for all 8 models. 

# Integrating Model Outputs with omics data and 





# Reproduce report 
Download the "Whole Project" zip file, extract all files and run the DATAT3888 RMD files 



# Deployment of Shiny Apps
To deploy the Shiny Apps successfully, ensure that you have the "Shiny_Apps.Rmd" files in the same directory as the "shiny_output" folder. The "shiny_output" folder contains the output files of the eight models. This setup will enable the Shiny Apps to access and utilize the model outputs.

For more detailed instructions and code explanations, refer to the individual files and documentation provided within this repository.
