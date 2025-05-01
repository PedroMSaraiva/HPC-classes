#!/bin/bash


# Create datasets directory if it doesn't exist
DATASET_DIR="./dataset"
if [ ! -d "$DATASET_DIR" ]; then
    echo "Creating dataset directory..."
    mkdir -p "$DATASET_DIR"
fi

# Download and extract each dataset
download_and_extract() {
    local name=$1
    local url=$2
    
    echo "Downloading $name..."
    
    # Download the file
    curl -L -o ~/Downloads/$name.zip "$url" --silent
    
    if [ $? -ne 0 ]; then
        echo "Error downloading $name. Skipping..."
        return
    fi
    
    echo "Extracting $name..."
    # Create a subdirectory for each dataset
    mkdir -p "$DATASET_DIR/$name"
    
    # Extract to the dataset directory
    unzip -q ~/Downloads/$name.zip -d "$DATASET_DIR/$name"
    
    # Remove the zip file
    rm ~/Downloads/$name.zip
    
    echo "$name processed successfully."
}

# Process all datasets
echo "Starting download and extraction of all datasets..."

download_and_extract "optdigits" "https://www.kaggle.com/api/v1/datasets/download/hanyfyoussef/optdigits"
download_and_extract "brain-stroke-dataset" "https://www.kaggle.com/api/v1/datasets/download/jillanisofttech/brain-stroke-dataset"
download_and_extract "abalone-dataset" "https://www.kaggle.com/api/v1/datasets/download/rodolfomendes/abalone-dataset"
download_and_extract "predicting-covid19-vaccine-hesitancy" "https://www.kaggle.com/api/v1/datasets/download/cid007/predicting-covid19-vaccine-hesitancy"
download_and_extract "splicejunction-gene-sequences-dataset" "https://www.kaggle.com/api/v1/datasets/download/muhammetvarl/splicejunction-gene-sequences-dataset"
download_and_extract "marketing-data" "https://www.kaggle.com/api/v1/datasets/download/jackdaoud/marketing-data"
download_and_extract "fetal-health-classification" "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/fetal-health-classification"
download_and_extract "yeastcsv" "https://www.kaggle.com/api/v1/datasets/download/samanemami/yeastcsv"
download_and_extract "contraceptive-prevalence-survey" "https://www.kaggle.com/api/v1/datasets/download/joelzcharia/contraceptive-prevalence-survey"
download_and_extract "titanic" "https://www.kaggle.com/api/v1/datasets/download/heptapod/titanic"
download_and_extract "cyber-security-salaries" "https://www.kaggle.com/api/v1/datasets/download/deepcontractor/cyber-security-salaries"
download_and_extract "diabetes-data-set" "https://www.kaggle.com/api/v1/datasets/download/mathchi/diabetes-data-set"
download_and_extract "data-science-job-salaries" "https://www.kaggle.com/api/v1/datasets/download/ruchi798/data-science-job-salaries"
download_and_extract "cirrhosis-prediction-dataset" "https://www.kaggle.com/api/v1/datasets/download/fedesoriano/cirrhosis-prediction-dataset"
download_and_extract "ionosphere" "https://www.kaggle.com/api/v1/datasets/download/jamieleech/ionosphere"
download_and_extract "lung-cancer-dataset-by-staceyinrobert" "https://www.kaggle.com/api/v1/datasets/download/imkrkannan/lung-cancer-dataset-by-staceyinrobert"
download_and_extract "habermans-survival-data-set" "https://www.kaggle.com/api/v1/datasets/download/gilsousa/habermans-survival-data-set"


echo "All datasets have been downloaded, extracted, and organized."
echo "The zip files have been removed and dataset/ has been added to .gitignore." 