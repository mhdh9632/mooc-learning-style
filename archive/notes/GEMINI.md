# Project Context: KDD Cup 2015 MOOC Dropout Prediction

## Project Overview
This is a Data Science and Machine Learning project focused on the KDD Cup 2015 competition. The main objective is to predict student dropouts in Massive Open Online Courses (MOOCs) based on user interaction logs. The project relies on analyzing large datasets containing logs of user events, course objects, and course dates to build predictive models.

**Main Technologies:**
*   Python
*   Jupyter Notebook (Google Colab environment preferred)
*   Pandas (for data manipulation)
*   Kaggle API (for data downloading)

## Directory Structure & Key Files
*   **`Project.ipynb`**: The core Jupyter Notebook of the project. It contains the code for downloading the dataset via the Kaggle API, loading it into Pandas DataFrames, and performing initial data exploration and presumably model training.
*   **`train.csv.zip` / `train.csv/`**: The training dataset files, including interaction logs, course objects, dates, and the ground truth labels for dropouts.
*   **`test.csv`**: The testing dataset used for evaluating the model's predictions.

## Building and Running
1.  **Environment Setup:** This project is designed to run within a Jupyter Notebook environment. It is currently configured with snippets specific to **Google Colab** (e.g., mounting Google Drive).
2.  **Dependencies:** Ensure you have the necessary Python libraries installed, particularly `pandas` and the `kaggle` CLI tool if you intend to re-download the data.
    ```bash
    # TODO: Create a requirements.txt if more dependencies are added. For now, standard DS libraries are assumed.
    pip install pandas kaggle
    ```
3.  **Execution:**
    *   Open `Project.ipynb` in your Jupyter environment or Google Colab.
    *   Run the cells sequentially. Note that the first cell requires Kaggle API credentials (`kaggle.json`) to be placed in the appropriate directory (e.g., Google Drive if using Colab) to download the dataset.
    *   If running locally, ensure the data files (`test.csv`, `train.csv.zip`) are in the correct relative paths as referenced in the notebook.

## Development Conventions
*   **Data Exploration:** The notebook follows a structured approach to data exploration, clearly separating the loading and displaying of different data frames (`log_train`, `object_df`, `date_df`, `truth_df`) with distinct headers and sample outputs.
*   **Environment:** The code assumes a cloud-based environment (Google Colab) for handling the large dataset sizes effectively, as evidenced by the `google.colab` imports. If developing locally, ensure your machine has sufficient memory to handle datasets in the gigabyte range.
