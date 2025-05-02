# Table of Contents

1. [Dataset Description](#dataset-description-and-cleaning)
2. [EDA (Exploratory Data Analysis)](#eda-exploratory-data-analysis)

## Dataset Description and Cleaning

Dataset URL: [Kaggle - Energy Consumption Prediction](https://www.kaggle.com/datasets/mrsimple07/energy-consumption-prediction/data)

This dataset captures a wide range of features that influence energy consumption. It includes temporal, environmental, and operational attributes relevant to building energy usage.

**Key Features:**
| Feature         |	Description                                                                          |
|-----------------|--------------------------------------------------------------------------------------|
|Timestamp        |	Timestamp for each recorded data point.                                       |
|Temperature      |	Simulated ambient temperature (°C).                                                  |
|Humidity         |	Simulated relative humidity (%).                                                     |
|SquareFootage    |	Size of the building/environment (square feet).                                               |
|Occupancy        |	Number of people present (integer).                                               |
|HVACUsage        |	HVAC system operational status ('On' or 'Off').                                     |
|LightingUsage    |	Lighting system operational status ('On' or 'Off').                                 |
|RenewableEnergy  |	Percentage of energy coming from renewable sources.                                       |
|DayOfWeek        |	Day of the week (categorical).                                                       |
|Holiday          |	Indicates whether the day is a holiday ('Yes' or 'No').                                   |
|EnergyConsumption|	Actual energy consumption value (target variable).                                         |


All data cleaning and preprocessing steps are implemented in the file processing_and_EDA.py

The dataset was first loaded using pandas, and all column names were standardized by converting them to lowercase and replacing spaces with underscores to ensure consistency and ease of reference. The timestamp column was then processed to extract temporal components such as month, day, and hour, allowing for more flexible time-based analysis. Columns like year, minute, and second were removed after confirming they contained no useful variation (e.g., the year was always 2022, and the minute/second fields were constant). Categorical string values throughout the dataset were cleaned by converting them to lowercase.
The dataset’s features were then organized by data type: numerical columns (including occupancy, which was treated as a continuous variable despite being an integer) and categorical columns. Duplicate rows were removed to avoid data leakage or bias in model training, and a check for missing values was performed, none were found, so no imputation was necessary.

## EDA (Exploratory Data Analysis)
