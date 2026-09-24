# ATL-Flight-Disruption-Model

This repository shows how to clean, explore, visualize, and model U.S. flight data in order to predict flight delays and cancellations. The project uses supervised machine learning models to estimate whether a flight is likely to have a departure delay, arrival delay, or cancellation based on available flight and airport-related features.


USAGE:

The cleanup notebook contains code for loading the flight dataset, removing unnecessary columns, handling missing values, preparing categorical variables, and creating target variables for delay and cancellation prediction.

The visualization notebook contains code for exploring delay and cancellation patterns across airlines, airports, months, days, and other flight-related features. It includes charts and summary statistics to better understand trends in the data before modeling.

The model notebook contains code for training and testing machine learning models for departure delay, arrival delay, and cancellation prediction. The models use classification approaches and evaluate results using metrics such as precision, recall, F1-score, ROC-AUC, and decision thresholds.

The final models are intended to work as risk-estimation tools rather than perfect prediction systems. The delay models show moderate performance, while the cancellation model ranks cancellation risk well but struggles with precision because cancellations are rare in the dataset.

Link to Flight dataset: https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024?select=flight_data_2024.csv Weather Data Set: https://www.wunderground.com/history/monthly/us/ga/atlanta/KATL/date/2024-1

Cleaned Data Set: atl_joined_weather_flights_2024 ZIP is our cleaned data ([download from the v1.0 release](https://github.com/arin-deshpande/ATL-Flight-Disruption-Model/releases/download/v1.0/atl_joined_weather_flights_2024.zip)) This is after running our orginal 1.31gb fight file through our cleanup code. Unable to upload our uncleaned datasets due to them being to large even being ZIPed


HOW TO RUN OUR Visuals Notebook

Make sure you install our dataset atl_joined_weather_flights_2024 and unZIP the file and place it into your Google Drive dierectory : /content/drive/MyDrive/ Also make sure you install our pickeled final model in (flight_models.zip) and also place it in Google Drive Run our Mount Google Drive code, and you should be able to run all our graphs


HOW TO RUN OUR MODELS

Make sure you install our dataset atl_joined_weather_flights_2024 and unZIP the file and place it into your Google Drive dierectory : /content/drive/MyDrive/ Run our Mount Google Drive code, then run our Training Model Data code section, after you can either go through our models and train them seperately if you'd like our go all the way to our final model and train it.


HOW TO RUN OUR UI At the end of learningModels NOTEBOOK

Make sure you have our pickeled final model in (flight_models.zip) installed and also place it in Google Drive RUN the Model User Interface code, then you should be able to run all the Example predictions

At the end of the Notebook there is "RUN this code box for User Inputs (UI)" run it and you can input your own flight and weather data to predict The flights delays/cancellations

Setup Instructions more indepth
This project was built in Google Colab.


Step 1: Upload files to Google Drive
Upload the following files to:

/content/drive/MyDrive/

Required files:

- [atl_joined_weather_flights_2024.zip](https://github.com/arin-deshpande/ATL-Flight-Disruption-Model/releases/download/v1.0/atl_joined_weather_flights_2024.zip) (12 MB)
- [flight_models.zip](https://github.com/arin-deshpande/ATL-Flight-Disruption-Model/releases/download/v1.0/flight_models.zip) (26 MB, needed for the final prediction demo)

Both are attached to the [v1.0 release](https://github.com/arin-deshpande/ATL-Flight-Disruption-Model/releases/tag/v1.0).
Then unzip them in Google Drive if needed so the notebooks can access:

atl_joined_weather_flights_2024.csv
flight_models.pkl


Step 2: Mount Google Drive
At the start of each notebook, run the Google Drive mount cell so Colab can access the files stored in Drive.

How to Run the Notebooks
Visuals Notebook
Notebook: ATL_Flight_Weather_Visuals.ipynb

This notebook uses the cleaned joined dataset to:

explore delay and cancellation patterns
compare airlines
compare weather severity levels
analyze delay rates by departure hour
generate the graphs used in the report
To run it:

make sure atl_joined_weather_flights_2024.csv is in Google Drive and flight_models.pkl
run the mount Google Drive cell
run the notebook cells in order
Models Notebook
Notebook: cs210ATL_Learning_Models.ipynb

This notebook:

loads the cleaned merged dataset
trains several machine learning model attempts
compares model outputs
builds final models for:
departure delay
arrival delay
cancellation
includes a small prediction demo / user interface at the end
To run it:

make sure atl_joined_weather_flights_2024.csv is in Google Drive
run the mount Google Drive cell
run the model training cells in order
If you want to use the saved final models instead of retraining, place:

flight_models.pkl
in Google Drive and run the model-loading / UI section.

How to Run the Final Prediction Demo
At the end of the models notebook, there is a user input section.

To use it:

make sure flight_models.pkl is in Google Drive
run the code cell that loads the pickled models
run the prediction helper function cells
run the example prediction cells or the user input cell
The demo allows you to enter:

month
airline code
origin airport
destination airport
scheduled departure time
distance
forecast average wind speed
forecast precipitation total
The model then returns:

departure delay probability
arrival delay probability
cancellation probability
and the threshold-based yes/no prediction for each one.

Notes
Valid airline and airport codes matter for the prediction demo.
The weather data is daily, not hourly, so the predictions should be interpreted as same-day disruption risk estimates rather than exact minute-by-minute forecasts.
The cancellation model ranks cancellation risk better than it classifies exact cancellations, since cancellations are rare in the dataset.
