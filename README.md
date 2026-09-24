# ATL Flight Disruption Model

Predicting departure delays, arrival delays, and cancellations for flights through Atlanta Hartsfield-Jackson (ATL) using 2024 U.S. flight data joined with daily Atlanta weather.

This was a CS210 group project built by me and 2 other teammates. We cleaned and joined the data, explored delay and cancellation patterns, and trained three random forest classifiers that estimate the risk of each type of disruption for a given flight. It ends with a small interactive demo where you enter a flight and a weather forecast and get back a probability for each outcome.

The models are meant as risk estimators, not exact predictors. Delays depend heavily on things the data doesn't capture (crew schedules, late inbound aircraft, air traffic control), so the goal is to rank flights by risk rather than call every delay correctly.

## Results

All three final models are random forests (200 trees, max depth 12, balanced class weights), evaluated on a held-out 20% test set of 136,751 flights. A delay means more than 15 minutes late. Each model uses a tuned decision threshold instead of the default 0.5.

| Model | ROC-AUC | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|
| Departure delay | 0.692 | 0.30 | 0.56 | 0.39 | 0.512 |
| Arrival delay | 0.686 | 0.31 | 0.51 | 0.38 | 0.519 |
| Cancellation | 0.899 | 0.07 | 0.71 | 0.12 | 0.506 |

Precision, recall, and F1 are for the positive class (delayed or cancelled).

**What the numbers mean:**

- **Delays** are hard to predict from schedule and daily weather alone. The models catch about half of real delays, and roughly 3 in 10 flights they flag actually end up delayed. That's modest, but well above guessing, since only about 18% of flights in the data are delayed.
- **Cancellations** are where the model is strongest at ranking risk (ROC-AUC 0.90) and catches 71% of them. Precision is low because only about 1% of flights are cancelled, so even a good model flags many flights that end up flying.

### How we got there

We started simple and added complexity only when it fixed a specific problem. Every step below comes from `cs210ATL_LearningModels.ipynb`.

| Iteration | Departure delay ROC-AUC | What we learned |
|---|---|---|
| 1. Logistic regression | 0.667 | 82% accuracy, but only because it predicts almost nothing as delayed (1% recall) |
| 2. Decision tree | 0.592 (no depth limit), 0.707 (depth 10) | With no depth limit it memorizes the training data (99% train vs 76% test accuracy) |
| 3. Random forest, untuned | 0.690 | Better ranking, but still predicts zero delays because of the class imbalance |
| 4. Random forest, balanced | 0.692 | Balanced class weights raise delay recall from 0% to 62% |
| 5. Final model, tuned threshold | 0.692 | A tuned threshold trades a little recall for better precision and accuracy |

The main lesson: accuracy was misleading here. A model that says "never delayed" is 82% accurate and useless, so we judged every model on recall, F1, and ROC-AUC instead.

## Notebooks

| Notebook | What it does |
|---|---|
| `cs210ATL_cleanup.ipynb` | Filters the 1.31 GB national flight file down to ATL flights, cleans it, joins daily weather, creates the delay, cancellation, and weather-severity columns, and writes the final CSV |
| `ATL_Flight_Weather_Visuals.ipynb` | Charts delay patterns by airline, departure hour, and weather severity (including how much longer delays run in severe weather), plus confusion matrices for the three final models |
| `cs210ATL_LearningModels.ipynb` | Trains and compares each model iteration, builds the three final models, saves them, and runs the prediction demo |
| `ATL_Model_Progression.ipynb` | A heavily commented walkthrough of the modeling work, from a baseline logistic regression through random forests with and without class balancing, explaining why each change was made. Also includes a batch of test flights and a Low / Moderate / High risk label |

## Features

The final models use nine inputs:

- **Flight:** month, airline, origin airport, destination airport, scheduled departure hour, distance
- **Weather:** total precipitation (inches), average wind speed (mph), and a severe-weather flag (set when precipitation is over 1.25 in or average wind is at least 15 mph)

Airline and airports are one-hot encoded; the rest are used as-is.

## Setup

The project was built in Google Colab and reads its files from Google Drive.

1. **Download the data and models** from the [v1.0 release](https://github.com/arin-deshpande/ATL-Flight-Disruption-Model/releases/tag/v1.0):
   - [`atl_joined_weather_flights_2024.zip`](https://github.com/arin-deshpande/ATL-Flight-Disruption-Model/releases/download/v1.0/atl_joined_weather_flights_2024.zip) (12 MB): the cleaned, joined dataset
   - [`flight_models.zip`](https://github.com/arin-deshpande/ATL-Flight-Disruption-Model/releases/download/v1.0/flight_models.zip) (26 MB): the trained models, needed for the prediction demo and the confusion-matrix charts
2. **Unzip both** and upload the files to the top level of your Google Drive (`/content/drive/MyDrive/`):
   - `atl_joined_weather_flights_2024.csv`
   - `flight_models.pkl`
3. **Open a notebook in Colab** and run the Google Drive mount cell at the top.
4. **Run the cells in order.**

You only need the raw Kaggle and weather files if you want to re-run `cs210ATL_cleanup.ipynb` from scratch. The other notebooks start from the cleaned CSV.

## Running the prediction demo

At the end of `cs210ATL_LearningModels.ipynb`:

1. Run the **Model User Interface** cell to load `flight_models.pkl` and define the prediction helpers. You can skip the training cells.
2. Run the three **Examples of Predictions** cells (clear, moderate, and severe weather), or
3. Run the **User Inputs (UI)** cell and enter your own flight: month, airline code, origin, destination, scheduled departure time (HHMM), distance, forecast average wind speed, and forecast precipitation.

Example output for a June Delta flight from ATL to LAX at 3 PM, with 5 mph wind and 0.5 in of rain:

```
Departure Delay Probability: 57.36% (Moderate Risk)
Arrival Delay Probability:   55.81% (Moderate Risk)
Cancellation Probability:    35.25% (Low Risk)

Predicted Outcomes
Departure Delay: Yes
Arrival Delay:   Yes
Cancellation:    No
```

Use valid airline and airport codes (for example `DL`, `AA`, `WN`; `ATL`, `LAX`, `EWR`). A code the models never saw in training won't cause an error, but the models treat it as unknown, so the prediction is less reliable.

## Data sources

- **Flights:** [Flight Data 2024](https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024) on Kaggle (U.S. Bureau of Transportation Statistics on-time data)
- **Weather:** [Weather Underground daily history for ATL](https://www.wunderground.com/history/monthly/us/ga/atlanta/KATL/date/2024-1), January through December 2024

The raw flight file is 1.31 GB and too large to include here, even zipped. The cleaned, joined dataset is in the release.

## Limitations

- **Weather is daily, not hourly.** A 7 AM flight and a 9 PM flight on the same day get the same weather, so predictions are same-day risk estimates, not forecasts for a specific departure time.
- **Only ATL weather is used.** Weather at the other airport isn't included, even though a storm at the destination can delay a flight just as easily.
- **No knock-on effects.** The models don't know about late inbound aircraft, crew timing, or air traffic control delays, which cause many real delays.
- **Cancellations are rare** (about 1% of flights), so the cancellation model is better at ranking risk than at saying yes or no.
- **One year of data.** Everything is trained and tested on 2024, so the results may not carry over to other years.

## License

MIT
