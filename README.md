# CS325 Final Project -- Marathon Finish Time Predictor
    Cody Smith, Thomas Youse, John Cordwell, James Kurschner

A team ML project predicting a runner's marathon finish time given prior training data. 

## Setup 
    pip install -r requirements.txt

## Data 
    Place 'train.csv' and 'test.csv' in the 'csv/' directory before running.
    Expected columns: personal_best_minutes, weekly_mileage_miles, vo2_max, 
                    training_program, injury_severity, gender, marathon_weather,
                    course_difficulty, actual_finish_time_minutes, etc..

## Usage
    python3 src/run_model.py 
    * Allow 5-15 minutes for the model to complete its run depending on machine specs *

## Output 
    Results are saved to results/:
    - feature_importances.png 
    - actual_vs_predicted.png
    - permutation_feature_importance.png

