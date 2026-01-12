from src.experiment_environment import ExperimentEnvironment
import json
import os
#Create cache and output directories if they don't exist
os.makedirs('cache', exist_ok=True)
os.makedirs('output', exist_ok=True)
#Get experiment config
with open('config.json', 'r') as f:
    config = json.load(f)
#Instantiate an experiment with parameters in the config
experiment = ExperimentEnvironment(config=config)
experiment.run()