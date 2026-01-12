from src.experiment_environment import ExperimentEnvironment
import json


#Get experiment config
with open('config.json', 'r') as f:
    config = json.load(f)
#Instantiate an experiment with parameters in the config
experiment = ExperimentEnvironment(config=config)
experiment.run()