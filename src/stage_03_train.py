from src.utils.all_utils import read_yaml, create_directory,save_local_df
import argparse
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import joblib

def split_and_save(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    split_data_dir = config["artifacts"]["split_data_dir"]
    train_data_filename = config["artifacts"]["train"]
    train_data_path = os.path.join(artifacts_dir, split_data_dir, train_data_filename )
    train_data = pd.read_csv(train_data_path)

    train_y = train_data["quality"]
    train_x = train_data.drop("quality",axis = 1)
    l1_ratio = params["model_params"]["ElasticNet"]["l1_ratio"]
    alpha = params["model_params"]["ElasticNet"]["alpha"]
    random_state = params["base"]["random_state"]
    lr = ElasticNet(alpha  = alpha, l1_ratio = l1_ratio , random_state = random_state)
    lr.fit(train_x,train_y)

    model_dir = config["artifacts"]["model_dir"]
    model_file_name = config["artifacts"]["model_filename"]
    model_fol = os.path.join(artifacts_dir, model_dir)
    create_directory([model_fol])
    model_path = os.path.join(model_fol, model_file_name)
    joblib.dump(lr, model_path)
if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config" , "-c", default = "config/config.yaml")
    args.add_argument("--params" , "-p", default = "params.yaml")

    parsed_args = args.parse_args()
    split_and_save(config_path = parsed_args.config, params_path = parsed_args.params )
