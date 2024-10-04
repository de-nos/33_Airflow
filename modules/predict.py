import glob
import json
import os

import dill
import pandas as pd


path = os.environ.get('PROJECT_PATH', '.')

def load_n_pred(model):
    file_list = glob.glob(f'{path}/data/test/*.json')
    print(f'Found {len(file_list)} .json files:')
    df_res = pd.DataFrame(columns=['id', 'price_pred'])
    index = 0
    for filename in file_list:
        # print(f'{' '*4}{filename}:', end='')
        with open(filename) as f:
            form = json.load(f)
            df = pd.DataFrame([form])
            y = model.predict(df)
            df_res.loc[index] = [form['id'], y[0]]
            index += 1
            # print(f"{' '*2}id={form['id']}: {y[0]}")
    return df_res

def predict():
    model_filename = f'{path}/data/models/cars_pipe.pkl'
    with open(model_filename, 'rb') as file:
        model = dill.load(file)
    df = load_n_pred(model)
    print(df)
    pred_csv_filename = f'{path}/data/predictions/pred.csv'
    df.to_csv(pred_csv_filename)
    print(f'Predictions are saved in "{pred_csv_filename}"')


if __name__ == '__main__':
    predict()
