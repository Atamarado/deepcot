import os
import re
import pandas as pd
import pickle

PATTERN = r'saved_models/(?P<model>.+?)_(?P<num_layers>\d+)_layers_(?P<num_landmarks>\d+)_landmarks_(?P<fit_layer_epochs>-?[\d,\s]+)-_(?P<freeze_weights>\d+)_feature_(?P<feature>.+?)_seeds_(?P<seed>\d+)'

if __name__ == '__main__':

    CONFIG_COLS = ["model", "num_layers", "fit_layer_epochs", "freeze_weights", "num_landmarks", "feature"]

    configs = []
    mAP = []
    mcAP = []
    runtimes = []

    listdir = os.listdir('saved_models')

    for directory in os.listdir('saved_models'):
        path = directory + "/log_tran&test"+str(-1)+".txt"
        if not os.path.exists(path):
            continue

        try:
            with open(path, 'rb') as f:
                res_dict = pickle.load(f)
        except:
            continue

        # mAP.append(res_dict['test_mAP']*100)
        # mcAP.append(res_dict['test_mcAP']*100)
        runtimes.append(res_dict['test_runtime'])

        config = vars(res_dict['config'])
        config = {k: config[k] for k in CONFIG_COLS+["seed"]}
        config['fit_layer_epochs'] = tuple(config['fit_layer_epochs']) if config['model'] in ['nystromformer', 'continual_nystrom'] else tuple()

        match = re.match(PATTERN, directory)
        if match:
            config['freeze_weights'] = match.group('freeze_weights')

        configs.append(config)

    res_df = pd.DataFrame.from_records(configs)
    # res_df["mAP"] = mAP
    # res_df["mcAP"] = mcAP
    res_df["runtime"] = runtimes

    res_df = res_df.groupby(CONFIG_COLS).agg({
        # 'mAP': ['mean', 'std'],
        # 'mcAP': ['mean', 'std'],
        'runtime': ['mean', 'std'],
        'seed': ['count'],
    }).reset_index()
    res_df = res_df[res_df['num_layers'] == 2]

    base_runtime = res_df.iloc[0]['runtime']['mean']
    res_df['rel_runtime'] = base_runtime / res_df['runtime']['mean']

    res_df.to_csv('results_oad.csv')


