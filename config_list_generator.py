data_seed_list = range(5)
model_seed_list = range(5)
model_type_list = ["base_continual", "base", "deepcot", "nystromformer", "continual_nystrom"]
num_layers_list = range(1, 3)
task_list = ["gtzan"]
seq_len_list = [-1]

if __name__=="__main__":
    filename = "config_list.txt"
    with open(filename, "w+") as f:
        for model in model_type_list:
            for task in task_list:
                feature_list = ["anet", "kin"] if task == "thumos" else ["anet"]
                for feature in feature_list:
                    for seq_len in seq_len_list:
                        if task == "gtzan":
                            data_seed_list = range(5)
                        else:
                            data_seed_list = [0]
                        for data_seed in data_seed_list:
                            for model_seed in model_seed_list:
                                for num_layers in num_layers_list:
                                    fit_layer_epochs = '[25' + ',25' * (num_layers-1) + ']'
                                    model_completed = False
                                    for num_landmarks in [4]:
                                        if model in ['base_continual', 'base', 'deepcot'] and model_completed:
                                            break
                                        model_completed = True
                                        params = "--data_seed {} --model_seed {} --model {} --num_layers {} --seq_len {} --dataset {} --feature {} --fit_layer_epochs {} --num_landmarks {} --freeze_weights false\n".format(
                                            data_seed, model_seed, model, num_layers, seq_len, task, feature, fit_layer_epochs, num_landmarks)
                                        f.write(params)