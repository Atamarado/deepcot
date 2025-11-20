from run_glue_roformer import get_window_sizes, GLUE_TASKS

if __name__=="__main__":
    filename = "config_list_glue.txt"
    with open(filename, "w+") as f:
        for model in ['roformer', 'soft', 'modernbert', 'fnet']:
            if model == 'soft':
                model_name = 'roformer'
            else:
                model_name = model
            for ws_factor in [0.5, 1, 2]:
                task_list = GLUE_TASKS

                window_sizes = get_window_sizes(ws_factor)
                for task in task_list:
                    window_size = window_sizes[task]
                    deepcot_options = [True, False] if model in ['roformer', 'soft'] else [False]
                    for deepcot in deepcot_options:
                        params = f"--model {model_name} --task {task} --window_size {window_size}"
                        if deepcot:
                            params += " --deepcot --deepcot_train --forward_steps_train"
                        if model == 'soft':
                            params += " --reduced_attention"
                        params += "\n"
                        f.write(params)
