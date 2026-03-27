import argparse
import collections
import os.path
import random
import warnings
import sys
from datetime import datetime
from time import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

root = "ROOT-PATH"
os.chdir(root)
sys.path.append(root)

from recipes.finetune.setting import *
from recipes.finetune.train import Trainer
from src.utils.statistics.model_statistic import count_parameters
from src.utils.log import BestModels
from src.utils.scheduler import ExponentialDown


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def prepare_run():

    # parse the argument
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--gpu',
                        default=0,
                        type=int,
                        help='selection of gpu when you run separate trainings on single server')
    parser.add_argument('--multigpu', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--random_seed', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--config_dir', type=str)
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--continual', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--dataset', type=str, default='dcase', help='options: [dcase, urban]')
    args = parser.parse_args()

    assert args.dataset in ['dcase', 'urban']

    #set configurations
    configs = get_configs(config_dir=args.config_dir)

    print("=" * 50 + "start!!!!" + "=" * 50)
    if configs["generals"]["test_only"]:
        print(" " * 40 + "<" * 10 + "test only" + ">" * 10)

    configs = get_save_directories(configs, args.save_folder)
    # set logger
    my_logger = get_logger(configs["generals"]["save_folder"], (not configs["generals"]["test_only"]),
                           log_level=eval("logging." + configs["generals"]["log_level"].upper()))

    my_logger.logger.info("date & time of start is : " + str(datetime.now()).split('.')[0])
    my_logger.logger.info("torch version is: " + str(torch.__version__))

    # set device
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    my_logger.logger.info("number of GPUs: " + str(torch.cuda.device_count()))
    configs["training"]["device"] = device
    my_logger.logger.info("device: " + str(device))

    # set seed
    if args.random_seed:
        seed = random.randint(0, 10000)
        setup_seed(seed)
        my_logger.logger.info("use random seed {}".format(seed))
        configs["training"]["seed"] = seed
    else:
        seed = configs["training"]["seed"]
        setup_seed(seed)
        my_logger.logger.info("use fix seed {}".format(seed))

    # do not show warning
    if not configs["generals"]["warn"]:
        warnings.filterwarnings("ignore")

    configs['PaSST_SED']['continual'] = args.continual
    configs['PaSST_SED']['window_size'] = 1188 # TODO: Hotfix number
    return configs, my_logger, args


def adapt_state_dict(net, state_dict, continual, rotate_conv_weights=False, encoder_blocks=10, multigpu=True):
    if continual:
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not (
               (match := re.search(r"blocks\.(\d+)", k))
               and int(match.group(1)) >= encoder_blocks
            )
        }

        if continual and not 'module.patch_transformer.time_new_pos_embed.pe.weight' in state_dict.keys():
            patch_trans_old_key = 'module.patch_transformer.time_new_pos_embed'
            patch_trans_old_key = patch_trans_old_key if patch_trans_old_key in state_dict.keys() else 'time_new_pos_embed'
            state_dict['module.patch_transformer.time_new_pos_embed.pe.weight'] = state_dict[patch_trans_old_key].squeeze().transpose(-1, -2)
            del state_dict['module.patch_transformer.time_new_pos_embed']
        if rotate_conv_weights:
            state_dict['module.patch_transformer.patch_embed.proj.weight'] = state_dict['module.patch_transformer.patch_embed.proj.weight'].transpose(-1, -2)

    if not multigpu:
        state_dict = {
            key.replace("module.", "", 1) if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        } # Hotfix for single GPU

    net.load_state_dict(state_dict, strict=False)
    return net

if __name__ == "__main__":
    configs, my_logger, args = prepare_run()

    # set network
    net, ema_net = get_models_passt(configs, args.dataset)

    if args.dataset == 'urban':
        labels_urban = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling',
                        'gun_shot', 'jackhammer', 'siren', 'street_music']
        LabelDict = collections.OrderedDict((k, i) for i, k in enumerate(labels_urban))
    else:
        LabelDict = get_labeldict()

    # set encoder
    encoder = get_encoder_passt(LabelDict, configs)

    # get dataset
    train_dataset, valid_dataset, test_dataset, batch_sampler, tot_train_data = get_datasets(
        configs,
        encoder,
        evaluation=configs["generals"]["test_on_public_eval"],
        test_only=configs["generals"]["test_only"],
        logger=my_logger.logger,
        dataset=args.dataset
    )

    # logger.info("---------------model structure---------------")
    # logger.info(train_cfg['net'])

    # set dataloader
    test_loader = DataLoader(test_dataset,
                             batch_size=configs["training"]["batch_size_val"],
                             num_workers=configs["training"]["num_workers"], drop_last=True)
    if not configs["generals"]["test_only"]:
        if args.dataset == 'urban':
            train_loader = DataLoader(train_dataset,
                                      batch_size=configs["training"]["batch_size"],
                                      num_workers=configs["training"]["num_workers"])
        else:
            train_loader = DataLoader(train_dataset,
                                      batch_sampler=batch_sampler,
                                      num_workers=configs["training"]["num_workers"])
        val_loader = DataLoader(valid_dataset,
                                batch_size=configs["training"]["batch_size_val"],
                                num_workers=configs["training"]["num_workers"], drop_last=True)
    else:
        train_loader, val_loader = None, None

    # set learning rate
    optim_kwargs = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-8}
    total_lr = get_params(net, configs, my_logger.logger)

    optimizer = optim.AdamW(total_lr, **optim_kwargs)

    my_logger.logger.info("Total  Params: %.3f M" % (count_parameters(net, trainable_only=False) * 1e-6))

    my_logger.logger.info("Total Trainable Params: %.3f M" % (count_parameters(net) * 1e-6))

    if not configs["generals"]["test_only"]:
        total_iter = configs["training"]["n_epochs"] * len(train_loader)
        start_iter = configs["training"]["n_epochs_cut"] * len(train_loader)

        my_logger.logger.info("learning rate keep no change until iter{}, then expdown, total iter:{}".format(
            start_iter, total_iter))

        scheduler = ExponentialDown(optimizer=optimizer,
                                    start_iter=start_iter,
                                    total_iter=total_iter,
                                    exponent=configs['opt']['exponent'],
                                    warmup_iter=configs["training"]["lr_warmup_epochs"] * len(train_loader),
                                    warmup_rate=configs["training"]["lr_warmup_rate"])

        my_logger.logger.info("learning rate warmup until iter{}, then keep, total iter:{}".format(
            start_iter, total_iter))
    else:
        scheduler = None

    #### move to gpus ########
    if args.multigpu:
        net = nn.DataParallel(net)
        if ema_net is not None:
            ema_net = nn.DataParallel(ema_net)
    else:
        logging.warning("Run with only single GPU!")
    net = net.to(configs["training"]["device"])
    if ema_net is not None:
        ema_net = ema_net.to(configs["training"]["device"])

    ##############################                TRAIN/VALIDATION                ##############################
    trainer = Trainer(optimizer=optimizer,
                      my_logger=my_logger,
                      net=net,
                      ema_net=ema_net,
                      config=configs,
                      encoder=encoder,
                      scheduler=scheduler,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader,
                      device=configs["training"]["device"],
                      continual=args.continual,
                      multigpu=args.multigpu,
                      dataset=args.dataset)

    if not configs['generals']["test_only"]:
        my_logger.logger.info('   training starts!')
        start_time = time()
        bestmodels = BestModels()

        # load existed model
        if ("finetune_mlm" in configs['generals'].keys()) and configs['generals']["finetune_mlm"]:
            params_dict = torch.load(configs['training']["best_paths"][0])
            params = {k: v
                      for k, v in params_dict.items() if ".decoder" in k}  # only load decoder part from pretrain model

            adapt_state_dict(trainer.ema_net, params, continual=args.continual,
                             rotate_conv_weights=True, encoder_blocks=configs['PaSST_SED']['passt_feature_layer'],
                             multigpu=args.multigpu)
            adapt_state_dict(trainer.net, params, continual=args.continual,
                             rotate_conv_weights=True, encoder_blocks=configs['PaSST_SED']['passt_feature_layer'],
                             multigpu=args.multigpu)
            #
            # trainer.ema_net.load_state_dict(params, strict=False)
            # trainer.net.load_state_dict(params, strict=False)

        # load existed model
        elif configs['generals']["load_from_existed_path"]:
            # trainer.ema_net.load_state_dict(torch.load(configs['training']["best_paths"][1]))
            # trainer.net.load_state_dict(torch.load(configs['training']["best_paths"][0]))

            adapt_state_dict(trainer.ema_net, torch.load(configs['training']["best_paths"][1]), continual=args.continual,
                             rotate_conv_weights=True, encoder_blocks=configs['PaSST_SED']['passt_feature_layer'],
                             multigpu=args.multigpu)
            adapt_state_dict(trainer.net, torch.load(configs['training']["best_paths"][0]), continual=args.continual,
                             rotate_conv_weights=True, encoder_blocks=configs['PaSST_SED']['passt_feature_layer'],
                             multigpu=args.multigpu)

            val_metrics = trainer.validation(epoch=-1)
            bestmodels.update(net, ema_net, 0, my_logger.logger, val_metrics)

        for epoch in range(configs['training']["n_epochs"]):
            epoch_time = time()
            #training
            trainer.train(epoch)
            # validation
            if (not epoch % configs["generals"]["validation_interval"]) or (epoch
                                                                            == configs['training']["n_epochs"] - 1):
                val_metrics = trainer.validation(epoch)
                bestmodels.update(net, ema_net, epoch + 1, my_logger.logger, val_metrics)
            if not epoch % 2:
                bestmodels.save_bests(configs['training']["best_paths"])

        #save model parameters & history dictionary
        my_logger.logger.info("        best student/teacher val_metrics: %.3f / %.3f" %
                              bestmodels.save_bests(configs['training']["best_paths"]))
        my_logger.logger.info("   training took %.2f mins" % ((time() - start_time) / 60))

    ##############################                        TEST                        ##############################
    my_logger.logger.info("   test starts!")
    # test on best model

    if ema_net is not None:
        trainer.ema_net.load_state_dict(torch.load(configs['training']["best_paths"][1]), strict=False)
    trainer.net.load_state_dict(torch.load(configs['training']["best_paths"][0]), strict=False)

    # trainer.net = adapt_state_dict(trainer.net, torch.load(configs['training']["best_paths"][0]), continual=args.continual, rotate_conv_weights=True, encoder_blocks=configs['PaSST_SED']['passt_feature_layer'], multigpu=args.multigpu)
    # trainer.ema_net = adapt_state_dict(trainer.ema_net, torch.load(configs['training']["best_paths"][1]), continual=args.continual, rotate_conv_weights=True, encoder_blocks=configs['PaSST_SED']['passt_feature_layer'], multigpu=args.multigpu)

    # Separate the encoder block in two parts to avoid shared memory issues
    if args.multigpu:
        trainer.net.module.prepare_continual()
        if ema_net is not None:
            trainer.ema_net.module.prepare_continual()
    else:
        trainer.net.prepare_continual()
        if ema_net is not None:
            trainer.ema_net.prepare_continual()

    trainer.test()
    trainer.efficiency_test(args.continual)

    my_logger.logger.info("date & time of end is : " + str(datetime.now()).split('.')[0])

    print("<" * 30 + "DONE!" + ">" * 30)
    my_logger.__del__()
