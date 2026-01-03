# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.configurator import Config
from utils.utils import init_seed, get_model, dict2str
from utils.trainer import Trainer
import platform
import os
import torch
from tqdm import tqdm


def inference_quick_start(model, dataset, config_dict, user_id, mg=False):
    # load saved model
    load_path = os.path.join('saved', f"{model}_best.pth")
    checkpoint = torch.load(load_path, weights_only=False, map_location=torch.device('cpu'))
    config_dict.update(checkpoint['config'])
    
    config = Config(model, config_dict, mg)
    dataset = dataset[dataset['customer_index'] == user_id].reset_index(drop=True)

    # load data
    dataset = RecDataset(config, dataset)
    train_dataset, val_dataset, test_dataset = dataset.split()
    print('\n====Training====\n' + str(train_dataset))
    print('\n====Validation====\n' + str(val_dataset))
    print('\n====Testing====\n' + str(test_dataset))
    
    # wrap into dataloader
    train_data = TrainDataLoader(config, dataset, batch_size=config['train_batch_size'], shuffle=True)
    test_data = EvalDataLoader(config, dataset, additional_dataset=dataset, batch_size=config['eval_batch_size'])

    # model loading and initialization
    model = get_model(config['model'])(config, train_data).to(config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"### Loaded model from {load_path} ###")

    # trainer loading and initialization
    trainer = Trainer(config, model, mg)

    # model inference
    rec_list = trainer.inference(test_data)[user_id]
    print(rec_list)
    raise
    print('Test result: {}'.format(dict2str(test_result)))


def quick_start(model, dataset, config_dict, mg=False, saved=False):
    config = Config(model, config_dict, mg)
    
    # load data
    dataset = RecDataset(config, dataset)
    train_dataset, val_dataset, test_dataset = dataset.split()
    print('\n====Training====\n' + str(train_dataset))
    print('\n====Validation====\n' + str(val_dataset))
    print('\n====Testing====\n' + str(test_dataset))
    
    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=False)
    valid_data = EvalDataLoader(config, val_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
    test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    print('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
        
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in tqdm(combinators, desc="Hyper-parameter tuning with Training loops"):
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])

        # set random state of dataloader
        train_data.pretrain_setup(config['seed'])
        # trainer loading and initialization
        trainer = Trainer(config, model, mg)
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid, best_model_state_dict = trainer.fit(train_data, valid_data=valid_data, test_data=test_data)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
            if saved:
                os.makedirs(config['checkpoint_dir'], exist_ok=True)
                save_path = os.path.join(config['checkpoint_dir'], f"{config['model']}_best.pth")
                torch.save({
                    'model_state_dict': best_model_state_dict,
                    'best_valid_score': best_test_value,
                    'best_hyper_tuple': hyper_ret[best_test_idx][0],
                    'config': config.final_config_dict,   # hoặc chỉ lưu các key cần thiết
                }, save_path)
                print(f"### Best model saved to {save_path} ###")
        idx += 1


    print("\n============All Over=====================")
    for (p, k, v) in hyper_ret:
        print('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'], p, dict2str(k), dict2str(v)))

    print('\n\n█████████████ BEST ████████████████')
    print('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))
