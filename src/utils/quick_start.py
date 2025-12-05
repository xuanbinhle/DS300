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
from utils.data_utils import init_seed, get_model, dict2str
from utils.trainer import Trainer
import platform
import os


def quick_start(model, dataset, config_dict, mg=False):
    config = Config(model, config_dict, mg)
    
    # load data
    dataset = RecDataset(config, dataset)
    train_dataset, val_dataset, test_dataset = dataset.split()
    print('\n====Training====\n' + str(train_dataset))
    print('\n====Validation====\n' + str(val_dataset))
    print('\n====Testing====\n' + str(test_dataset))
    
    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
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
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
        
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])
        
        
        print(config)
        print(f"========={idx+1}/{total_loops}: Parameters:{config['hyper_parameters']}={hyper_tuple}=======")

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        print(model)

        # trainer loading and initialization
        trainer = Trainer(config, model, mg)
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        print('best valid result: {}'.format(dict2str(best_valid_result)))
        print('test result: {}'.format(dict2str(best_test_upon_valid)))
        print(f"### Current BEST:\nParameters: {config['hyper_parameters']}={hyper_ret[best_test_idx][0]}, \
              \nValid: {dict2str(hyper_ret[best_test_idx][1])},\nTest: {dict2str(hyper_ret[best_test_idx][2])}\n\n\n")


    print("\n============All Over=====================")
    for (p, k, v) in hyper_ret:
        print('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    print('\n\n█████████████ BEST ████████████████')
    print('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))
