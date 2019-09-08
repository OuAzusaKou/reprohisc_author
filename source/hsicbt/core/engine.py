from .. import *
from .  import *

from .train_misc     import *
from .train_hsic     import *
from .train_standard import *
from ..utils.const   import *

def training_standard(config_dict):

    print_emph("Standard training")

    train_loader, test_loader = get_dataset_from_code(
        config_dict['data_code'], config_dict['batch_size'])

    torch.manual_seed(1234)
    vanilla_model  = ModelVanilla(**config_dict)
    torch.manual_seed(1234)
    hsic_model     = model_distribution(config_dict)
    model          = ModelEnsemble(hsic_model, vanilla_model)


   
    if config_dict['verbose']:
        print(model)

    optimizer = optim.SGD( filter(lambda p: p.requires_grad, model.parameters()), 
            lr = config_dict['learning_rate'], momentum=.9, weight_decay=1E-5)

    batch_log_list = []
    epoch_log_dict = {}
    epoch_log_dict['train_acc'] = []
    epoch_log_dict['train_loss'] = []
    epoch_log_dict['test_acc'] = []
    epoch_log_dict['test_loss'] = []
    nepoch = config_dict['epochs_standard']

    if DEBUG_MODE:
        nepoch = 2

    for cepoch in range(1, nepoch+1):
        log = standard_train(cepoch, model, train_loader, optimizer, config_dict)
        batch_log_list.append(log)
        train_acc, train_loss = misc.get_accuracy_epoch(model, train_loader)
        epoch_log_dict['train_acc'].append(train_acc) 
        epoch_log_dict['train_loss'].append(train_loss)
        test_acc, test_loss = misc.get_accuracy_epoch(model, test_loader)
        epoch_log_dict['test_acc'].append(test_acc) 
        epoch_log_dict['test_loss'].append(test_loss)
        print_highlight("Epoch - [{:04d}]: Training Acc: {:.2f}".format(cepoch, train_acc), 'green')
        print_highlight("Epoch - [{:04d}]: Testing  Acc: {:.2f}".format(cepoch, test_acc), 'green')

    if config_dict['task'] == 'needle':
        activations_extraction(model, train_loader, get_tmp_path("activation-needle-standard.npy"), 1)

    save_logs(batch_log_list, get_batch_log_filepath(
        config_dict['task'], TTYPE_STANDARD, config_dict['data_code'], config_dict['exp_index']))
    save_logs(epoch_log_dict, get_epoch_log_filepath(
        config_dict['task'], TTYPE_STANDARD, config_dict['data_code'], config_dict['exp_index']))

    del model
    return batch_log_list, epoch_log_dict

def training_format_combined(config_dict):

    print_emph("Format training")

    train_loader, test_loader = get_dataset_from_code(
        config_dict['data_code'], config_dict['batch_size'])
    
    vanilla_model = ModelVanilla(**config_dict)
    num_hsic_model = len(config_dict['model_file'])
    hsic_models = []
    for i in range(num_hsic_model):
        hsic_model = model_distribution(config_dict).to(config_dict['device'])
        model = load_model(get_model_path("{}".format(config_dict['model_file'][i])))
        hsic_model.load_state_dict(model)
        # hsic_model.eval()
        hsic_models.append(hsic_model)
    
    optimizer = optim.SGD( filter(lambda p: p.requires_grad, vanilla_model.parameters()), 
            lr = config_dict['learning_rate'], momentum=.9, weight_decay=0.001)

    ensemble_model = ModelEnsembleComb(hsic_models, vanilla_model)
    if config_dict['verbose']:
        print(ensemble_model)

    batch_log_list = []
    epoch_log_dict = {}
    epoch_log_dict['train_acc'] = []
    epoch_log_dict['train_loss'] = []
    epoch_log_dict['test_acc'] = []
    epoch_log_dict['test_loss'] = []

    nepoch = config_dict['epochs_format']

    if DEBUG_MODE:
        nepoch = 2

    for cepoch in range(1, nepoch+1):
        log = standard_train(cepoch, ensemble_model, train_loader, optimizer, config_dict)
        batch_log_list.append(log)

        train_acc, train_loss = misc.get_accuracy_epoch(ensemble_model, train_loader)
        epoch_log_dict['train_acc'].append(train_acc) 
        epoch_log_dict['train_loss'].append(train_loss)
        test_acc, test_loss = misc.get_accuracy_epoch(ensemble_model, test_loader)
        epoch_log_dict['test_acc'].append(test_acc) 
        epoch_log_dict['test_loss'].append(test_loss)
        print_highlight("Epoch - [{:04d}]: Training Acc: {:.2f}".format(cepoch, train_acc), 'green')
        print_highlight("Epoch - [{:04d}]: Testing  Acc: {:.2f}".format(cepoch, test_acc), 'green')

    save_logs(batch_log_list, get_batch_log_filepath(
        config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], config_dict['exp_index']))
    save_logs(epoch_log_dict, get_epoch_log_filepath(
        config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], config_dict['exp_index']))


    return batch_log_list, epoch_log_dict

def training_format(config_dict):

    print_emph("Format training")

    train_loader, test_loader = get_dataset_from_code(
        config_dict['data_code'], config_dict['batch_size'])
    torch.manual_seed(1234)
    vanilla_model = ModelVanilla(**config_dict)
    torch.manual_seed(1234)
    hsic_model = model_distribution(config_dict)
    
    optimizer = optim.SGD( filter(lambda p: p.requires_grad, vanilla_model.parameters()), 
            lr = config_dict['learning_rate'], momentum=.9, weight_decay=0.001)

    model = load_model(get_model_path("{}".format(
        config_dict['model_file'])))

    hsic_model.load_state_dict(model)
    hsic_model.eval()

    ensemble_model = ModelEnsemble(hsic_model, vanilla_model)
    if config_dict['verbose']:
        print(ensemble_model)

    batch_log_list = []
    epoch_log_dict = {}
    epoch_log_dict['train_acc'] = []
    epoch_log_dict['train_loss'] = []
    epoch_log_dict['test_acc'] = []
    epoch_log_dict['test_loss'] = []

    nepoch = config_dict['epochs_format']

    if DEBUG_MODE:
        nepoch = 2

    for cepoch in range(1, nepoch+1):
        log = standard_train(cepoch, ensemble_model, train_loader, optimizer, config_dict)
        batch_log_list.append(log)

        train_acc, train_loss = misc.get_accuracy_epoch(ensemble_model, train_loader)
        epoch_log_dict['train_acc'].append(train_acc) 
        epoch_log_dict['train_loss'].append(train_loss)
        test_acc, test_loss = misc.get_accuracy_epoch(ensemble_model, test_loader)
        epoch_log_dict['test_acc'].append(test_acc) 
        epoch_log_dict['test_loss'].append(test_loss)
        print_highlight("Epoch - [{:04d}]: Training Acc: {:.2f}".format(cepoch, train_acc), 'green')
        print_highlight("Epoch - [{:04d}]: Testing  Acc: {:.2f}".format(cepoch, test_acc), 'green')

    save_logs(batch_log_list, get_batch_log_filepath(
        config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], config_dict['exp_index']))
    save_logs(epoch_log_dict, get_epoch_log_filepath(
        config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], config_dict['exp_index']))

    del ensemble_model
    return batch_log_list, epoch_log_dict

def training_hsic(config_dict):

    print_emph("HSIC-Bottleneck training")

    train_loader, test_loader = get_dataset_from_code(
        config_dict['data_code'], config_dict['batch_size'])

    model = model_distribution(config_dict)
    if config_dict['verbose']:
        print(model)

    nepoch = config_dict['epochs_hsic']

    if DEBUG_MODE:
        nepoch = 2

    epoch_range = range(1, nepoch+1)
    if config_dict['checkpoint']:
        model_dict = load_model(get_model_path("{}".format(
            config_dict['model_file']), config_dict['checkpoint']))
        epoch_range = range(config_dict['checkpoint']+1, 
            config_dict['checkpoint']+config_dict['epochs_hsic']+1)
        model.load_state_dict(model_dict)

    batch_log_list = []
    epoch_log_dict = {}
    epoch_log_dict['train_acc'] = []
    epoch_log_dict['test_acc'] = []


    
    for cepoch in epoch_range:
        log = hsic_train(cepoch, model, train_loader, config_dict)
        batch_log_list.append(log)

        if  config_dict['task'] == 'hsic-train'     or \
            config_dict['task'] == 'activation'     or \
            config_dict['task'] == 'sigma-combined' or \
            config_dict['task'] == 'varied-dim'     or \
            config_dict['task'] == 'varied-epoch':
            # save with same filename for convenience
            save_model(model, get_model_path("{}".format(
                config_dict['model_file'])))
            # save with each indexed
            filename = os.path.splitext(config_dict['model_file'])[0]
            filename = "{}-{:04d}.pt".format(filename, cepoch)
            save_model(model, get_model_path("{}".format(
                filename)))

    
        if config_dict['task'] == 'hsic-solve':
            train_acc, reordered = misc.get_accuracy_hsic(model=model, dataloader=train_loader)
            test_acc, reordered = misc.get_accuracy_hsic(model=model, dataloader=test_loader)
            print_highlight("Epoch - [{:04d}]: Training Acc: {:.2f}".format(cepoch, train_acc), 'green')
            print_highlight("Epoch - [{:04d}]: Testing  Acc: {:.2f}".format(cepoch, test_acc), 'green')
            epoch_log_dict['train_acc'].append(train_acc)
            epoch_log_dict['test_acc'].append(test_acc)

    if config_dict['task'] == 'hsic-solve':
        activations_extraction(model, train_loader, "./assets/tmp/activation-onehot.npy")
    if config_dict['task'] == 'needle':
        activations_extraction(model, train_loader, "./assets/tmp/activation-needle-hsic.npy", 1)

    save_logs(batch_log_list, get_batch_log_filepath(
        config_dict['task'], TTYPE_HSICTRAIN, config_dict['data_code'], config_dict['exp_index']))
    save_logs(epoch_log_dict, get_epoch_log_filepath(
        config_dict['task'], TTYPE_HSICTRAIN, config_dict['data_code'], config_dict['exp_index']))

    return batch_log_list, epoch_log_dict
