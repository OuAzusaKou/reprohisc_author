from .. import *

from ..utils import meter
from ..utils import misc
from ..utils.io import *
from ..utils.color import *
from ..utils.dataset   import get_dataset_from_code

from ..model.mvanilla    import ModelVanilla
from ..model.mhlinear    import ModelLinear
from ..model.mhconv      import ModelConv
from ..model.mreslinear  import ModelResLinear
from ..model.mresconv    import ModelResConv
from ..model.mensemble   import ModelEnsemble
from ..model.mniddle     import ModelNiddle

from ..math.hsic import *


def _activations_extraction(model, data_loader, filepath, out_dim=10, hid_idx=-1,):

    out_activation = np.zeros([len(data_loader)*data_loader.batch_size, out_dim])
    out_label = np.zeros([len(data_loader)*data_loader.batch_size,])
    device = next(model.parameters()).device

    for batch_idx, (data, target) in enumerate(data_loader):
        
        if len(data)<data_loader.batch_size:
            break

        data = data.to(device)
        output, hiddens = model(data)
        
        begin = batch_idx*data_loader.batch_size
        end = (batch_idx+1)*data_loader.batch_size
        out_activation[begin:end] = hiddens[hid_idx].detach().cpu().numpy()
        out_label[begin:end] = target.detach().cpu().numpy()

    
    np.save(filepath, {"activation":out_activation, "label":out_label})
    print("Saved", filepath, out_activation.shape)

def _hsic_objective(hidden, h_target, h_data, sigma_hy, sigma_hx):
        
    hsic_hy_val = hsic_normalized_cca( hidden, h_target, sigma=sigma_hy)
    hsic_hx_val = hsic_normalized_cca( hidden, h_data,   sigma=sigma_hx)

    return hsic_hx_val, hsic_hy_val
  
def _normal_train(cepoch, model, data_loader, optimizer, config_dict):

    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    prec1 = total_loss = hx_l = hy_l = -1

    batch_acc    = meter.AverageMeter()
    batch_loss   = meter.AverageMeter()
    batch_hischx = meter.AverageMeter()
    batch_hischy = meter.AverageMeter()

    batch_log = {}
    batch_log['batch_acc'] = []
    batch_log['batch_loss'] = []
    batch_log['batch_hsic_hx'] = []
    batch_log['batch_hsic_hy'] = []

    model = model.to(config_dict['device'])


    n_data = config_dict['batch_size'] * len(data_loader)
    
    pbar = tqdm(enumerate(data_loader), total=n_data/config_dict['batch_size'], ncols=150)
    # for batch_idx, (data, target) in enumerate(data_loader):
    for batch_idx, (data, target) in pbar:

        data   = data.to(config_dict['device'])
        target = target.to(config_dict['device'])
        output, hiddens = model(data)

        h_target = target.view(-1,1)
        h_target = misc.to_categorical(h_target, num_classes=10).float()
        
        h_data = data.view(-1, np.prod(data.size()[1:]))

        # # # if want to monitor hsic
        # for i in range(len(hiddens)):
        #     hx_l, hy_l = hsic_objective(
        #             hiddens[i], 
        #             h_target=h_target.float(), 
        #             h_data=h_data,
        #             sigma_hx=config_dict['sigma_hx'],
        #             sigma_hy=config_dict['sigma_hy']
        #         )

        optimizer.zero_grad()
        loss = cross_entropy_loss(output, target)
        loss.backward()
        optimizer.step()


        loss = float(loss.detach().cpu().numpy())
        prec1, prec5 = misc.get_accuracy(output, target, topk=(1, 5)) 
        prec1 = float(prec1.cpu().numpy())
    
        batch_acc.update(prec1)   
        batch_loss.update(loss)  
        batch_hischx.update(hx_l)
        batch_hischy.update(hy_l)

        msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] Loss:{loss:.4f} Acc:{acc:.4f} H_hx:{H_hx:.4f} H_hy:{H_hy:.4f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*config_dict['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*config_dict['batch_size']/n_data), 
                        loss = batch_loss.avg, 
                        acc  = batch_acc.avg,
                        H_hx = batch_hischx.avg, 
                        H_hy = batch_hischy.avg,
                    )


        # # # preparation log information and print progress # # #
        if ((batch_idx+1) % config_dict['log_batch_interval'] == 0):
            
            batch_log['batch_acc'].append(batch_acc.avg)
            batch_log['batch_loss'].append(batch_loss.avg)
            batch_log['batch_hsic_hx'].append(batch_hischx.avg)
            batch_log['batch_hsic_hy'].append(batch_hischy.avg)

        pbar.set_description(msg)

    return batch_log

def _hsic_train(cepoch, model, data_loader, config_dict):


    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    prec1 = total_loss = hx_l = hy_l = -1

    batch_acc    = meter.AverageMeter()
    batch_loss   = meter.AverageMeter()
    batch_hischx = meter.AverageMeter()
    batch_hischy = meter.AverageMeter()

    batch_log = {}
    batch_log['batch_acc'] = []
    batch_log['batch_loss'] = []
    batch_log['batch_hsic_hx'] = []
    batch_log['batch_hsic_hy'] = []

    model = model.to(config_dict['device'])

    n_data = config_dict['batch_size'] * len(data_loader)
    

    # for batch_idx, (data, target) in enumerate(data_loader):
    pbar = tqdm(enumerate(data_loader), total=n_data/config_dict['batch_size'], ncols=130)
    for batch_idx, (data, target) in pbar:

        data   = data.to(config_dict['device'])
        target = target.to(config_dict['device'])
        output, hiddens = model(data)

        h_target = target.view(-1,1)
        h_target = misc.to_categorical(h_target, num_classes=10).float()
        h_data = data.view(-1, np.prod(data.size()[1:]))

        for i in range(len(hiddens)):
            output, hiddens = model(data)
            params, param_names = misc.get_layer_parameters(model=model, layer_idx=i) # so we only optimize one layer at a time
            optimizer = optim.SGD(params, lr = config_dict['hsic_learning_rate'], momentum=.9, weight_decay=0.001)
            optimizer.zero_grad()
            if len(hiddens[i].size()) > 2:
                hiddens[i] = hiddens[i].view(-1, np.prod(hiddens[i].size()[1:]))

            hx_l, hy_l = _hsic_objective(
                    hiddens[i], 
                    h_target=h_target.float(), 
                    h_data=h_data,
                    sigma_hx=config_dict['sigma_hx'],
                    sigma_hy=config_dict['sigma_hy']
                )
            loss = config_dict['lambda_x']*hx_l - config_dict['lambda_y']*hy_l
            loss.backward()
            optimizer.step()
        
        # if config_dict['hsic_solve']:
        #     prec1, reorder_list = misc.get_accuracy_hsic(model, data_loader)
    
        batch_acc.update(prec1)   
        batch_loss.update(total_loss)  
        batch_hischx.update(hx_l)
        batch_hischy.update(hy_l)

        # # # preparation log information and print progress # # #

        msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] H_hx:{H_hx:.4f} H_hy:{H_hy:.4f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*config_dict['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*config_dict['batch_size']/n_data), 
                        H_hx = batch_hischx.avg, 
                        H_hy = batch_hischy.avg,
                )

        if ((batch_idx+1) % config_dict['log_batch_interval'] == 0):
            batch_log['batch_acc'].append(batch_loss.avg)
            batch_log['batch_loss'].append(batch_acc.avg)
            batch_log['batch_hsic_hx'].append(batch_hischx.avg)
            batch_log['batch_hsic_hy'].append(batch_hischy.avg)
        
        pbar.set_description(msg)
        
    return batch_log

def _model_distribution(config_dict):

    if config_dict['model'] == 'niddle':
        model = ModelNiddle(**config_dict)
    elif config_dict['model'] == 'conv':
        model = ModelConv(**config_dict)
    elif config_dict['model'] == 'linear':
        model = ModelLinear(**config_dict)
    elif config_dict['model'] == 'resnet-linear':
        model = ModelResLinear(**config_dict)
    elif config_dict['model'] == 'resnet-conv':
        model = ModelResConv(**config_dict)
    else:
        raise ValueError("Unknown model name or not support [{}]".format(config_dict['model']))

    return model

def training_standard(config_dict):

    print_emph("Standard training")

    train_loader, test_loader = get_dataset_from_code(
        config_dict['data_code'], config_dict['batch_size'])

    vanilla_model  = ModelVanilla(**config_dict)
    hsic_model     = _model_distribution(config_dict)
    model          = ModelEnsemble(hsic_model, vanilla_model)
    if config_dict['verbose']:
        print(model)

    optimizer = optim.SGD( filter(lambda p: p.requires_grad, model.parameters()), 
            lr = config_dict['learning_rate'], momentum=.9, weight_decay=0.001)

    batch_log_list = []
    epoch_log_dict = {}
    epoch_log_dict['train_acc'] = []
    epoch_log_dict['train_loss'] = []
    epoch_log_dict['test_acc'] = []
    epoch_log_dict['test_loss'] = []

    for cepoch in range(config_dict['epochs_standard']):
        log = _normal_train(cepoch, model, train_loader, optimizer, config_dict)
        batch_log_list.append(log)
        train_acc, train_loss = misc.get_accuracy_epoch(model, train_loader)
        epoch_log_dict['train_acc'].append(train_acc) 
        epoch_log_dict['train_loss'].append(train_loss)
        test_acc, test_loss = misc.get_accuracy_epoch(model, test_loader)
        epoch_log_dict['test_acc'].append(test_acc) 
        epoch_log_dict['test_loss'].append(test_loss)
        print_highlight("Epoch - [{:04d}]: Training Acc: {:.2f}".format(cepoch, train_acc), 'green')
        print_highlight("Epoch - [{:04d}]: Testing  Acc: {:.2f}".format(cepoch, test_acc), 'green')

    if config_dict['task'] == 'niddle':
        _activations_extraction(model, train_loader, "./assets/activation-niddle-standard.npy", 1)

    return batch_log_list, epoch_log_dict

def training_format(config_dict):

    print_emph("Format training")

    train_loader, test_loader = get_dataset_from_code(
        config_dict['data_code'], config_dict['batch_size'])
    
    vanilla_model = ModelVanilla(**config_dict)
    hsic_model = _model_distribution(config_dict)
    
    optimizer = optim.SGD( filter(lambda p: p.requires_grad, hsic_model.parameters()), 
            lr = config_dict['learning_rate'], momentum=.9, weight_decay=0.001)

    model = load_model("models/{}".format(config_dict['model_file']))

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

    for cepoch in range(config_dict['epochs_format']):
        log = _normal_train(cepoch, ensemble_model, train_loader, optimizer, config_dict)
        batch_log_list.append(log)

        train_acc, train_loss = misc.get_accuracy_epoch(ensemble_model, train_loader)
        epoch_log_dict['train_acc'].append(train_acc) 
        epoch_log_dict['train_loss'].append(train_loss)
        test_acc, test_loss = misc.get_accuracy_epoch(ensemble_model, test_loader)
        epoch_log_dict['test_acc'].append(test_acc) 
        epoch_log_dict['test_loss'].append(test_loss)
        print_highlight("Epoch - [{:04d}]: Training Acc: {:.2f}".format(cepoch, train_acc), 'green')
        print_highlight("Epoch - [{:04d}]: Testing  Acc: {:.2f}".format(cepoch, test_acc), 'green')

    return batch_log_list, epoch_log_dict

def training_hsic(config_dict):

    print_emph("HSIC-Bottleneck training")

    train_loader, test_loader = get_dataset_from_code(
        config_dict['data_code'], config_dict['batch_size'])

    model = _model_distribution(config_dict)
    if config_dict['verbose']:
        print(model)

    batch_log_list = []
    epoch_log_dict = {}
    epoch_log_dict['train_acc'] = []
    epoch_log_dict['test_acc'] = []
    
    for cepoch in range(config_dict['epochs_hsic']):
        log = _hsic_train(cepoch, model, train_loader, config_dict)
        batch_log_list.append(log)

        if config_dict['task'] == 'hsic-train':
            save_model(model, "models/{}".format(config_dict['model_file']))
            filename = os.path.splitext(config_dict['model_file'])[0]
            filename = "{}-{:04d}.pt".format(filename, cepoch)
            save_model(model, "models/{}".format(filename))

    
        if config_dict['task'] == 'hsic-solve':
            train_acc, reordered = misc.get_accuracy_hsic(model=model, dataloader=train_loader)
            test_acc, reordered = misc.get_accuracy_hsic(model=model, dataloader=test_loader)
            print_highlight("Epoch - [{:04d}]: Training Acc: {:.2f}".format(cepoch, train_acc), 'green')
            print_highlight("Epoch - [{:04d}]: Testing  Acc: {:.2f}".format(cepoch, test_acc), 'green')
            epoch_log_dict['train_acc'].append(train_acc)
            epoch_log_dict['test_acc'].append(test_acc)

    if config_dict['task'] == 'hsic-solve':
        _activations_extraction(model, train_loader, "./assets/activation-onehot.npy")
    if config_dict['task'] == 'niddle':
        _activations_extraction(model, train_loader, "./assets/activation-niddle-hsic.npy", 1)

    return batch_log_list, epoch_log_dict