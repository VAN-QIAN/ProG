from prompt_graph.pretrain import Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE, PrePrompt, DGI
from prompt_graph.utils import seed_everything
from prompt_graph.utils import mkdir, get_args, ConfigParser, get_logger
import random

def run_model(task=None, model_name=None, dataset_name=None, config_file=None,
              saved_model=True, train=True, other_args=None):
    config = ConfigParser(task, model_name, dataset_name,
                            config_file, saved_model, train, other_args)
    exp_id = config.get('exp_id', None)
    if exp_id is None:
        # Make a new experiment ID
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = exp_id
    # logger
    logger = get_logger(config)
    logger.info('Begin pipeline, config_file={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(config_file), str(model_name), str(dataset_name), str(exp_id)))
    logger.info(config.config)
    # seed
    # seed = config.get('seed', 0)

    # args = get_args()
    seed_everything(config['seed'])
    mkdir('./pre_trained_gnn/')
    # config.task = 'DGI'
    if config['task'] == 'SimGRACE':
        pt = SimGRACE(config_name=config['config_file'].split('_')[-1],dataset_name=config['dataset'], gnn_type=config['model'], hid_dim=config['nhid'], gln=config['num_layer'], num_epoch=config['num_epochs'], device=config['device'],lr=config['learning_rate'])
    if config['task'] == 'GraphCL':
        pt = GraphCL(dataset_name=config['dataset'], gnn_type=config['model'], hid_dim=config['nhid'], gln=config['num_layer'], num_epoch=config['num_epochs'], device=config['device'],lr=config['learning_rate'])
    if config['task'] == 'Edgepred_GPPT':
        pt = Edgepred_GPPT(config_name=config['config_file'].split('_')[-1],dataset_name=config['dataset'], gnn_type=config['model'], hid_dim=config['nhid'], gln=config['num_layer'], num_epoch=config['num_epochs'], device=config['device'],lr=config['learning_rate'])
    if config['task'] == 'Edgepred_Gprompt':
        pt = Edgepred_Gprompt(dataset_name=config['dataset'], gnn_type=config['model'], hid_dim=config['nhid'], gln=config['num_layer'], num_epoch=config['num_epochs'], device=config['device'],lr=config['learning_rate'])
    if config['task'] == 'DGI':
        pt = DGI(dataset_name=config['dataset'], gnn_type=config['model'], hid_dim=config['nhid'], gln=config['num_layer'], num_epoch=config['num_epochs'], device=config['device'],lr=config['learning_rate'])
    if config['task'] == 'MultiGprompt':
        nonlinearity = 'prelu'
        pt = PrePrompt(config['dataset'], config['nhid'], nonlinearity, 0.9, 0.9, 0.1, 0.001, 2, 0.3)
    print(pt.hid_dim)
    pt.pretrain(epochs=config['num_epochs'], lr=config['learning_rate'])