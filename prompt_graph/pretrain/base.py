import torch
from prompt_graph.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer
from torch.optim import Adam
from logging import getLogger

class PreTrain(torch.nn.Module):
    def __init__(self,config_name='0', gnn_type='TransformerConv', dataset_name='Cora', hid_dim=128, gln=2, num_epoch=100, device=5, lr=0.001):
        super().__init__()
        self.device = torch.device(
            "cuda:%d" % device if torch.cuda.is_available() else "cpu")
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.num_layer = gln
        self.epochs = num_epoch
        self.hid_dim = hid_dim
        self.lr = lr
        self.config_name = config_name
        self._logger = getLogger()
    
    def initialize_gnn(self, input_dim, hid_dim):
        if self.gnn_type == 'GAT':
                self.gnn = GAT(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer)
        elif self.gnn_type == 'GCN':
                self.gnn = GCN(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer)
        elif self.gnn_type == 'GraphSAGE':
                self.gnn = GraphSAGE(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer)
        elif self.gnn_type == 'GIN':
                self.gnn = GIN(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer)
        elif self.gnn_type == 'GCov':
                self.gnn = GCov(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer)
        elif self.gnn_type == 'GraphTransformer':
                self.gnn = GraphTransformer(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer)
        else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        print(self.gnn)
        self.gnn.to(self.device)
        self._logger.info(self.gnn)
        for name, param in self.gnn.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        total_num = sum([param.nelement() for param in self.gnn.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))
        
        self.optimizer = Adam(self.gnn.parameters(), lr=self.lr)


        
#     def load_node_data(self):
#         self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
#         self.data.to(self.device)
#         self.input_dim = self.dataset.num_features
#         self.output_dim = self.dataset.num_classes

