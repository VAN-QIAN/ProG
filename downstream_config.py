from ProG.utils import seed_everything, seed

seed_everything(seed)

from ProG.pre_train import PreTrain
from ProG.utils import mkdir, load_data4pretrain
from ProG.prompt import GNN, LightPrompt, HeavyPrompt
from torch import nn, optim
from ProG.data import multi_class_NIG
import torch
from torch_geometric.loader import DataLoader
from ProG.eva import acc_f1_over_batches
from ProG.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer
from ProG.utils import mkdir
from ProG.config_parser import ConfigParser

def initialize_gnn(input_dim, hid_dim, out_dim, num_layer=2, gnn_type='GCN'):
        if gnn_type == 'GAT':
            gnn = GAT(input_dim=input_dim, hid_dim=out_dim, num_layer=num_layer)
        elif gnn_type == 'GCN':
            gnn = GCN(input_dim=input_dim, hid_dim=out_dim, num_layer=num_layer)
        elif gnn_type == 'GraphSAGE':
            gnn = GraphSAGE(input_dim=input_dim, hid_dim=out_dim, num_layer=num_layer)
        elif gnn_type == 'GIN':
            gnn = GIN(input_dim=input_dim, hid_dim=out_dim, num_layer=num_layer)
        elif gnn_type == 'GCov':
            gnn = GCov(input_dim=input_dim, hid_dim=out_dim, num_layer=num_layer)
        elif gnn_type == 'GraphTransformer':
            gnn = GraphTransformer(input_dim=input_dim, hid_dim=out_dim, num_layer=num_layer)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
            
        return gnn
# this file can not move in ProG.utils.py because it will cause self-loop import
def model_create(dataname, gnn_type, num_class,pre_train,config_num, epoch_num ,task_type='multi_class_classification', tune_answer=True):
    if task_type in ['multi_class_classification', 'regression']:
        config_file = f'./random_config_new/config_{config_num}'
        config = ConfigParser(config_file)
        if dataname == 'Cora':
            input_dim = 1433
        elif dataname == 'CiteSeer':
            input_dim = 3703
        hid_dim = config.get('nhid')
        num_layer = config.get('num_layer')
        lr, wd = 0.001, 0.00001
        tnpc = 100  # token number per class
 
        # load pre-trained GNN
        # gnn = GNN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=2, gnn_type=gnn_type)
        gnn = initialize_gnn(input_dim, hid_dim, hid_dim, num_layer=num_layer, gnn_type=gnn_type)
        print(gnn_type)
        pre_train_path = './pre_trained_model/{}/{}/config_{}.{}.epoch_{}.pth'.format(dataname,pre_train,config_num,gnn_type ,epoch_num )
        print(pre_train_path)
        gnn.load_state_dict(torch.load(pre_train_path))
        print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
        for p in gnn.parameters():
            p.requires_grad = False

        if tune_answer:
            PG = HeavyPrompt(token_dim=input_dim, token_num=10, cross_prune=0.1, inner_prune=0.3)
        else:
            PG = LightPrompt(token_dim=input_dim, token_num_per_group=tnpc, group_num=num_class, inner_prune=0.01)

        opi = optim.Adam(filter(lambda p: p.requires_grad, PG.parameters()),
                         lr=lr,
                         weight_decay=wd)

        if task_type == 'regression':
            lossfn = nn.MSELoss(reduction='mean')
        else:
            lossfn = nn.CrossEntropyLoss(reduction='mean')

        if tune_answer:
            if task_type == 'regression':
                answering = torch.nn.Sequential(
                    torch.nn.Linear(hid_dim, num_class),
                    torch.nn.Sigmoid())
            else:
                answering = torch.nn.Sequential(
                    torch.nn.Linear(hid_dim, num_class),
                    torch.nn.Softmax(dim=1))

            opi_answer = optim.Adam(filter(lambda p: p.requires_grad, answering.parameters()), lr=0.01,
                                    weight_decay=0.00001)
        else:
            answering, opi_answer = None, None
        gnn.to(device)
        PG.to(device)
        return gnn, PG, opi, lossfn, answering, opi_answer
    else:
        raise ValueError("model_create function hasn't supported {} task".format(task_type))



def prompt_w_o_h(dataname="CiteSeer", gnn_type="TransformerConv", pre_train='GraphCL',config_num=0,epoch_num =50 ,num_class=6, task_type='multi_class_classification'):
    _, _, train_list, test_list = multi_class_NIG(dataname, num_class, shots=100)

    train_loader = DataLoader(train_list, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_list, batch_size=10, shuffle=True)

    gnn, PG, opi_pg, lossfn, answering, opi_answer = model_create(dataname, gnn_type, num_class,pre_train, config_num ,epoch_num , task_type, False)
    # Here we have: answering, opi_answer=None, None
    lossfn.to(device)
    

    prompt_epoch = 200
    # training stage
    PG.train()
    for j in range(1, prompt_epoch + 1):
        running_loss = 0.
        for batch_id, train_batch in enumerate(train_loader):
            # print(train_batch)
            train_batch = train_batch.to(device)
            emb0 = gnn(train_batch.x, train_batch.edge_index, train_batch.batch)
            pg_batch = PG.inner_structure_update()
            pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
            # cross link between prompt and input graphs
            dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
            if task_type == 'multi_class_classification':
                sim = torch.softmax(dot, dim=1)
            elif task_type == 'regression':
                sim = torch.sigmoid(dot)  # 0-1
            else:
                raise KeyError("task type error!")

            train_loss = lossfn(sim, train_batch.y)
            opi_pg.zero_grad()
            train_loss.backward()
            opi_pg.step()
            running_loss += train_loss.item()

            if batch_id % 5 == 4:  # report every 5 updates
                last_loss = running_loss / 5  # loss per batch
                print(
                    'epoch {}/{} | batch {}/{} | loss: {:.8f}'.format(j, prompt_epoch, batch_id+1, len(train_loader),
                                                                      last_loss))

                running_loss = 0.

        if j % 5 == 0:
            PG.eval()
            PG = PG.to("cpu")
            gnn = gnn.to("cpu")
            acc_f1_over_batches(test_loader, PG, gnn, answering, num_class, task_type, device)

            PG.train()
            PG = PG.to(device)
            gnn = gnn.to(device) 


def train_one_outer_epoch(epoch, train_loader, opi, lossfn, gnn, PG, answering):
    for j in range(1, epoch + 1):
        running_loss = 0.
        # bar2=tqdm(enumerate(train_loader))
        for batch_id, train_batch in enumerate(train_loader):  # bar2
            # print(train_batch)
            train_batch = train_batch.to(device)
            prompted_graph = PG(train_batch)
            # print(prompted_graph)

            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            # print(graph_emb)
            pre = answering(graph_emb)
            # print(pre)
            train_loss = lossfn(pre, train_batch.y)
            # print('\t\t==> answer_epoch {}/{} | batch {} | loss: {:.8f}'.format(j, answer_epoch, batch_id,
            #                                                                     train_loss.item()))

            opi.zero_grad()
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()

            if batch_id % 5 == 4:  # report every 5 updates
                last_loss = running_loss / 5  # loss per batch
                # bar2.set_description('answer_epoch {}/{} | batch {} | loss: {:.8f}'.format(j, answer_epoch, batch_id,
                #                                                                     last_loss))
                print(
                    'epoch {}/{} | batch {}/{} | loss: {:.8f}'.format(j, epoch, batch_id, len(train_loader), last_loss))

                running_loss = 0.


def prompt_w_h(dataname="CiteSeer", gnn_type="TransformerConv", pre_train='GraphCL',epoch_num =50 ,num_class=6, task_type='multi_class_classification'):
    _, _, train_list, test_list = multi_class_NIG(dataname, num_class, shots=100)

    train_loader = DataLoader(train_list, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_list, batch_size=10, shuffle=True)

    gnn, PG, opi_pg, lossfn, answering, opi_answer = model_create(dataname, gnn_type, num_class,pre_train, epoch_num , task_type, True)
    answering.to(device)

    # inspired by: Hou Y et al. MetaPrompting: Learning to Learn Better Prompts. COLING 2022
    # if we tune the answering function, we update answering and prompt alternately.
    # ignore the outer_epoch if you do not wish to tune any use any answering function
    # (such as a hand-crafted answering template as prompt_w_o_h)
    outer_epoch = 10
    answer_epoch = 20  # 50
    prompt_epoch = 20  # 50

    # training stage
    for i in range(1, outer_epoch + 1):
        print(("{}/{} frozen gnn | frozen prompt | *tune answering function...".format(i, outer_epoch)))
        # tune task head
        answering.train()
        PG.eval()
        train_one_outer_epoch(answer_epoch, train_loader, opi_answer, lossfn, gnn, PG, answering)

        print("{}/{}  frozen gnn | *tune prompt |frozen answering function...".format(i, outer_epoch))
        # tune prompt
        answering.eval()
        PG.train()
        train_one_outer_epoch(prompt_epoch, train_loader, opi_pg, lossfn, gnn, PG, answering)

        # testing stage
        answering.eval()
        PG.eval()
        acc_f1_over_batches(test_loader, PG, gnn, answering, num_class, task_type, device = device)     


import argparse
import torch

def run_program(args):
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda:6")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Example function call (replace with actual function)
    print(f"Running with dataname={args.dataname}, gnn_type={args.gnn_type}, pre_train={args.pre_train}, epoch_num={args.epoch_num}, num_class={args.num_class}")


if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda:6")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")

    print(device)
    # device = torch.device('cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, required=True)
    parser.add_argument("--gnn_type", type=str, required=True)
    parser.add_argument("--pre_train", type=str, required=True)
    parser.add_argument("--config_num", type=int, required=True)
    parser.add_argument("--epoch_num", type=int, required=True)
    parser.add_argument("--num_class", type=int, required=True)
    args = parser.parse_args()
    # pretrain()
    # prompt_w_o_h(dataname="Cora", gnn_type="TransformerConv", num_class=7, task_type='multi_class_classification')
    # prompt_w_h(dataname="Cora", gnn_type="TransformerConv", num_class=7, task_type='multi_class_classification')
    print(f"Running with dataname={args.dataname}, gnn_type={args.gnn_type}, pre_train={args.pre_train}, epoch_num={args.epoch_num}, num_class={args.num_class}")
    
    prompt_w_o_h(dataname=args.dataname, gnn_type=args.gnn_type, pre_train=args.pre_train,config_num=args.config_num ,epoch_num=args.epoch_num, num_class=args.num_class, task_type="multi_class_classification")
    # prompt_w_h(dataname="CiteSeer", gnn_type="TransformerConv", num_class=6, task_type='multi_class_classification')
