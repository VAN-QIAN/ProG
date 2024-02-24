from ProG.pretrain import Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE

# pt = Edgepred_GPPT(dataset_name = 'Cora', gnn_type = 'GCN', hid_dim = 128, gln =3, num_epoch=100)

# pt = Edgepred_GPPT(dataset_name = 'MUTAG', gnn_type = 'GCN', hid_dim = 128, gln =3, num_epoch=100)
# pt = Edgepred_Gprompt(dataset_name = 'Cora', gnn_type = 'GCN', hid_dim = 128, gln =3, num_epoch=100)
pt = GraphCL(dataset_name = 'ENZYMES', gnn_type = 'GCN', hid_dim = 128, gln =3, num_epoch=50)
# pt = SimGRACE(dataset_name = 'MUTAG', gnn_type = 'GCN', hid_dim = 128, gln =3, num_epoch=50)

pt.pretrain()
