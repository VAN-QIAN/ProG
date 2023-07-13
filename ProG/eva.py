import torch
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
import warnings
import numpy as np
import torchmetrics



class Evaluator:
    def __init__(self, eval_metric='hits@50'):

        self.eval_metric = eval_metric
        if 'hits@' in self.eval_metric:
            self.K = int(self.eval_metric.split('@')[1])

    def _parse_and_check_input(self, input_dict):
        if 'hits@' in self.eval_metric:
            if not 'y_pred_pos' in input_dict:
                raise RuntimeError('Missing key of y_pred_pos')
            if not 'y_pred_neg' in input_dict:
                raise RuntimeError('Missing key of y_pred_neg')

            y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

            if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
                raise ValueError('y_pred_pos needs to be either numpy ndarray or torch tensor')

            if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
                raise ValueError('y_pred_neg needs to be either numpy ndarray or torch tensor')

            if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = torch.from_numpy(y_pred_pos)

                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = torch.from_numpy(y_pred_neg)

                y_pred_pos = y_pred_pos.to(y_pred_neg.device)

                type_info = 'torch'

            else:
                type_info = 'numpy'

            if not y_pred_pos.ndim == 1:
                raise RuntimeError('y_pred_pos must to 1-dim arrray, {}-dim array given'.format(y_pred_pos.ndim))

            return y_pred_pos, y_pred_neg, type_info

        elif 'mrr' == self.eval_metric:

            if not 'y_pred_pos' in input_dict:
                raise RuntimeError('Missing key of y_pred_pos')
            if not 'y_pred_neg' in input_dict:
                raise RuntimeError('Missing key of y_pred_neg')

            y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

            if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
                raise ValueError('y_pred_pos needs to be either numpy ndarray or torch tensor')

            if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
                raise ValueError('y_pred_neg needs to be either numpy ndarray or torch tensor')

            if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = torch.from_numpy(y_pred_pos)

                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = torch.from_numpy(y_pred_neg)

                y_pred_pos = y_pred_pos.to(y_pred_neg.device)

                type_info = 'torch'
            else:
                type_info = 'numpy'

            if not y_pred_pos.ndim == 1:
                raise RuntimeError('y_pred_pos must to 1-dim arrray, {}-dim array given'.format(y_pred_pos.ndim))

            if not y_pred_neg.ndim == 2:
                raise RuntimeError('y_pred_neg must to 2-dim arrray, {}-dim array given'.format(y_pred_neg.ndim))

            return y_pred_pos, y_pred_neg, type_info

        else:
            raise ValueError('Undefined eval metric %s' % (self.eval_metric))

    def eval(self, input_dict):

        if 'hits@' in self.eval_metric:
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_hits(y_pred_pos, y_pred_neg, type_info)
        elif self.eval_metric == 'mrr':
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_mrr(y_pred_pos, y_pred_neg, type_info)

        else:
            raise ValueError('Undefined eval metric %s' % (self.eval_metric))




    def _eval_hits(self, y_pred_pos, y_pred_neg, type_info):

        if type_info == 'torch':
            res=torch.topk(y_pred_neg, self.K)
            kth_score_in_negative_edges=res[0][:,-1]
            hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

        else:
            kth_score_in_negative_edges = np.sort(y_pred_neg)[-self.K]
            hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

        return {'hits@{}'.format(self.K): hitsK}

    def _eval_mrr(self, y_pred_pos, y_pred_neg, type_info):
        if type_info == 'torch':
            y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
            argsort = torch.argsort(y_pred, dim=1, descending=True)
            ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
            ranking_list = ranking_list[:, 1] + 1
            mrr_list = 1. / ranking_list.to(torch.float)
            return mrr_list.mean()
        else:
            y_pred = np.concatenate([y_pred_pos.reshape(-1, 1), y_pred_neg], axis=1)
            argsort = np.argsort(-y_pred, axis=1)
            ranking_list = (argsort == 0).nonzero()
            ranking_list = ranking_list[1] + 1
            mrr_list = 1. / ranking_list.astype(np.float32)
            return mrr_list.mean()


def mrr_hit(normal_label: np.ndarray, pos_out: np.ndarray, metric: list = None):
    if isinstance(normal_label, np.ndarray) and isinstance(pos_out, np.ndarray):
        pass
    else:
        warnings.warn('it would be better if normal_label and out are all set as np.ndarray')

    results = {}
    if not metric:
        metric = ['mrr', 'hits']

    if 'hits' in metric:
        hits_evaluator = Evaluator(eval_metric='hits@50')
        flag = normal_label
        pos_test_pred = torch.from_numpy(pos_out[flag == 1])
        neg_test_pred = torch.from_numpy(pos_out[flag == 0])

        for N in [100]:
            neg_test_pred_N = neg_test_pred.view(-1, 100)
            for K in [1, 5, 10]:
                hits_evaluator.K = K
                test_hits = hits_evaluator.eval({
                    'y_pred_pos': pos_test_pred,
                    'y_pred_neg': neg_test_pred_N,
                })[f'hits@{K}']

                results[f'Hits@{K}@{N}'] = test_hits

    if 'mrr' in metric:
        mrr_evaluator = Evaluator(eval_metric='mrr')
        flag = normal_label
        pos_test_pred = torch.from_numpy(pos_out[flag == 1])
        neg_test_pred = torch.from_numpy(pos_out[flag == 0])

        neg_test_pred = neg_test_pred.view(-1, 100)

        mrr = mrr_evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })

        if isinstance(mrr, torch.Tensor):
            mrr = mrr.item()
        results['mrr'] = mrr
    return results

def eva(pre, label, task_type='multi_class_classification'):
    if task_type == 'regression':
        mae = mean_absolute_error(label, pre)
        mse = mean_squared_error(label, pre)
        return {"mae": mae, "mse": mse}
    elif task_type == 'multi_class_classification':
        pre_cla = torch.argmax(pre, dim=1)
        acc = accuracy_score(label, pre_cla)
        mac_f1 = f1_score(label, pre_cla, average='macro')
        mic_f1 = f1_score(label, pre_cla, average='micro')
        return {"acc": acc, "mac_f1": mac_f1, "mic_f1": mic_f1}
    elif task_type == 'link_prediction':
        normal_label = label
        pos_out = pre[:, 1]
        results = mrr_hit(normal_label, pos_out)
        return results
    else:
        raise NotImplemented(
            "eva() function is currently only used for multi-class classification  and link_prediction tasks!")


def testing_tune_answer(test_batch, PG, gnn, answering,lossfn, task_type='multi_class_classification'):
        print("testing tune answer...")
        prompted_graph = PG(test_batch)

        graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        # print(graph_emb)
        pre = answering(graph_emb)
        # print(pre)
        v_loss = lossfn(pre, test_batch.y)
        # print('\t\t==> answer_epoch {}/{} | batch {} | loss: {:.8f}'.format(j, answer_epoch, batch_id,
        #                                                                     train_loss.item()))

        pre = pre.detach()
        print("calculate results...")
        res = eva(pre, test_batch.y, task_type=task_type)
        return res,v_loss




def testing(test, PG, gnn,task_type='multi_class_classification'):
    """
    You should first use PG.eval() before you call this function
    :param test:
    :param PG:
    :param gnn:
    :param task_type:
    :return:
    """


    emb0 = gnn(test.x, test.edge_index, test.batch)
    pg_batch = PG.token_view()
    pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
    dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))

    if task_type == 'multi_class_classification':
        pre = torch.softmax(dot, dim=1)
    elif task_type == 'regression':
        pre = torch.sigmoid(dot)
        pre = pre.detach()

    res = eva(pre, test.y, task_type=task_type)
    return res


def calculate_accuracy_over_batches(test_loader,PG,gnn,answering,num_class, task_type):
    correct = 0
    test_num = 0
    if task_type=="multi_class_classification":
        metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class)
    else:
        raise NotImplementedError


    for batch_id, test_batch in enumerate(test_loader):  # bar2

        prompted_graph = PG(test_batch)

        graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        # print(graph_emb)
        pre = answering(graph_emb)

        pre = pre.detach()
        y = test_batch.y

        pre_cla = torch.argmax(pre, dim=1)
        # print(pre_cla)
        # print(y)

        acc = metric(pre_cla, y)
        print(f"Accuracy on batch {batch_id}: {acc}")

    #
    #     correct += torch.count_nonzero(pre_cla == y)
    #     test_num += pre.shape[0]
    #     if batch_id%10==0:
    #         print("accumulated ACC @ batch {}/{} acc: {}".format(batch_id, len(test_loader), correct * 1.0 / test_num))
    #
    # acc = correct * 1.0 / test_num
    acc = metric.compute()
    print("Final True ACC: ", acc.item())
    metric.reset()