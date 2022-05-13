import random
import time

import torch
import torch as th
import torch.nn as nn
import torch.functional as F
import dgl
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import tqdm
from collections import Counter
from operator import itemgetter
from tkinter import _flatten
import numpy as np
from datasketch import MinHashLSHForest, MinHash, MinHashLSH


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def Reordering(self, over_all, indexx):
        over_all_2 = []  # 以字符串形式保存所有重用的节点标号
        for ind_x in range(len(over_all)):  # 先拿到所有的子图，统计信息
            bb = [str(x).encode('utf-8') for x in over_all[ind_x]]
            over_all_2.append(bb)

        # 局部敏感哈希计算相似度,LSH
        names = globals()
        for i in range(len(over_all_2)):  # 定义动态变量并minhash
            names['M' + str(i)] = MinHash(num_perm=128)  # 定义递增动态变量, '1', '2'
            names['M' + str(i)].update_batch(over_all_2[i])
        lsh = MinHashLSH(threshold=0.01, num_perm=128)
        for i in range(len(over_all_2)):
            lsh.insert(i, names['M' + str(i)])  # 插入哈希表
        # print('哈希化完成')
        Indexx = lsh.query2()  # 最整体相似度
        IND = np.array(indexx)[Indexx]
        return IND

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        T = []
        T1 = []
        for l, layer in enumerate(self.layers):

            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
            # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            sampler = dgl.dataloading.MultiLayerDropoutSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()).to(g.device),  # 所有的点
                sampler,
                device=device if num_workers == 0 else None,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers)
            ticc = time.time()
            if l == 0:
                '''没有缓存,每次用上个batch重用的部分,所有的batch重新排序了'''
                # over_all_2 = []  # 以字符串形式保存所有重用的节点标号
                over_all_1 = []  # 以数字形式保存所有重用的节点标号
                output_nodes_1 = []
                blocks_1 = []
                Len_bat = []
                for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):  # 先拿到所有的子图，统计信息
                    input_nodes = input_nodes.tolist()
                    over_all_1.append(input_nodes)
                    output_nodes_1.append(output_nodes)
                    blocks_1.append(blocks)
                    Len_bat.append(len(input_nodes))

                    #
                    # bb = [str(x).encode('utf-8') for x in input_nodes]
                    # over_all_2.append(bb)  # 变成字符穿数组

                # print(Len_bat)
                # print(Len_bat)
                Ind1 = []
                Ind2 = []
                Ind3 = []  # 保存的是下标
                Ind4 = []  # 保存的是下标
                Ind5 = []  # 保存的是下标
                Ind6 = []  # 保存的是下标
                Ind7 = []  # 保存的是下标
                Ind8 = []  # 保存的是下标
                # for in_s in range(len(Len_bat)):  # reddit
                #     # if 0 < Len_bat[in_s] < Len_bat[round(len(Len_bat)/8)]:
                #     #     Ind1.append(in_s)
                #     # elif Len_bat[round(len(Len_bat)/8)] < Len_bat[in_s] < Len_bat[round(len(Len_bat)*2/8)]:
                #     #     Ind2.append(in_s)
                #     # elif Len_bat[round(len(Len_bat)*2/8)] < Len_bat[in_s] < Len_bat[round(len(Len_bat)*3/8)]:
                #     #     Ind3.append(in_s)
                #     # elif Len_bat[round(len(Len_bat)*3/8)] < Len_bat[in_s] < Len_bat[round(len(Len_bat)*4/8)]:
                #     #     Ind4.append(in_s)
                #     # elif Len_bat[round(len(Len_bat)*4/8)] < Len_bat[in_s] < Len_bat[round(len(Len_bat)*5/8)]:
                #     #     Ind5.append(in_s)
                #     # elif Len_bat[round(len(Len_bat)*5/8)] < Len_bat[in_s] < Len_bat[round(len(Len_bat)*6/8)]:
                #     #     Ind6.append(in_s)
                #     # elif Len_bat[round(len(Len_bat)*6/8)] < Len_bat[in_s] < Len_bat[round(len(Len_bat)*7/8)]:
                #     #     Ind7.append(in_s)
                #     # else:
                #     #     Ind8.append(in_s)
                #
                #     if 0 < in_s < round(len(Len_bat)/8):
                #         Ind1.append(in_s)
                #     elif round(len(Len_bat)/8) < in_s < round(len(Len_bat)*2/8):
                #         Ind2.append(in_s)
                #     elif round(len(Len_bat)*2/8) < in_s < round(len(Len_bat)*3/8):
                #         Ind3.append(in_s)
                #     elif round(len(Len_bat)*3/8) < in_s < round(len(Len_bat)*4/8):
                #         Ind4.append(in_s)
                #     elif round(len(Len_bat)*4/8) < in_s < round(len(Len_bat)*5/8):
                #         Ind5.append(in_s)
                #     elif round(len(Len_bat)*5/8) < in_s < round(len(Len_bat)*6/8):
                #         Ind6.append(in_s)
                #     elif round(len(Len_bat)*6/8) < in_s < round(len(Len_bat)*7/8):
                #         Ind7.append(in_s)
                #     else:
                #         Ind8.append(in_s)

                for in_s in range(len(Len_bat)):  # product
                    if 0 < Len_bat[in_s] < 10000:
                        Ind1.append(in_s)
                    elif 10000 < Len_bat[in_s] < 30000:
                        Ind2.append(in_s)
                    else:
                        Ind3.append(in_s)

                print(len(Ind1))
                print(len(Ind2))
                print(len(Ind3))
                print(len(Ind4))
                print(len(Ind5))
                print(len(Ind6))
                print(len(Ind7))
                print(len(Ind8))
                # #
                Find_ord = []
                a1 = np.array(over_all_1)[Ind1]
                # [::10]
                in_i_1 = self.Reordering([a1[ind][::10] for ind in range(len(a1))], Ind1)
                Find_ord.extend(in_i_1)  # 每一类的batches,以及这些batch原来的下标
                a2 = np.array(over_all_1)[Ind2]
                in_i_2 = self.Reordering([a2[ind][::100] for ind in range(len(a2))], Ind2)
                Find_ord.extend(in_i_2)  # 每一类的batches,以及这些batch原来的下标
                a3 = np.array(over_all_1)[Ind3]
                in_i_3 = self.Reordering([a3[ind][::100] for ind in range(len(a3))], Ind3)
                Find_ord.extend(in_i_3)  # 每一类的batches,以及这些batch原来的下标

                # a4 = np.array(over_all_1)[Ind4]
                # in_i_4 = self.Reordering([a4[ind] for ind in range(len(a4))], Ind4)
                # Find_ord.extend(in_i_4)  # 每一类的batches,以及这些batch原来的下标
                # a5 = np.array(over_all_1)[Ind5]
                # in_i_5 = self.Reordering([a5[ind] for ind in range(len(a5))], Ind5)
                # Find_ord.extend(in_i_5)  # 每一类的batches,以及这些batch原来的下标
                # a6 = np.array(over_all_1)[Ind6]
                # in_i_6 = self.Reordering([a6[ind] for ind in range(len(a6))], Ind6)
                # Find_ord.extend(in_i_6)  # 每一类的batches,以及这些batch原来的下标
                # a7 = np.array(over_all_1)[Ind7]
                # in_i_7 = self.Reordering([a7[ind] for ind in range(len(a7))], Ind7)
                # Find_ord.extend(in_i_7)  # 每一类的batches,以及这些batch原来的下标
                # a8 = np.array(over_all_1)[Ind8]
                # in_i_8 = self.Reordering([a8[ind] for ind in range(len(a8))], Ind8)
                # Find_ord.extend(in_i_8)  # 每一类的batches,以及这些batch原来的下标


                # for in_s in range(len(Len_bat)):  # reddit 全部细分类
                #     if 0 < Len_bat[in_s] < 1000000000:
                #         Ind1.append(in_s)
                # Find_ord = []
                # a1 = np.array(over_all_1)[Ind1]
                # in_i_1 = self.Reordering([a1[ind] for ind in range(len(a1))], Ind1)
                # Find_ord.extend(in_i_1)  # 每一类的batches,以及这些batch原来的下标



                # Find_ord.extend(Ind1)  # 只有粗分类
                # Find_ord.extend(Ind2)
                # Find_ord.extend(Ind3)

                # Find_ord = torch.arange(233).tolist()

                # print(Find_ord)
                #  计算需要缓存和替换的内容
                new_inputnodes = [x for _, x in sorted(zip(Find_ord, over_all_1))]  # 按照index重新排序
                new_outputnodes = [x for _, x in sorted(zip(Find_ord, output_nodes_1))]
                new_blocks = [x for _, x in sorted(zip(Find_ord, blocks_1))]
            tocc = time.time()
            print(tocc-ticc)

            input_nodes_temp = []
            gpu_flag_1 = torch.zeros(g.num_nodes()).bool().to(device)
            gpu_flag_1.requires_grad_(False)
            localid2cacheid_1 = torch.cuda.LongTensor(g.num_nodes()).fill_(0).to(device)
            localid2cacheid_1.requires_grad_(False)
            #
            # gpu_flag = torch.zeros(g.num_nodes()).bool().to(device)
            # gpu_flag.requires_grad_(False)
            # localid2cacheid = torch.cuda.LongTensor(g.num_nodes()).fill_(0).to(device)
            # localid2cacheid.requires_grad_(False)

            for r_inter in range(len(over_all_1)):
                blocks = new_blocks[r_inter]
                input_nodes = new_inputnodes[r_inter]
                input_nodes = torch.LongTensor(input_nodes)
                output_nodes = new_outputnodes[r_inter]
                if input_nodes_temp != []:
                    # t0 = time.time()  # 计时
                    # gpu_flag = torch.zeros(g.num_nodes()).bool().to(device)
                    # gpu_flag.requires_grad_(False)
                    # localid2cacheid = torch.cuda.LongTensor(g.num_nodes()).fill_(0).to(device)
                    # localid2cacheid.requires_grad_(False)
                    t0 = time.time()  # 计时
                    gpu_flag = gpu_flag_1.clone()
                    localid2cacheid = localid2cacheid_1.clone()

                    # t0 = time.time()  # 计时
                    gpu_flag[input_nodes_temp] = True  # 缓存起来的点的id是input_nodes_temp
                    t1 = time.time()  # 计时
                    T1.append(t1 - t0)  # 计时

                    t0 = time.time()  # 计时
                    localid2cacheid[input_nodes_temp] = torch.arange(len(input_nodes_temp)).to(device)

                    t1 = time.time()  # 计时
                    T.append(t1 - t0)  # 计时
                    t0 = time.time()  # 计时

                    gpu_mask = gpu_flag[input_nodes]  # 子图中哪些点是在GPU里的
                    nids_in_gpu = input_nodes[gpu_mask]  # 真的的下标
                    cacheid = localid2cacheid[nids_in_gpu]  # 在已经cache的内容中的id
                    cpu_mask = ~gpu_mask
                    nids_in_cpu = input_nodes[cpu_mask]
                    frame = torch.cuda.FloatTensor(input_nodes.size(0), x.size(1)).to(device)  # 定好frame
                    frame[gpu_mask] = h[cacheid]  # 缓存起来的具体值
                    t1 = time.time()  # 计时
                    T1.append(t1 - t0)  # 计时
                    t0 = time.time()  # 计时
                    # h.cpu()
                    frame[cpu_mask] = x[nids_in_cpu].to(device)
                    t1 = time.time()  # 计时
                    T.append(t1 - t0)  # 计时

                    t0 = time.time()  # 计时
                    h = frame.detach()  # 保留h为下一次所用
                    # h = frame
                    block = blocks[0]
                    t1 = time.time()  # 计时
                    T1.append(t1 - t0)  # 计时

                    t0 = time.time()  # 计时
                    block = block.int().to(device)  # block 在这里就是一个图
                    t1 = time.time()  # 计时
                    T.append(t1 - t0)  # 计时

                    t0 = time.time()  # 计时
                    h1 = layer(block, h)
                    if l != len(self.layers) - 1:
                        h1 = self.activation(h1)
                        h1 = self.dropout(h1)
                    y[output_nodes] = h1.cpu()
                    input_nodes_temp = input_nodes
                    t1 = time.time()  # 计时
                    T1.append(t1 - t0)  # 计时
                else:
                    block = blocks[0]
                    t0 = time.time()  # 计时
                    block = block.int().to(device)  # block 在这里就是一个图
                    h = x[input_nodes].to(device)
                    t1 = time.time()  # 计时
                    T.append(t1 - t0)  # 计时
                    t0 = time.time()  # 计时
                    h1 = layer(block, h)  # 降维
                    if l != len(self.layers) - 1:
                        h1 = self.activation(h1)
                        h1 = self.dropout(h1)
                    y[output_nodes] = h1.cpu()
                    input_nodes_temp = input_nodes
                    t1 = time.time()  # 计时
                    T1.append(t1 - t0)  # 计时
            t0 = time.time()  # 计时
            x = y
            t1 = time.time()  # 计时
            T1.append(t1 - t0)  # 计时
        return y, sum(T), sum(T1)


def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test
