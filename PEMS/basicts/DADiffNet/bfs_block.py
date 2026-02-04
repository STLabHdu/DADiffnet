import math
import queue
from collections import defaultdict
from copy import deepcopy

import numpy
import torch


def get_link_list(adj, num_nodes):
    res = []
    for i in range(num_nodes):
        temp = []
        for j in range(num_nodes):
            if adj[i][j] != 0:
                temp.append(j)
        res.append(deepcopy(temp))
    return res



def get_tree_emb_onDis_link_list(adj_mx, num_nodes, padType, device, max_dis, dis_adj_mx):
    # print(self.adj_mx[44])
    graph_emb = []
    for i in range(len(adj_mx)):
        temp_emb = [i]
        index_set = set()
        index_set.add(i)
        index_queue = queue.Queue()
        index_queue.put((i, max_dis))
        flag = True
        while (index_queue.empty() == False) and flag == True:
            flag = False
            temp_size = index_queue.qsize()
            for k in range(temp_size):
                temp, dis = index_queue.get()
                for j in range(len(adj_mx[temp])):
                    if adj_mx[temp][j] != temp and adj_mx[temp][j] not in index_set and dis >= dis_adj_mx[temp][adj_mx[temp][j]]:
                        flag = True
                        index_set.add(adj_mx[temp][j])
                        index_queue.put((adj_mx[temp][j], dis-dis_adj_mx[temp][adj_mx[temp][j]]))
                        temp_emb.append(adj_mx[temp][j])
        graph_emb.append(deepcopy(temp_emb))

    graph_emb_len = 0
    if padType == "avg":
        graph_emb_len = math.ceil(getAverLength(graph_emb))
        for i in range(num_nodes):
            graph_emb[i] = modifyListLen(graph_emb[i], graph_emb_len)
    elif padType == "max":
        graph_emb_len = getMaximumLength(graph_emb)
        for i in range(num_nodes):
            graph_emb[i] = modifyListLen(graph_emb[i], graph_emb_len)

    graph_emb_tensor = torch.tensor(graph_emb, dtype=torch.float).to(device)

    return graph_emb_len, graph_emb_tensor

def get_link_listAndDecay(adj, num_nodes):
    res = []
    for i in range(num_nodes):
        temp = []
        for j in range(num_nodes):
            # Use proper tensor indexing
            if adj[i, j].item() != 0:  # .item() gets the Python scalar value
                temp.append((j, float(adj[i, j])))
        res.append(deepcopy(temp))
    return res

def get_treeEmbAndDecay_link_list(adj_mx, emb_depth, num_nodes, padType, device):
    # print(self.adj_mx[44])
    graph_emb = []
    for i in range(len(adj_mx)):
        temp_emb_depth = emb_depth
        temp_emb = [(i,1)]
        index_set = set()
        index_set.add(i)
        index_queue = queue.Queue()
        index_queue.put((i, 1))
        flag = True
        while temp_emb_depth != 0 and (index_queue.empty() == False) and flag == True:
            flag = False
            temp_size = index_queue.qsize()
            for k in range(temp_size):
                temp = index_queue.get()
                for j in range(len(adj_mx[temp[0]])):
                    if adj_mx[temp[0]][j][0] != temp[0] and adj_mx[temp[0]][j][0] not in index_set:
                        flag = True
                        index_set.add(adj_mx[temp[0]][j][0])
                        index_queue.put((adj_mx[temp[0]][j][0], temp[1]*adj_mx[temp[0]][j][1]))
                        temp_emb.append((adj_mx[temp[0]][j][0], temp[1]*adj_mx[temp[0]][j][1]))
            temp_emb_depth -= 1
        graph_emb.append(deepcopy(temp_emb))

    graph_emb_len = 0
    if padType == "avg":
        graph_emb_len = math.ceil(getAverLength(graph_emb))
        for i in range(num_nodes):
            graph_emb[i] = modifyListLenWithDecay(graph_emb[i], graph_emb_len)
    elif padType == "max":
        graph_emb_len = getMaximumLength(graph_emb)
        for i in range(num_nodes):
            graph_emb[i] = modifyListLenWithDecay(graph_emb[i], graph_emb_len)

    graph_emb_tensor = torch.tensor(graph_emb, dtype=torch.float).to(device)

    return graph_emb_len, graph_emb_tensor

def get_treeEmbAndDecayAndTarLen_link_list(adj_mx, emb_depth, num_nodes, padType, device):
    # print(self.adj_mx[44])
    graph_emb = []
    tar_len_emb = []
    for i in range(len(adj_mx)):
        temp_emb_depth = emb_depth
        temp_emb = [(i,1)]
        tar_len_temp_emb = [(i,1)]
        index_set = set()
        index_set.add(i)
        index_queue = queue.Queue()
        index_queue.put((i, 1))
        flag = True
        while temp_emb_depth != 0 and (index_queue.empty() == False) and flag == True:
            flag = False
            temp_size = index_queue.qsize()
            for k in range(temp_size):
                temp = index_queue.get()
                for j in range(len(adj_mx[temp[0]])):
                    if adj_mx[temp[0]][j][0] != temp[0] and adj_mx[temp[0]][j][0] not in index_set:
                        flag = True
                        index_set.add(adj_mx[temp[0]][j][0])
                        index_queue.put((adj_mx[temp[0]][j][0], temp[1]*adj_mx[temp[0]][j][1]))
                        temp_emb.append((adj_mx[temp[0]][j][0], temp[1]*adj_mx[temp[0]][j][1]))
                        if temp_emb_depth == 1:
                            tar_len_temp_emb.append((adj_mx[temp[0]][j][0], temp[1]*adj_mx[temp[0]][j][1]))
            temp_emb_depth -= 1
        graph_emb.append(deepcopy(temp_emb))
        tar_len_emb.append(deepcopy(tar_len_temp_emb))

    graph_emb_len = 0
    tar_len_emb_len = 0
    if padType == "avg":
        graph_emb_len = math.ceil(getAverLength(graph_emb))
        tar_len_emb_len = math.ceil(getAverLength(tar_len_emb))
        for i in range(num_nodes):
            graph_emb[i] = modifyListLenWithDecay(graph_emb[i], graph_emb_len)
            tar_len_emb[i] = modifyListLenWithDecay(tar_len_emb[i], tar_len_emb_len)
    elif padType == "max":
        graph_emb_len = getMaximumLength(graph_emb)
        tar_len_emb_len = getMaximumLength(tar_len_emb)
        for i in range(num_nodes):
            graph_emb[i] = modifyListLenWithDecay(graph_emb[i], graph_emb_len)
            tar_len_emb[i] = modifyListLenWithDecay(tar_len_emb[i], tar_len_emb_len)

    graph_emb_tensor = torch.tensor(graph_emb, dtype=torch.float).to(device)
    tar_len_emb_tensor = torch.tensor(tar_len_emb, dtype=torch.float).to(device)

    return graph_emb_len, graph_emb_tensor, tar_len_emb_tensor

def get_tree_emb_link_list(adj_mx, emb_depth, num_nodes, padType, device):
    # print(self.adj_mx[44])
    graph_emb = []
    for i in range(len(adj_mx)):
        temp_emb_depth = emb_depth
        temp_emb = [i]
        index_set = set()
        index_set.add(i)
        index_queue = queue.Queue()
        index_queue.put(i)
        flag = True
        while temp_emb_depth != 0 and (index_queue.empty() == False) and flag == True:
            flag = False
            temp_size = index_queue.qsize()
            for k in range(temp_size):
                temp = index_queue.get()
                for j in range(len(adj_mx[temp])):
                    if adj_mx[temp][j] != temp and adj_mx[temp][j] not in index_set:
                        flag = True
                        index_set.add(adj_mx[temp][j])
                        index_queue.put(adj_mx[temp][j])
                        temp_emb.append(adj_mx[temp][j])
            temp_emb_depth -= 1
        graph_emb.append(deepcopy(temp_emb))

    graph_emb_len = 0
    if padType == "avg":
        graph_emb_len = math.ceil(getAverLength(graph_emb))
        for i in range(num_nodes):
            graph_emb[i] = modifyListLen(graph_emb[i], graph_emb_len)
    elif padType == "max":
        graph_emb_len = getMaximumLength(graph_emb)
        for i in range(num_nodes):
            graph_emb[i] = modifyListLen(graph_emb[i], graph_emb_len)

    graph_emb_tensor = torch.tensor(graph_emb, dtype=torch.float).to(device)

    return graph_emb_len, graph_emb_tensor

def get_tree_emb_link_list_on_Tstep(adj_mx, emb_depth, num_nodes, padType, device):
    # print(self.adj_mx[44])
    graph_emb = []
    for i in range(len(adj_mx)):
        temp_emb_depth = emb_depth
        temp_emb = []
        index_set = set()
        index_set.add(i)
        index_queue = queue.Queue()
        index_queue.put(i)
        flag = True
        while temp_emb_depth != 0 and (index_queue.empty() == False) and flag == True:
            flag = False
            temp_size = index_queue.qsize()
            for k in range(temp_size):
                temp = index_queue.get()
                for j in range(len(adj_mx[temp])):
                    if adj_mx[temp][j] != temp and adj_mx[temp][j] not in index_set:
                        flag = True
                        index_set.add(adj_mx[temp][j])
                        index_queue.put(adj_mx[temp][j])
                        if temp_emb_depth == 1:
                            temp_emb.append(adj_mx[temp][j])
            temp_emb_depth -= 1
        graph_emb.append(deepcopy(temp_emb))

    graph_emb_len = 0
    if padType == "avg":
        graph_emb_len = math.ceil(getAverLength(graph_emb))
        for i in range(num_nodes):
            graph_emb[i] = modifyListLen(graph_emb[i], graph_emb_len)
    elif padType == "max":
        graph_emb_len = getMaximumLength(graph_emb)
        for i in range(num_nodes):
            graph_emb[i] = modifyListLen(graph_emb[i], graph_emb_len)

    graph_emb_tensor = torch.tensor(graph_emb, dtype=torch.float).to(device)

    return graph_emb_len, graph_emb_tensor

def get_tree_emb_link_list_greaterThanTarStep(adj_mx, emb_depth, tar_step, num_nodes, padType, device):
    # print(self.adj_mx[44])
    graph_emb = []
    for i in range(len(adj_mx)):
        temp_emb_depth = emb_depth
        temp_emb = []
        index_set = set()
        index_set.add(i)
        index_queue = queue.Queue()
        index_queue.put(i)
        flag = True
        while temp_emb_depth != 0 and (index_queue.empty() == False) and flag == True:
            flag = False
            temp_size = index_queue.qsize()
            for k in range(temp_size):
                temp = index_queue.get()
                for j in range(len(adj_mx[temp])):
                    if adj_mx[temp][j] != temp and adj_mx[temp][j] not in index_set:
                        flag = True
                        index_set.add(adj_mx[temp][j])
                        index_queue.put(adj_mx[temp][j])
                        if temp_emb_depth >= tar_step:
                            temp_emb.append(adj_mx[temp][j])
            temp_emb_depth -= 1
        graph_emb.append(deepcopy(temp_emb))

    graph_emb_len = 0
    if padType == "avg":
        graph_emb_len = math.ceil(getAverLength(graph_emb))
        for i in range(num_nodes):
            graph_emb[i] = modifyListLen(graph_emb[i], graph_emb_len)
    elif padType == "max":
        graph_emb_len = getMaximumLength(graph_emb)
        for i in range(num_nodes):
            graph_emb[i] = modifyListLen(graph_emb[i], graph_emb_len)

    graph_emb_tensor = torch.tensor(graph_emb, dtype=torch.float).to(device)

    return graph_emb_len, graph_emb_tensor

def get_tree_emb(adj_mx, emb_depth, num_nodes, padType, device):
    # print(self.adj_mx[44])
    adj_type = type(adj_mx)
    graph_emb = []
    for i in range(len(adj_mx)):
        temp_emb_depth = emb_depth
        temp_emb = [i]
        index_set = set()
        index_set.add(i)
        index_queue = queue.Queue()
        index_queue.put(i)
        flag = True
        while temp_emb_depth != 0 and (index_queue.empty() == False) and flag == True:
            flag = False
            temp_size = index_queue.qsize()
            for k in range(temp_size):
                temp = index_queue.get()
                for j in range(len(adj_mx[temp])):
                    if j != temp and adj_mx[temp][j] != 0 and j not in index_set:
                        flag = True
                        index_set.add(j)
                        index_queue.put(j)
                        temp_emb.append(j)
            temp_emb_depth -= 1
        graph_emb.append(deepcopy(temp_emb))

    graph_emb_len = 0

    if padType == "avg":
        graph_emb_len = math.ceil(getAverLength(graph_emb))
        for i in range(num_nodes):
            graph_emb[i] = modifyListLen(graph_emb[i], graph_emb_len)
    elif padType == "max":
        graph_emb_len = getMaximumLength(graph_emb)
        for i in range(num_nodes):
            graph_emb[i] = modifyListLen(graph_emb[i], graph_emb_len)

    graph_emb_tensor = torch.tensor(graph_emb, dtype=torch.float).to(device)

    return graph_emb_len, graph_emb_tensor

def expand_adj_mx_link_list(adj_mx, dis_adj_mx, emb_depth, device):
    # print(self.adj_mx[44])
    row_length = len(adj_mx)
    expand_ed_adj_mx = []
    for i in range(row_length):
        expanded_row_mx = [] + row_length * [0.]
        expanded_row_mx[i] = 1.0
        temp_emb_depth = emb_depth
        index_set = set()
        index_set.add(i)
        index_queue = queue.Queue()
        index_queue.put(i)
        flag = True
        while temp_emb_depth != 0 and (index_queue.empty() == False) and flag == True:
            depth_set = set()
            flag = False
            temp_size = index_queue.qsize()
            for k in range(temp_size):
                temp = index_queue.get()
                for j in range(len(adj_mx[temp])):
                    if adj_mx[temp][j] != temp and adj_mx[temp][j] not in index_set:
                        flag = True
                        if adj_mx[temp][j] not in depth_set:
                            index_queue.put(adj_mx[temp][j])
                        depth_set.add(adj_mx[temp][j])
                        expanded_row_mx[adj_mx[temp][j]] = expanded_row_mx[adj_mx[temp][j]] + \
                                                           expanded_row_mx[temp] * dis_adj_mx[temp][adj_mx[temp][j]]
            index_set.update(depth_set)
            temp_emb_depth -= 1
        expand_ed_adj_mx.append(deepcopy(expanded_row_mx))

    graph_emb_tensor = torch.tensor(expand_ed_adj_mx, dtype=torch.float).to(device)

    return graph_emb_tensor

def modifyListLen(lst, tar_len):
    pre_len = len(lst)
    if pre_len > tar_len:
        lst = lst[:tar_len]
    elif pre_len < tar_len:
        lst = lst + (tar_len - pre_len) * [-1]
    return lst

def modifyListLenWithDecay(lst, tar_len):
    pre_len = len(lst)
    if pre_len > tar_len:
        lst = lst[:tar_len]
    elif pre_len < tar_len:
        lst = lst + (tar_len - pre_len) * [(-1, -1)]
    return lst


def getAverLength(lst):
    len_of_lst = len(lst)
    temp = 0
    for l in lst:
        temp += len(l)
    return temp / len_of_lst


def getMaximumLength(lst):
    maximum = 0
    for l in lst:
        maximum = max(maximum, len(l))
    return maximum


def printArray(lst):
    for row in lst:
        for elem in row:
            print(elem, end=' ')
        print()  # 用于在每一行结束后换行


def count_element_lengths(lst):
    length_count = defaultdict(int)
    for element in lst:
        length_count[len(element)] += 1
    return length_count