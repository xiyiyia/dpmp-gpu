import torch

import task.common as util

def import_model():
    model = torch.hub.load('huggingface/pytorch-transformers', 'model',
                           'bert-base-cased')
    util.set_fullname(model, 'bert_base')
    return model

def import_data(batch_size):
    data_0 = torch.randint(5000, size=[batch_size, 251])
    data_1 = torch.randint(low=0, high=2, size=[batch_size, 251])
    data = torch.cat((data_0.view(-1), data_1.view(-1)))
    
    target_0 = torch.rand(batch_size, 251, 768)
    target_1 = torch.rand(batch_size, 768)
    target = (target_0, target_1)

    return data, target

def partition_model(model):
    group_list = []
    childs = list(model.children())
    group_list.append([childs[0]])

    for c in childs[1].children():
        for sc in c.children():
            for ssc in sc.children():
                group_list.append([ssc])
    for c in childs[2].children():
        group_list.append([c])
    assert len(childs) == 3

    return group_list