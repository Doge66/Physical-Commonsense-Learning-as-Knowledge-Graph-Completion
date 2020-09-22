import sys
import torch

from data import get

class Sample(object):
    def __init__(self, name=None, label=None):
        self.name = name
        self.label = label

class Task(object):    
    def __init__(self, task):
        self.train_data, self.test_data = get(task)
        
    def get_train_examples(self):
        names, labels = self.train_data
        return self._create_examples(names, labels)

    def get_test_examples(self):
        names, labels = self.test_data
        return self._create_examples(names, labels)

    def _create_examples(self, names, labels):
        samples = []
        for name, label in zip(names, labels):
            samples.append(Sample(name=name, label=str(label.item())))
        return samples

# gather objects, properties, and affordances
def _distinct_first_second(task):
    first = set()
    second = set()
                
    train_samples = task.get_train_examples()
    for sample in train_samples:
        first.add(sample.name.split('/')[0])
        second.add(sample.name.split('/')[1])
        
    dev_samples = task.get_test_examples()
    for sample in dev_samples:
        first.add(sample.name.split('/')[0])
        second.add(sample.name.split('/')[1])
    
    return first, second

def get_entity_sets(task_mapping):
    objects = set()
    properties = set()
    affordances = set()
    for name, task in task_mapping.items():        
        print('-'*30)
        print(name)
        first, second = _distinct_first_second(task)
        if name == 'situated-OP':
            objects = objects.union(first)
            properties = properties.union(second)
        if name == 'situated-OA':
            objects = objects.union(first)
            affordances = affordances.union(second)
        if name == 'situated-AP':
            affordances = affordances.union(first)
            properties = properties.union(second)
        print('first: {}, second: {}'.format(len(first), len(second)))
    print('-'*30)
    print('objects: {}, properties: {}: affordances: {}'.format(len(objects), len(properties), len(affordances)))
    return objects, properties, affordances


def openke_predict(model, h, t, rs, truth):
    min_val = sys.float_info.max
    label = 0
    n,_ = rs.shape
    for i in range(n):
        # val = model.predict({'batch_h': torch.from_numpy(h).cuda().long(), 'batch_t': torch.from_numpy(t).cuda().long(),
        #                      'batch_r': torch.from_numpy(rs[i]).cuda().long(), 'mode': 'normal'})
        val = model.predict({'batch_h': torch.from_numpy(h).long(), 'batch_t': torch.from_numpy(t).long(),
                             'batch_r': torch.from_numpy(rs[i]).long(), 'mode': 'normal'})
        if val < min_val:
            min_val = val
            label = i
    return int(label==truth)

def get_entity_relationship_dicts():
    ent_list = {}
    rel_list = {}
    with open('./data/kge/openke/entity2id.txt', 'r') as f:
        content = f.readlines()[1:]
        ent_list = {x.split()[0]:i for i, x in enumerate(content)}
        f.close()
        
    with open('./data/kge/openke/relation2id.txt', 'r') as f:
        content = f.readlines()[1:]
        rel_list = {x.split()[0]:i for i, x in enumerate(content)}
        f.close()
    
    return ent_list, rel_list