class Data:

    def __init__(self, data_dir="data/", reverse=False, add_constraint=True):
        self.add_constraint = add_constraint
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.test_data

        train_data2 = self.load_data2(data_dir, "train", reverse=reverse)
        test_data2 = self.load_data2(data_dir, "test", reverse=reverse)
        data2 = train_data2 + test_data2
        self.entities = self.get_entities(data2)

        self.train_relations = self.get_relations(self.train_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.test_relations \
                if i not in self.train_relations]
        print(self.relations)
        print(len(self.relations))

    def load_data2(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            if self.add_constraint==True:
                data = [i.split() for i in data]
            else:
                res = []
                for i in data:
                    if 'NOT' in i:
                        continue
                    else:
                        res.append(i.split())
                data = res
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities
