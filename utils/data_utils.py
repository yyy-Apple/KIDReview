import dgl
import torch

NODE_TYPE = {'entity': 0, 'root': 1, 'relation': 2}


class Datum:
    """
    A single data point. Contains id, title, entity_text, entity_type, relation, oracle, review,
    reference embedding and citation embedding.
    """
    def __init__(self, paper_id, title, oracle, intro, ext, abs_ext, review,
                 entities, types, relations, ref_emb, citation_emb):
        self.paper_id = paper_id
        self.raw_title = title
        self.raw_oracle = oracle
        self.raw_intro = intro
        self.raw_ext = ext
        self.raw_abs_ext = abs_ext
        self.raw_review = review
        self.raw_entities = entities
        self.raw_types = types.split(" ")
        self.raw_relations = []
        for relation in relations:
            words = relation.split(" ")
            for i in range(len(words)):
                if words[i] == '--' and words[i + 2] == '--' and words[i + 1].upper() == words[i + 1]:
                    head = " ".join(words[:i])
                    relation_text = words[i] + words[i + 1] + words[i + 2]
                    tail = " ".join(words[i + 3:])

                    # Make sure entities in relations are recognized
                    if head in self.raw_entities and tail in self.raw_entities:
                        self.raw_relations.append([head, relation_text, tail])
                    break
        self.ref_emb = ref_emb
        self.citation_emb = citation_emb
        self.graph = self.build_graph()

    @classmethod
    def from_json(cls, json_data):
        return cls(json_data['id'], json_data['title'], json_data['oracle'], json_data['intro'], json_data['ext'],
                   json_data['abs_ext'], json_data['review'], json_data['entities'], json_data['types'],
                   json_data['relations'],  json_data['ref_emb'], json_data['citation_emb'])

    def __str__(self):
        return '\n'.join([str(k) + ":\t" + str(v) for k, v in self.__dict__.items()])

    def __len__(self):
        return len(self.raw_review)

    def build_graph(self):
        """ Build the graph as the paper described. All relations are converted to nodes, add a global node. All
            entities are connected to the global node, all nodes (entity, relation, root) has a self-loop edge.
            For nodes,
            1. First, we add all entity nodes.
            2. Then, we add the root node.
            3. Finally, we add the relations and inverse relations nodes.
            For edges,
            1. All entity nodes are connected to the root node.
            2. All nodes has self-loop edge.
            3. Add adjacent edges.
        """
        graph = dgl.DGLGraph()
        entity_num = len(self.raw_entities)
        relation_num = len(self.raw_relations)

        # Add nodes
        graph.add_nodes(entity_num, {'type': torch.ones(entity_num) * NODE_TYPE['entity']})
        graph.add_nodes(1, {'type': torch.ones(1) * NODE_TYPE['root']})
        graph.add_nodes(2 * relation_num, {'type': torch.ones(2 * relation_num) * NODE_TYPE['relation']})

        # Add edges
        graph.add_edges(list(range(entity_num)), entity_num)
        graph.add_edges(entity_num, list(range(entity_num)))
        graph.add_edges(list(range(entity_num + 1 + 2 * relation_num)), list(range(entity_num + 1 + 2 * relation_num)))

        u, v = [], []

        for i, relation in enumerate(self.raw_relations):
            head_idx = self.raw_entities.index(relation[0])
            tail_idx = self.raw_entities.index(relation[2])
            rel_idx = entity_num + 1 + 2 * i
            rel_inv_idx = rel_idx + 1
            # Add head -> rel, rel -> tail, tail -> rel_inv, rel_env -> head
            u.extend([head_idx, rel_idx, tail_idx, rel_inv_idx])
            v.extend([rel_idx, tail_idx, rel_inv_idx, head_idx])

        if len(u) > 0:
            graph.add_edges(u, v)
        return graph

