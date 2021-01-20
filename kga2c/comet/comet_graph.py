import os
import sys
import argparse
import torch

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
import networkx as nx

class CometHelper():
    def __init__(self, args):
        self.opt, self.state_dict = interactive.load_model_file(args.model_file)
        self.data_loader, self.text_encoder = interactive.load_data("conceptnet", self.opt)
        self.n_ctx = self.data_loader.max_e1 + self.data_loader.max_e2 + self.data_loader.max_r
        self.n_vocab = len(self.text_encoder.encoder) + self.n_ctx  


        self.model = interactive.make_model(self.opt, self.n_vocab, self.n_ctx, self.state_dict)
        
        cfg.device = int(args.device_comet)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        self.model.cuda(cfg.device)

        self.sampling_algorithm = args.sampling_algorithm
        self.relation = args.relation
        self.sampler = interactive.set_sampler(self.opt, self.sampling_algorithm, self.data_loader)
        # self.graph_state = None

    def generate(self, obj):

        # if self.relation not in data.conceptnet_data.conceptnet_relations:
        #     relation = "all"
        outputs = [None] * len(obj)
        for idx,event in enumerate(obj):
            outputs[idx] = interactive.get_conceptnet_sequence(
                event, self.model, self.sampler, self.data_loader, self.text_encoder, self.relation)
        return outputs

    def make_graph(self, obj):
        graph_state = nx.DiGraph()
        edges =  self.generate(obj)
        for edge in edges:
            edge = edge[self.relation]
            subject = edge["e1"]
            # relation = edge["relation"]
            relation = "has a"
            predicate = edge["beams"]
            for pred in (predicate):
                graph_state.add_edge(subject, pred, rel=relation)
        
        return graph_state
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--model_file", type=str, default="comet/models/1e-05_adam_64_15500.pickle")
    parser.add_argument("--relation", type=str, default="CapableOf")
    parser.add_argument("--sampling_algorithm", type=str, default="beam-5")
    args = parser.parse_args()
    comet_helper = CometHelper(args)
    while True:
        input_event = "help"
        relation = "help"
        sampling_algorithm = args.sampling_algorithm

        while input_event is None or input_event.lower() == "help":
            input_event = input("Give an input entity (e.g., go on a hike -- works best if words are lemmatized): ")
        comet_helper.generate(input_event)
        

        

