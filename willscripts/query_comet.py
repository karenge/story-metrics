import os
import sys
import torch

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive


class COMET:

    def __init__(self):
        self.model_file = "model/pretrained/conceptnet_pretrained_model.pickle"
        self.sampling_algorithm = "beam-3"

        opt, state_dict = interactive.load_model_file(self.model_file)
        self.data_loader, self.text_encoder = interactive.load_data("conceptnet", opt)
        n_ctx = self.data_loader.max_e1 + self.data_loader.max_e2 + self.data_loader.max_r
        n_vocab = len(self.text_encoder.encoder) + n_ctx
        self.model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)
        cfg.device = "cpu"
        self.sampler = interactive.set_sampler(opt, self.sampling_algorithm, self.data_loader)


    def has_property(self, input_event):
        outputs = interactive.rel_get_sequence(
            input_event, self.model, self.sampler, self.data_loader, self.text_encoder, "HasProperty")
        return outputs

    def expect(self, input_event, rel):
        outputs = interactive.rel_get_sequence(
            input_event, self.model, self.sampler, self.data_loader, self.text_encoder, rel)
        return outputs
