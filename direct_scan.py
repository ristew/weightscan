import datetime
import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from autoencoder import Autoencoder

class Scan():
    def __init__(self, prompts):
        # Determine if CUDA is available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        torch.set_default_tensor_type(torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor)
        torch.set_float32_matmul_precision('medium')

        # self.model_name = 'openai-community/gpt2-medium'
        self.model_name = 'HuggingFaceTB/SmolLM-135M'
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, quantization_config=quantization_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.top_k = 5
        self.prompts = prompts
        self.states = []

    def norm(self, state):
        if self.model_name == 'microsoft/phi-2':
            return self.model.model.final_layernorm(state)
        elif self.model_name.startswith('openai-community/gpt2'):
            return self.model.transformer.ln_f(state)
        elif self.model_name == 'EleutherAI/pythia-410m':
            return self.model.gpt_neox.final_layer_norm(state)
        else:
            return self.model.model.norm(state)

    def decode_tok(self, idx):
        return self.tokenizer.decode([idx.item()])

    def forward(self):
        for prompt in self.prompts:
            enc = self.tokenizer(prompt, return_tensors='pt', return_attention_mask=False)
            input_ids = enc['input_ids'].to(self.device)
            tok_len = input_ids.shape[1]
            print(f'prompt {prompt}\ntok_len {tok_len}')
            for i in range(1, tok_len + 1):
                toks = input_ids[:, :i]
                output = self.model.forward(toks, output_hidden_states=True)
                top_prob, top_index = torch.topk(F.softmax(output.logits[0, -1, :], dim=-1), 1)
                print(self.decode_tok(toks[0, -1]), self.decode_tok(top_index), top_prob.item())
                states = self.get_normed_states(output)
                self.states.append(states)

        self.autoencoder = Autoencoder(
            input_dim=self.states[0][0][0][0].size()[0],
            compressed_dim=(4096, 3),
            lr=8e-4,
            weight_decay=0.001,
            num_epochs=2,
        ).to(self.device)
        self.embeddings = self.autoencode()

    def get_normed_states(self, output):
        hidden_states = output.hidden_states
        print('hidden state shape', hidden_states[0][0].shape)
        normed_states = [self.norm(hs).detach() for hs in hidden_states[:-1]]
        final = hidden_states[-1]
        normed_states.append(final)
        return normed_states

    def autoencode(self):
        self.autoencoder.train_set(self.states)
        res = [self.autoencoder(n.float().to(self.device))[0][0] for n in self.states[-1]]
        return res

    def logits(self, state):
        state = state.unsqueeze(0)
        if self.model_name == 'EleutherAI/pythia-410m':
            return self.model.embed_out(state).float()
        else:
            return self.model.lm_head(state).float()

    def logprobs(self, state):
        logits = self.logits(state)
        return F.softmax(logits[0], dim=-1)

    def top_tokens(self, state):
        top_probs, top_indices = torch.topk(self.logprobs(state)[0, -1, :], self.top_k)
        return [(self.tokenizer.decode([idx]), top_probs[j].item()) for j, idx in enumerate(top_indices)]

    def find_nearest_neighbors(self, embeddings, n_neighbors=4):
        nn = []
        for layer in embeddings:
            neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            neighbors.fit(layer)
            distances, indices = neighbors.kneighbors(layer)
            nn.append([list(zip(indices[i].tolist(), distances[i].tolist()))[1:] for i in range(len(distances))])
        return nn

    def visualize(self):
        points = [(p * 64).tolist() for p in self.embeddings]
        tops = [self.top_tokens(self.states[-1][i]) for i in range(len(self.embeddings))]
        nn_indices = self.find_nearest_neighbors(points)

        data = json.dumps({
            'points': points,
            'tops': tops,
            'prompt': self.prompts[-1],
            'neighbors': nn_indices
        })
        with open('visualize_template.html', 'r') as template_file:
            template = template_file.read()
            modified = template.replace('$$POINTS$$', data)
            with open('visualize.html', 'w') as out_file:
                out_file.write(modified)
                print('wrote modified template')

    def test(self):
        self.forward()
        self.visualize()

if __name__ == '__main__':
    Scan([
'Standing atop the hyperbridge at the end of time, ready to let loose, 12 45 72 END',
'Welcome to the new kind of bread show, big fish',
'Bagel crab bagel crab bagel',
'''Bagel/Tuna=>Bagel
Shrimp/Baguette=>Baguette
Lobster/Ciabatta=>Ciabatta
Sourdough/Crab=>Sourdough
Focaccia/Cod=>Focaccia
Rye/Anchovies=>Rye
Mussels/Brioche=>Brioche
Salmon/Croissant=>''',
    ]).test()
