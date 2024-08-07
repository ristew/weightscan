import datetime
import os
import math
import jax.random as random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from umap import UMAP
from trimap import TriMap
from umap.aligned_umap import AlignedUMAP
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from autoencoder import Autoencoder


class Scan():
    def __init__(self, prompt):
        torch.set_default_device('cuda')
        torch.set_float32_matmul_precision('medium')
        self.model_name = 'microsoft/Phi-3-mini-4k-instruct'
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=quantization_config, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.top_k = 5
        self.prompt = prompt

    def gemma(self):
        return self.model_name == 'google/gemma-2b'
    def norm(self, state):
        if self.model_name == 'microsoft/phi-2':
            return self.model.model.final_layernorm(state)
        else:
            return self.model.model.norm(state)

    def forward(self):
        enc = self.tokenizer(self.prompt, return_tensors='pt', return_attention_mask=False)
        input_ids = enc['input_ids'].to('cuda')
        output = self.model.forward(input_ids, output_hidden_states=True)
        self.get_normed_states(output)
        # toks = self.normed_states[0]
        # chunks = torch.chunk(toks, toks.size(1), dim=1)
        # for c in chunks:
        #     print('normed', nt.shape, self.top_tokens(c))
        self.embeddings = self.embed()

    def get_normed_states(self, output):
        hidden_states = output.hidden_states
        # print(hidden_states[0][0].shape)
        normed_states = [self.norm(hs) for hs in hidden_states[:-1]]
        # output layer is already normed
        final = hidden_states[-1]
        normed_states.append(final)
        self.normed_states = normed_states
        return normed_states

    def autoencode(self):
        self.autoencoder = Autoencoder(input_dim=self.normed_states[0][0][0].size()[0])
        self.autoencoder.train(self.normed_states)

        res = [self.autoencoder(n.float())[0] for n in self.normed_states]
        return res

    def embed(self):
        # what if we ran each of the intermediate layers through the final layer?
        basis = torch.stack(self.autoencode()).squeeze()
        print('fit basis', basis.shape)
        # reducer = UMAP(n_components=3, metric='cosine', min_dist=0, n_neighbors=50).fit(basis.cpu().detach().numpy().reshape(-1, basis.size(-1)))
        key = random.PRNGKey(42)
        reducer = TriMap(n_dims=3, distance='cosine').fit(key, basis[1].cpu().detach().numpy())
        print('reducer fit, transforming...')
        return [reducer.transform(key, state.cpu().detach().numpy()) for state in basis]

    def logits(self, state):
        return self.model.lm_head(state.unsqueeze(0)).float()

    def logprobs(self, state):
        logits = self.logits(state)
        probs = F.softmax(logits[0], dim=-1)
        return torch.topk(probs[0, -1, :], self.top_k)

    def top_tokens(self, state):
        top_probs, top_indices = self.logprobs(state)
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
        points = [(p / 10).tolist() for p in self.embeddings]
        tops = [self.top_tokens(self.normed_states[i]) for i in range(len(self.embeddings))]
        nn_indices = self.find_nearest_neighbors(points)

        # Prepare data including nearest neighbors
        data = json.dumps({
            'points': points,
            'tops': tops,
            'prompt': self.prompt,
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
    Scan('''<|user|>\nParis is to France as Berlin is to:<|end|>\n<|assistant|>Answer:''').test()
