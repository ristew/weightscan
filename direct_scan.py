import datetime
import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from umap import UMAP
from umap.aligned_umap import AlignedUMAP
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from autoencoder import Autoencoder
import seaborn as sns
import imageio
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image

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
        self.embeddings = self.autoencode()

    def get_normed_states(self, output):
        hidden_states = output.hidden_states
        print('hidden state shape', hidden_states[0][0].shape)
        normed_states = [self.norm(hs) for hs in hidden_states[:-1]]
        # output layer is already normed
        final = hidden_states[-1]
        normed_states.append(final)
        self.normed_states = normed_states
        return normed_states

    def autoencode(self):
        self.autoencoder = Autoencoder(
            input_dim=self.normed_states[0][0][0].size()[0],
            compressed_dim=(1024, 3),
            temporal_weight=1e5,
            lr=0.002,
            num_epochs=8,
        )
        self.autoencoder.train(self.normed_states)

        res = [self.autoencoder(n.float())[0][0] for n in self.normed_states]
        return res

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
        points = [(p * 16).tolist() for p in self.embeddings]
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

    def plot_embedding(self, embedding, layer_id):
        plt.figure(figsize=(10, 10))
        ax = sns.kdeplot(x=embedding[:, 0], y=embedding[:, 1], cmap="mako", levels=50)
        plt.scatter(x=embedding[:, 0], y=embedding[:, 1], c="white", edgecolors="white")

        ax.set_facecolor('black')
        plt.gcf().set_facecolor('black')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.title.set_color('white')
        plt.axis([self.global_x_min, self.global_x_max, self.global_y_min, self.global_y_max])
        plt.title(f"layer {layer_id}")
        top_probs, top_indices = self.logprobs(self.normed_states[layer_id])
        top_tokens = [(self.tokenizer.decode([idx]), top_probs[j].item()) for j, idx in enumerate(top_indices)]
        top_info = ", ".join([f"({repr(token)}, {prob:.2f})" for token, prob in top_tokens[:5]])
        text_str = f"prompt: {repr(self.prompt)}\ntop_k({self.top_k}): {top_info}"
        plt.text(self.global_x_min + 1, self.global_y_min + 1, text_str,
                    verticalalignment='bottom', horizontalalignment='left',
                    color='white', fontsize=10)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    def plot(self):
        self.global_x_min = min(embedding.cpu().detach()[:, 0].min() for embedding in self.embeddings)
        self.global_x_max = max(embedding.cpu().detach()[:, 0].max() for embedding in self.embeddings)
        self.global_y_min = min(embedding.cpu().detach()[:, 1].min() for embedding in self.embeddings)
        self.global_y_max = max(embedding.cpu().detach()[:, 1].max() for embedding in self.embeddings)

        buffer = max(self.global_x_max - self.global_x_min, self.global_y_max - self.global_y_min) / 6
        self.global_x_min -= buffer
        self.global_x_max += buffer
        self.global_y_min -= buffer
        self.global_y_max += buffer
        images = []
        for i, embedding in enumerate(self.embeddings):
            embedding = embedding.cpu().detach()
            images.append(self.plot_embedding(embedding, i))

        imageio.mimsave('hidden_states.mp4', images, format='MP4', fps=2)

    def test(self):
        self.forward()
        self.visualize()
        self.plot()

if __name__ == '__main__':
    Scan('''<|user|>\nComplete the analogy: Paris is to France as Berlin is to:<|end|>\n<|assistant|>Answer:''').test()
