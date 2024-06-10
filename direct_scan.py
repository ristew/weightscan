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

class Sample():
    def __init__(self):
        pass

class Scan():
    def __init__(self, prompts):
        torch.set_default_device('cuda')
        torch.set_float32_matmul_precision('medium')
        self.model_name = 'openai-community/gpt2'
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.top_k = 5
        self.prompts = prompts
        self.states = []

    def norm(self, state):
        if self.model_name == 'microsoft/phi-2':
            return self.model.model.final_layernorm(state)
        elif self.model_name == 'openai-community/gpt2':
            return self.model.transformer.ln_f(state)
        else:
            return self.model.model.norm(state)

    def forward(self):
        for prompt in self.prompts:
            enc = self.tokenizer(prompt, return_tensors='pt', return_attention_mask=False)
            input_ids = enc['input_ids'].to('cuda')
            output = self.model.forward(input_ids, output_hidden_states=True)
            states = self.get_normed_states(output)
            self.states.append(states)

        self.autoencoder = Autoencoder(
            input_dim=self.states[0][0][0][0].size()[0],
            compressed_dim=(2048, 3),
            temporal_weight=1e4,
            lr=1e-3,
            num_epochs=5,
            training_set=self.states
        )
        self.embeddings = self.autoencode()

    def get_normed_states(self, output):
        hidden_states = output.hidden_states
        print('hidden state shape', hidden_states[0][0].shape)
        normed_states = [self.norm(hs) for hs in hidden_states[:-1]]
        # output layer is already normed
        final = hidden_states[-1]
        normed_states.append(final)
        return normed_states

    def autoencode(self):
        self.autoencoder.train()

        res = [self.autoencoder(n.float())[0][0] for n in self.states[0]]
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
        tops = [self.top_tokens(self.states[0][i]) for i in range(len(self.embeddings))]
        nn_indices = self.find_nearest_neighbors(points)

        # Prepare data including nearest neighbors
        data = json.dumps({
            'points': points,
            'tops': tops,
            'prompt': self.prompts[0],
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
        top_probs, top_indices = self.logprobs(self.states[0][layer_id])
        top_tokens = [(self.tokenizer.decode([idx]), top_probs[j].item()) for j, idx in enumerate(top_indices)]
        top_info = ", ".join([f"({repr(token)}, {prob:.2f})" for token, prob in top_tokens[:5]])
        text_str = f"prompt: {repr(self.prompts[0])}\ntop_k({self.top_k}): {top_info}"
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
        # self.plot()

if __name__ == '__main__':
    Scan([
        'When I go fishing, I like to think about',
        '22 * 8 =',
        'It is going to be okay',
        'How are you today?',
    ]).test()
