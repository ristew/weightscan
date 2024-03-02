import datetime
import os
import math
import json
import imageio
from io import BytesIO
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from umap import UMAP
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import seaborn as sns


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, compressed_dim=(1024, 8)):
        super(TransformerAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.encoder_fc = nn.Linear(input_dim, compressed_dim[0] * compressed_dim[1])
        self.decoder_fc = nn.Linear(compressed_dim[0] * compressed_dim[1], input_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        x_pooled = self.global_pool(x.transpose(1, 2))
        x_pooled = x_pooled.squeeze(-1)
        encoded = self.encoder_fc(x_pooled)
        encoded = encoded.view(-1, self.compressed_dim[0], self.compressed_dim[1])
        decoded = self.decoder_fc(encoded.view(-1, self.compressed_dim[0] * self.compressed_dim[1]))
        decoded = decoded.view(-1, self.input_dim)
        return encoded, decoded

class Scan():
    def __init__(self):
        torch.set_default_device('cuda')
        torch.set_float32_matmul_precision('medium')
        # model_name = 'MistralAI/Mistral-7B-v0.1'
        # self.model_name = 'google/gemma-2b'
        self.model_name = 'microsoft/phi-2'
        # self.model_name = 'stabilityai/stablelm-3b-4e1t'
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=quantization_config, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.top_k = 50
        self.prompt = '''A B A A => A A\nB A B A B => B\nA A B B =>'''
        # self.prompt = 'jesus'

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
        return self.model.forward(input_ids, output_hidden_states=True)

    def get_normed_states(self, output):
        hidden_states = output.hidden_states
        print(hidden_states[0][0].shape)
        normed_states = [self.norm(hs) for hs in hidden_states[:-1]]
        # output layer is already normed
        final = hidden_states[-1]
        normed_states.append(final)
        self.normed_states = normed_states
        return normed_states

    def autoencode(self):
        autoencoder = TransformerAutoencoder(input_dim=self.normed_states[0][0][0].size()[0])
        num_epochs = 10
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0002)
        for i in range(num_epochs):
            for data in self.normed_states:
                print('data', data.shape)
                optimizer.zero_grad()
                encoded, decoded = autoencoder(data.float())
                loss = criterion(decoded, data.float())
                print('loss', loss)
                loss.backward(retain_graph=True)
                optimizer.step()

        res = [autoencoder(n.float())[0] for n in self.normed_states]
        print('autoencoded', res)
        return res

    def embed(self):
        basis = torch.stack(self.autoencode()).squeeze()
        print('fit basis', basis.shape)
        reducer = UMAP(n_components=3, metric='cosine', min_dist=0).fit(basis.cpu().detach().numpy().reshape(-1, basis.size(-1)))
        print('reducer fit, transforming...')
        return [reducer.transform(state.cpu().detach().numpy()) for state in basis]

    def logits(self, layer=-1):
        return self.model.lm_head(self.normed_states[layer].unsqueeze(0)).float()

    def logprobs(self, layer=-1):
        probs = F.softmax(self.logits(layer)[0], dim=-1)
        return torch.topk(probs[0, -1, :], self.top_k)

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
        top_probs, top_indices = self.logprobs(layer_id)
        print(top_indices.shape)
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


    def test(self):
        output = self.forward()
        self.get_normed_states(output)
        # embeddings = [p.squeeze().cpu().detach().numpy() for p in self.autoencode()]
        embeddings = self.embed()
        print('embeddings', embeddings)
        self.global_x_min = min(embedding[:, 0].min() for embedding in embeddings)
        self.global_x_max = max(embedding[:, 0].max() for embedding in embeddings)
        self.global_y_min = min(embedding[:, 1].min() for embedding in embeddings)
        self.global_y_max = max(embedding[:, 1].max() for embedding in embeddings)
        buffer = max(self.global_x_max - self.global_x_min, self.global_y_max - self.global_y_min) / 6
        self.global_x_min -= buffer
        self.global_x_max += buffer
        self.global_y_min -= buffer
        self.global_y_max += buffer
        images = []
        for i, embedding in enumerate(embeddings):
            print(f'plot embedding {i}')
            images.append(self.plot_embedding(embedding, i))

        imageio.mimsave('hidden_states.mp4', images, format='MP4', fps=2)
        points = json.dumps([(p * 16).tolist() for p in embeddings])

        print(points)
        with open('visualize_template.html', 'r') as template_file:
            template = template_file.read()
            modified = template.replace('$$POINTS$$', points)
            with open('visualize.html', 'w') as out_file:
                out_file.write(modified)
                print('wrote modified template')

Scan().test()
