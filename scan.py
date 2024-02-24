import datetime
import os

import imageio
from io import BytesIO
import numpy as np
import torch
import torch.nn.functional as F
import umap
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import seaborn as sns

class Scan():
    def __init__(self):
        torch.set_default_device('cuda')
        self.model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2', torch_dtype='auto', trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', trust_remote_code=True)
        self.top_k = 50
        self.prompt = '''A A B A => 3\nA B B B A => 2\nA => 1\nA A B A B => 3\nA A =>'''

    def forward(self):
        enc = self.tokenizer(self.prompt, return_tensors='pt', return_attention_mask=False)
        input_ids = enc['input_ids'].to('cuda')
        return self.model.forward(input_ids, output_hidden_states=True)

    def get_normed_states(self, output):
        hidden_states = output.hidden_states
        normed_states = [self.model.model.final_layernorm(hs.squeeze(0)) for hs in hidden_states[:-1]]
        # output layer is already normed
        normed_states.append(hidden_states[-1].squeeze(0))
        self.normed_states = normed_states
        return normed_states

    def umap(self):
        umap_reducer = umap.UMAP(n_components=2, metric='cosine', min_dist=0.0, n_neighbors=5)
        embeddings = []
        output = self.forward()
        self.get_normed_states(output)
        final = self.normed_states[-1]
        basis = torch.stack([(0.1 * state + 0.9 * final) for state in self.normed_states[:-1]])
        print(final)
        umap_reducer.fit(basis.cpu().detach().numpy().reshape(-1, basis.size(-1)))
        return [umap_reducer.transform(state.cpu().detach().numpy()) for state in self.normed_states]

    def logits(self, layer=-1):
        return self.model.lm_head(self.normed_states[layer]).float()

    def logprobs(self, layer=-1):
        probs = F.softmax(self.logits(layer).unsqueeze(0), dim=-1)
        return torch.topk(probs[0, -1, :], self.top_k)


    def test(self):
        embeddings = self.umap()
        print(embeddings)
        global_x_min = min(embedding[:, 0].min() for embedding in embeddings)
        global_x_max = max(embedding[:, 0].max() for embedding in embeddings)
        global_y_min = min(embedding[:, 1].min() for embedding in embeddings)
        global_y_max = max(embedding[:, 1].max() for embedding in embeddings)

        buffer = 7
        global_x_min -= buffer
        global_x_max += buffer
        global_y_min -= buffer
        global_y_max += buffer
        images = []
        for i, embedding in enumerate(embeddings):
            plt.figure(figsize=(10, 10))
            ax = sns.kdeplot(x=embedding[:, 0], y=embedding[:, 1], cmap="mako", levels=50)
            ax.set_facecolor('black')
            plt.gcf().set_facecolor('black')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.title.set_color('white')
            plt.axis([global_x_min, global_x_max, global_y_min, global_y_max])
            plt.title(f"layer {i}")
            top_probs, top_indices = self.logprobs(i)
            top_tokens = [(self.tokenizer.decode([idx]), top_probs[j].item()) for j, idx in enumerate(top_indices)]
            top_info = ", ".join([f"({repr(token)}, {prob:.2f})" for token, prob in top_tokens[:5]])
            text_str = f"prompt: {repr(self.prompt)}\ntop_k({self.top_k}): {top_info}"
            plt.text(global_x_min + 1, global_y_min + 1, text_str,
                     verticalalignment='bottom', horizontalalignment='left',
                     color='white', fontsize=10)
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            images.append(Image.open(buf))
            plt.close()

        imageio.mimsave('hidden_states.mp4', images, format='MP4', fps=2)

Scan().test()
