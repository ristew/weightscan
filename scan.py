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
        self.top_k = 5
        self.prompt = '''After losing one of my 4 gpus I have only'''

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

    def get_top_k_embeddings(self, final_state, top_k_indices):
        """
        Extract embeddings corresponding to top_k logits from the final hidden state.
        """
        # Gather embeddings for top_k_indices
        top_k_embeddings = final_state[:, top_k_indices]
        return top_k_embeddings

    def umap(self):
        umap_reducer = umap.UMAP(n_components=2, metric='cosine', min_dist=0.0, n_neighbors=30)
        embeddings = []
        output = self.forward()
        self.get_normed_states(output)

        # Compute logits and select top_k logits
        logits = self.logits()  # Make sure this returns logits for the last layer
        log_probs, top_k_indices = self.logprobs()  # Adjust to get indices of top logits

        # Get embeddings corresponding to top_k logits from the final hidden state
        final_state_embeddings = self.normed_states[-1]  # Final hidden state
        top_k_embeddings = self.get_top_k_embeddings(final_state_embeddings, top_k_indices.cpu().numpy())

        # Reshape for UMAP
        top_k_embeddings_np = top_k_embeddings.cpu().detach().numpy().reshape(-1, top_k_embeddings.size(-1))

        # Fit UMAP using embeddings of top_k logits
        umap_reducer.fit(top_k_embeddings_np)
        return [umap_reducer.transform(state.cpu().detach().numpy()) for state in self.normed_states]

    def logits(self):
        return self.model.lm_head(self.normed_states[-1]).float().unsqueeze(0)

    def logprobs(self):
        probs = F.softmax(self.logits(), dim=-1)
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
            top_probs, top_indices = self.logprobs()
            top_tokens = [(self.tokenizer.decode([idx]), top_probs[j].item()) for j, idx in enumerate(top_indices)]
            top_info = ", ".join([f"({repr(token)}, {prob:.2f})" for token, prob in top_tokens])
            text_str = f"prompt: {repr(self.prompt)}\ntop_k({self.top_k}): {top_info}"
            plt.text(global_x_min + 1, global_y_min + 1, text_str,
                     verticalalignment='bottom', horizontalalignment='left',
                     color='white', fontsize=8)
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            images.append(Image.open(buf))
            plt.close()

        imageio.mimsave('hidden_states.mp4', images, format='MP4', fps=2)

Scan().test()
