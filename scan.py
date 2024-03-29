import datetime
import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from umap import UMAP
from umap.aligned_umap import AlignedUMAP
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, compressed_dim=(768, 16)):
        super(TransformerAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.hidden_dim = 512
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.encoder_input = nn.Linear(input_dim, self.hidden_dim)
        self.encoder_output = nn.Linear(self.hidden_dim, compressed_dim[0] * compressed_dim[1])
        self.decoder_input = nn.Linear(compressed_dim[0] * compressed_dim[1], self.hidden_dim)
        self.decoder_output = nn.Linear(self.hidden_dim, input_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        x_pooled = self.global_pool(x.transpose(1, 2))
        x_pooled = x_pooled.squeeze(-1)
        a = self.encoder_input(x_pooled)
        a = torch.relu(a)
        a = self.encoder_output(a)
        encoded = a.view(-1, self.compressed_dim[0], self.compressed_dim[1])
        b = self.decoder_input(a)
        b = torch.relu(b)
        b = self.decoder_output(b)
        decoded = b.view(-1, self.input_dim)
        return encoded, decoded

class Scan():
    def __init__(self, prompt):
        torch.set_default_device('cuda')
        torch.set_float32_matmul_precision('medium')
        self.model_name = 'stabilityai/stablelm-2-1_6b'
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
        self.embeddings = self.embed()

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
        num_epochs = 4
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)
        for i in range(num_epochs):
            for data in self.normed_states:
                optimizer.zero_grad()
                encoded, decoded = autoencoder(data.float())
                loss = criterion(decoded, data.float())
                loss.backward(retain_graph=True)
                optimizer.step()
                print('loss', loss)

        res = [autoencoder(n.float())[0] for n in self.normed_states]
        print('autoencoded', res)
        return res

    def embed(self):
        # what if we ran each of the intermediate layers through the final layer?
        basis = torch.stack(self.autoencode()).squeeze()
        print('fit basis', basis.shape)
        reducer = UMAP(n_components=3, metric='cosine', min_dist=0).fit(basis.cpu().detach().numpy().reshape(-1, basis.size(-1)))
        print('reducer fit, transforming...')
        return [reducer.transform(state.cpu().detach().numpy()) for state in basis]

    def logits(self, layer=-1):
        return self.model.lm_head(self.normed_states[layer].unsqueeze(0)).float()

    def logprobs(self, layer=-1):
        logits = self.logits(layer)
        probs = F.softmax(logits[0], dim=-1)
        print('logprobs', logits.shape, probs.shape)
        return torch.topk(probs[0, -1, :], self.top_k)

    def top_tokens(self, layer_id):
        top_probs, top_indices = self.logprobs(layer_id)
        print('top_tokens', top_probs.shape, top_indices.shape)
        return [(self.tokenizer.decode([idx]), top_probs[j].item()) for j, idx in enumerate(top_indices)]

    def visualize(self):
        points = [(p * 12).tolist() for p in self.embeddings]
        tops = [self.top_tokens(i) for i in range(len(self.embeddings))]
        data = json.dumps({'points': points, 'tops': tops})
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
    Scan('''Pattern matching

input:
0, 0, 0
0, 1, 0
0, 0, 0
output:
1, 0, 1
0, 0, 0
1, 0, 1
input:
0, 0, 0
0, 9, 0
0, 0, 0
output:
3, 0, 3
0, 0, 0
3, 0, 3
input:
0, 0, 0
0, 4, 0
0, 0, 0
output:
''').test()
