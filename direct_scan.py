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

        self.model_name = 'openai-community/gpt2'
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
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
            input_ids = enc['input_ids'].to(self.device)
            output = self.model.forward(input_ids, output_hidden_states=True)
            states = self.get_normed_states(output)
            self.states.append(states)

        self.autoencoder = Autoencoder(
            input_dim=self.states[0][0][0][0].size()[0],
            compressed_dim=(128, 3),
            lr=4e-5,
            weight_decay=0.001,
            num_epochs=3,
            layer_weight=1e1,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.autoencoder.lr, weight_decay=self.autoencoder.weight_decay)

        self.train_autoencoder()
        self.embeddings = self.autoencode()

    def get_normed_states(self, output):
        hidden_states = output.hidden_states
        normed_states = [self.norm(hs) for hs in hidden_states[:-1]]
        final = hidden_states[-1]
        normed_states.append(final)
        return normed_states

    def train_autoencoder(self):
        for epoch in range(self.autoencoder.num_epochs):
            for sample in self.states:
                for layer in sample:
                    encoded, loss = self.autoencoder.train_layer(layer, self.optimizer)
                    print(f'Epoch {epoch}, Layer Loss: {loss.item():6.3g}')

    def autoencode(self):
        encoded_results = []
        with torch.no_grad():
            for layer_states in self.states[0]:  # Iterate over layers in the first state
                encoded, _ = self.autoencoder(layer_states)
                encoded_results.append(encoded.squeeze(0))

        return encoded_results

    def logits(self, state):
        return self.model.lm_head(state.unsqueeze(0)).float()

    def logprobs(self, state):
        logits = self.logits(state)
        return F.softmax(logits[0], dim=-1)

    def top_tokens(self, state):
        top_probs, top_indices = torch.topk(self.logprobs(state)[0, -1, :], self.top_k)
        return [(self.tokenizer.decode([idx]), top_probs[j].item()) for j, idx in enumerate(top_indices)]

    def find_nearest_neighbors(self, embeddings, n_neighbors=4):
        nn = []
        for layer in embeddings:
            nn_layer = []
            for token in layer:
                neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
                neighbors.fit(token)
                distances, indices = neighbors.kneighbors(token)
                nn_layer.append([list(zip(indices[i].tolist(), distances[i].tolist()))[1:] for i in range(len(distances))])
            nn.append(nn_layer)
        return nn

    def visualize(self):
        points = [(p).tolist() for p in self.embeddings]
        tops = [self.top_tokens(self.states[0][i]) for i in range(len(self.embeddings))]
        nn_indices = self.find_nearest_neighbors(points)

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

    def test(self):
        self.forward()
        self.visualize()

if __name__ == '__main__':
    Scan([
        'The sun rises in the east and sets in the',
        'North, south, east, and west',
        'South to north, west to east',
        'From the northwest to the southeast',
        'Traveling eastward towards the rising sun',
        'The compass points due north',
        'Westward expansion across the continent',
        'Southern winds bring warm air',
        'Northeast to southwest diagonal',
        'The ship sailed south by southeast',
        'Turn left to head west, right to go east',
        'Northern lights dance in the arctic sky',
        'Easterly breeze off the ocean',
        'Southwest desert meets northwest forest',
        'From sea to shining sea, east to west',
        'North Star guides travelers at night',
        'Southward migration of birds in winter',
        'West coast to east coast road trip',
        'The Great Wall stretches east to west',
        'Northernmost point of the continent',
        'Southeast Asian tropical climate',
        'Western frontier of the old world',
        'Due south to reach the equator',
        'East meets West at the international date line'
    ]).test()
