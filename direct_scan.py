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

    def forward_toks(self, toks):
        output = self.model.forward(toks, output_hidden_states=True)
        top_prob, top_index = torch.topk(F.softmax(output.logits[0, -1, :], dim=-1), 1)
        print(self.decode_tok(toks[0, -1]), self.decode_tok(top_index), top_prob.item())
        states = self.get_normed_states(output)
        self.states.append(states)

    def forward(self):
        for prompt in self.prompts:
            enc = self.tokenizer(prompt, return_tensors='pt', return_attention_mask=False)
            input_ids = enc['input_ids'].to(self.device)
            tok_len = input_ids.shape[1]
            print(f'prompt {prompt}\ntok_len {tok_len}')
            for i in range(1, tok_len + 1):
                self.forward_toks(input_ids[:, i-1:i])
                self.forward_toks(input_ids[:, :i])

        self.autoencoder = Autoencoder(
            input_dim=self.states[0][0][0][0].size()[0],
            compressed_dim=(4096, 3),
            lr=2e-4,
            weight_decay=0.001,
            num_epochs=1,
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
    "In my younger and more vulnerable years my father gave me some advice that I’ve been turning over in my mind ever since.",
    "“Whenever you feel like criticizing anyone,” he told me, “just remember that all the people in this world haven’t had the advantages that you’ve had.”",
    "He didn’t say any more, but we’ve always been unusually communicative in a reserved way, and I understood that he meant a great deal more than that.",
    "In consequence, I’m inclined to reserve all judgements, a habit that has opened up many curious natures to me and also made me the victim of not a few veteran bores.",
    "The abnormal mind is quick to detect and attach itself to this quality when it appears in a normal person, and so it came about that in college I was unjustly accused of being a politician, because I was privy to the secret griefs of wild, unknown men.",
    "Most of the confidences were unsought—frequently I have feigned sleep, preoccupation, or a hostile levity when I realized by some unmistakable sign that an intimate revelation was quivering on the horizon;",
    "for the intimate revelations of young men, or at least the terms in which they express them, are usually plagiaristic and marred by obvious suppressions.",
    "Reserving judgements is a matter of infinite hope.",
    "I am still a little afraid of missing something if I forget that, as my father snobbishly suggested, and I snobbishly repeat, a sense of the fundamental decencies is parcelled out unequally at birth.",
    "And, after boasting this way of my tolerance, I come to the admission that it has a limit.",
    "Conduct may be founded on the hard rock or the wet marshes, but after a certain point I don’t care what it’s founded on.",
    "When I came back from the East last autumn I felt that I wanted the world to be in uniform and at a sort of moral attention forever;",
    "I wanted no more riotous excursions with privileged glimpses into the human heart.",
    "Only Gatsby, the man who gives his name to this book, was exempt from my reaction—Gatsby, who represented everything for which I have an unaffected scorn.",
    "If personality is an unbroken series of successful gestures, then there was something gorgeous about him, some heightened sensitivity to the promises of life, as if he were related to one of those intricate machines that register earthquakes ten thousand miles away.",
    "This responsiveness had nothing to do with that flabby impressionability which is dignified under the name of the “creative temperament”—it was an extraordinary gift for hope, a romantic readiness such as I have never found in any other person and which it is not likely I shall ever find again.",
    "No—Gatsby turned out all right at the end; it is what preyed on Gatsby, what foul dust floated in the wake of his dreams that temporarily closed out my interest in the abortive sorrows and short-winded elations of men."
]
).test()
