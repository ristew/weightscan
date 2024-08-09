import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from autoencoder import Autoencoder

class Trainer:
    def __init__(self, prompts, model_name='HuggingFaceTB/SmolLM-135M'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        torch.set_default_tensor_type(torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor)
        torch.set_float32_matmul_precision('medium')

        self.model_name = model_name
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, quantization_config=quantization_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
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

    def get_normed_states(self, output):
        hidden_states = output.hidden_states
        print('hidden state shape', hidden_states[0][0].shape)
        normed_states = [self.norm(hs).detach() for hs in hidden_states[:-1]]
        final = hidden_states[-1]
        normed_states.append(final)
        return normed_states

    def train(self):
        self.forward()
        input_dim = self.states[0][0][0][0].size()[0]
        self.autoencoder = Autoencoder(
            input_dim=input_dim,
            compressed_dim=(4096, 3),
            lr=2e-4,
            weight_decay=0.001,
            num_epochs=1,
        ).to(self.device)
        self.autoencoder.train_set(self.states)
        self.save_weights()

    def save_weights(self, filename='autoencoder_weights.pth'):
        torch.save(self.autoencoder.state_dict(), filename)
        print(f"Weights saved to {filename}")

if __name__ == '__main__':
    prompts = [
        "In my younger and more vulnerable years my father gave me some advice that I've been turning over in my mind ever since.",
        # ... (include all the prompts here)
    ]
    trainer = Trainer(prompts)
    trainer.train()