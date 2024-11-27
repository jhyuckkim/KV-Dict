import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Buffer():
    """
    Buffer for storing KV vectors for training the autoencoder.
    """
    def __init__(self, cfg, model, tokenizer, texts):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.texts = texts
        self.device = cfg["device"]
        self.buffer = torch.zeros((cfg["batch_size"] * cfg["buffer_mult"], cfg["num_hidden_layers"] * 2, cfg["head_dim"]), device=self.device)
        self.text_pointer = 0

        random.shuffle(self.texts)
        self.refresh()
    
    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        while self.pointer < self.buffer.shape[0]:
            try:
                texts = self.texts[self.text_pointer:self.text_pointer+self.cfg["lm_batch_size"]]
                encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                input_ids = encoded_inputs["input_ids"].to(self.device)
                attention_mask = encoded_inputs["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                past_key_values = outputs.past_key_values

                kvs = []
                for l in range(self.cfg["num_hidden_layers"]):
                    keys, values = past_key_values[l]
                    kvs.append(keys)
                    kvs.append(values)
                kvs = torch.stack(kvs).permute(1, 3, 2, 0, 4).reshape(-1, self.cfg["num_hidden_layers"] * 2, self.cfg["head_dim"])
                mask = attention_mask.view(-1, 1).repeat(1, self.cfg["num_key_value_heads"]).view(-1)
                kvs = kvs[mask.bool()]

                buffer_slice_size = min(self.buffer.shape[0] - self.pointer, kvs.size(0))
                self.buffer[self.pointer:self.pointer + buffer_slice_size, :, :] = kvs[:buffer_slice_size]
                self.pointer += buffer_slice_size
                self.text_pointer += self.cfg["lm_batch_size"]

                if self.text_pointer > len(self.texts) - self.cfg["lm_batch_size"]:
                    self.text_pointer = 0

                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"Error encountered: {e}. Skipping this batch.")
                self.text_pointer += self.cfg["lm_batch_size"]

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(self.device)]

    def next(self):
        out = self.buffer[self.pointer:self.pointer + self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0] - self.cfg["batch_size"]:
            self.refresh()
        return out