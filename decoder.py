import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from mapping import Mapper
from captioningDataset import CaptioningDataset
import math
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print('lora not available')


class OPT(nn.Module):
    def __init__(self, model_name, device, precision=torch.float16, prefix_length=10, add_noise=True, variance=0.016):
        super(OPT, self).__init__()
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=precision,
            device_map='auto',

        ).to(device)
        self.add_noise = add_noise
        self.variance = variance
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.hidden_size = self._get_hidden_size()
        self.prefix_length = prefix_length
        self.fp = precision
        self.mapper = Mapper(768, self.hidden_size, self.prefix_length).to(self.device, dtype=precision)

    def generate(self, prompt, stochastic=False, max_tokens=50, seed=32):
        if stochastic:
            set_seed(seed)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated_ids = self.model.generate(input_ids, do_sample=stochastic, max_new_tokens=max_tokens, )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

    def caption(self, embeddings, stochastic=False, max_tokens=50, seed=32):
        if stochastic:
            set_seed(seed)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        bos = self.model.get_input_embeddings()(torch.tensor([2]).to(self.device, dtype=torch.long)).unsqueeze(0)
        prefix = self.mapper(embeddings.to(dtype=self.fp, device=self.device)).view(-1, self.prefix_length, self.hidden_size)
        prefix = torch.concat([bos, prefix], dim=1)
        generated_ids = self.model.generate(do_sample=stochastic, max_new_tokens=max_tokens, inputs_embeds=prefix)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def forward(self, batch):
        embeddings = batch['embeddings'].to(self.device, dtype=self.fp)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        captions = batch['captions']
        if self.add_noise:
            embeddings = self.noise_injection(embeddings)

        prefix_tokens = self.mapper(embeddings).view(-1, self.prefix_length, self.hidden_size)
        captions_emb = self.get_input_embeds(captions).to(self.device, dtype=self.fp)
        # print(prefix_tokens.shape, embeddings.shape, captions_emb.shape)

        # [batch, bos + prefix + caption, d_model]
        input_emb = torch.concat([captions_emb[:, :1, :], prefix_tokens, captions_emb[:, 1:, :]], dim=1).to(self.fp)
        # print(input_emb.shape)
        labels = self.tokenizer(captions, return_tensors="pt", padding=True).input_ids.to(
            self.device, self.fp)

        # opt ignores -100 labels during loss computation
        # print(input_emb.shape)
        ignore = torch.ones(labels.shape[0], self.prefix_length + 1).to(self.device) * -100

        labels = torch.concat([ignore, labels[:, 1:]], dim=1)
        return self.model(inputs_embeds=input_emb, labels=labels.to(torch.long))

    def get_input_embeds(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(self.device).squeeze(0)
        embeddings = self.model.get_input_embeddings()
        return embeddings(input_ids)

    def _get_hidden_size(self):
        ids = self.tokenizer("prompt", return_tensors="pt").input_ids.to(self.device).squeeze(0)
        return self.model.get_input_embeddings()(ids).shape[1]

    def noise_injection(self, x, ):
        return x + torch.randn(x.shape, device=self.device, dtype=self.fp) * math.sqrt(self.variance)

    def lora_model(self, r, alpha, dropout):
        for param in self.model.parameters():
            param.requires_grad = False
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",

        )
        self.model = get_peft_model(self.model, config).to(self.fp)


class hugging_decoder(OPT):
    def forward(self, embeddings, captions):
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        if self.add_noise:
            embeddings = self.noise_injection(embeddings)
        prefix_tokens = self.mapper(embeddings).view(-1, self.prefix_length, self.hidden_size)
        captions_emb = self.get_input_embeds(captions).to(self.device, dtype=self.fp)
        input_emb = torch.concat([captions_emb[:, :1, :], prefix_tokens, captions_emb[:, 1:, :]], dim=1).to(self.fp)
        labels = self.tokenizer(captions, return_tensors="pt", padding=True).input_ids.to(
            self.device, self.fp)

        ignore = torch.ones(labels.shape[0], self.prefix_length + 1).to(self.device) * -100
        labels = torch.concat([ignore, labels[:, 1:]], dim=1)
        return self.model(inputs_embeds=input_emb, labels=labels.to(torch.long))

    def get_input_embeds(self, prompt):
        # already tokenized when using trainer API
        # input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(self.device).squeeze(0)
        embeddings = self.model.get_input_embeddings()
        return embeddings(prompt)


if '__main__' == __name__:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "")
    dataset = CaptioningDataset('embeddings/coco_openclip_val.pkl')
    loader = dataset.get_loader()
    decoder = OPT('facebook/opt-350m', device)
    decoder.lora_model(16, 32, 0.05)

    for batch in loader:
        decoder(batch)
        break
