import torch

from transformers import BertTokenizer, BertModel


class CtxQueryGraphEncoder:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
    ):
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device)
        self._load_tokenizers()
        self._load_model()

    def _freeze_encoder(
        self,
    ):
        for params in self.model.params():
            params.requires_grad = False
        pass

    def _load_tokenizers(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def _load_model(self):
        self.model = BertModel.from_pretrained(self.model_name_or_path)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self._freeze_encoder()
        self.model.eval()

    def ecode_text(
        self,
        sentences: list | str,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings
