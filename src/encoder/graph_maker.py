import torch


class GraphMaker:
    def __init__(
        self,
        paragraph_encodings,
        question: any,
    ):
        node_features = torch.stack(paragraph_encodings)
        num_paragraphs = len(paragraph_encodings)
        pass
