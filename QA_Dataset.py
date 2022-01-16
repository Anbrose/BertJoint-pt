import torch
from torch.utils.data import DataLoader, Dataset

class QA_Dataset(Dataset):
    def __init__(self, features):
        self.features = features


    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        return_feature = (
            feature.input_ids,
            feature.input_mask,
            feature.segment_ids,
            feature.start_position,
            feature.end_position,
            feature.answer_type
        )
        return torch.tensor(feature.input_ids), torch.tensor(feature.input_mask), torch.tensor(feature.segment_ids), \
               feature.start_position, feature.end_position, feature.answer_type

