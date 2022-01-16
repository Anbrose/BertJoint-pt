from transformers import BertModel
import torch
import torch.nn as nn

class BertJointModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BertJointModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)


    def forward(self, input_ids, attn_masks, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attn_masks, token_type_ids=token_type_ids)
        #hidden_states = torch.cat(tuple([output.hidden_states[i] for i in [-1, -2, -3, -4]]))
        print(output.shape)
        return output