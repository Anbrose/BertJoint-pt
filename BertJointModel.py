from transformers import BertModel
import torch
import torch.nn as nn

class BertJointModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BertJointModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.position_classifier = nn.Sequential(
            nn.Dropout(p=0.05),
            nn.Linear(768, 2)
        )
        self.answerType_classifier = nn.Sequential(
            nn.Dropout(p=0.05),
            nn.Linear(768, 5)
        )


    def forward(self, input_ids, attn_masks, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attn_masks, token_type_ids=token_type_ids)

        # (last_hidden_state, pooler_output, hidden_states)
        last_hidden_state, pooler_output, hidden_states =  (outputs[0], outputs[1], outputs[2])
        # embedding_output = hidden_states[0]
        # attention_hidden_states = hidden_states[1:]
        position_logits = self.position_classifier(last_hidden_state)
        #print(position_logits.shape)

        answerType_logits = self.answerType_classifier(pooler_output)
        #print(answerType_logits)

        # last hidden_shape:[batch_size,512,768] => [batch_size, seq_len, hidden_dim], batch_size不足default不会填满
        #hidden_states = torch.cat(tuple([output.hidden_states[i] for i in [-1, -2, -3, -4]]))
        return position_logits[:,:,0], position_logits[:,:,1], answerType_logits