import torch
import torch.nn as nn
from transformers import PreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel


class modified_RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, from_hs, dropout_prob=0.1, num_labels=2):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class tsRobertaModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.dropout = nn.Dropout(0.1)
        self.tot_linear = nn.Linear(config.hidden_size, 100)
        self.ts_linear = nn.Linear(config.hidden_size + 100, config.hidden_size + 100)

        self.tot_classification = nn.Linear(100, 2)
        self.ts_classification = nn.Linear(config.hidden_size + 100, 2) # concat then to projection to 2.

        self.init_weights()

    def forward(self, input_ids, attention_mask, labels, tot_labels, return_dict):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        original_x = outputs[0][:, 0, :]

        # You write you new head here
        x = self.dropout(original_x)
        tot_hidden = self.tot_linear(x)
        tot_hidden = torch.tanh(tot_hidden)
        tot_hidden = self.dropout(tot_hidden)
        tot_logits = self.tot_classification(tot_hidden)

        y = self.dropout(original_x)
        ts_hidden = self.ts_linear(torch.cat((y, tot_hidden), dim=-1))
        ts_hidden = torch.tanh(ts_hidden)
        ts_hidden = self.dropout(ts_hidden)
        ts_logits = self.ts_classification(ts_hidden)


        loss = nn.CrossEntropyLoss()

        loss_ts = loss(ts_logits, labels)
        loss_tot = loss(tot_logits, tot_labels)

        if not return_dict:
            output = (ts_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=(loss_ts, loss_tot),
            logits=ts_logits,
        )
