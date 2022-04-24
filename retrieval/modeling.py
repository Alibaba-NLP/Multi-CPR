import os
import warnings

import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from transformers import  BertConfig, AutoModel, AutoModelForMaskedLM, AutoConfig, PretrainedConfig, \
    RobertaModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling, MaskedLMOutput
from transformers.models.roberta.modeling_roberta import RobertaLayer

from arguments import DataTrainingArguments, ModelArguments, TextMatchingForBertTrainingArguments
from transformers import TrainingArguments
import logging

logger = logging.getLogger(__name__)

class MyPooler(nn.Module):
    def __init__(self, pool_type="cls", in_size=768, out_size=768):
        super().__init__()
        self.dense = nn.Linear(in_size, out_size)
        self.activation = nn.Tanh()
        self.pool_type = pool_type
        #self.activation = nn.ReLU()

    def forward(self, hidden_states, attention_mask):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        if self.pool_type == "cls":
            rep_tensor = hidden_states[:, 0]
        elif self.pool_type == "max":
            x = hidden_states * attention_mask.unsqueeze(-1)
            rep_tensor = F.max_pool1d(x.transpose(1, 2).contiguous(), x.size(1)).squeeze(-1)
            #rep_tensor = torch.max((hidden_states * attention_mask.unsqueeze(-1)), axis=1).values
        elif self.pool_type == "avg":
            rep_tensor = ((hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        #pooled_output = self.dense(rep_tensor)
        #pooled_output = self.activation(pooled_output)
        pooled_output = rep_tensor
        return pooled_output

class Projection(nn.Module):
    def __init__(self, in_size=256, out_size=128):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(in_size, out_size)

    def forward(self, output):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        output = self.dropout(output)
        pooled_output = self.dense(output)
        return pooled_output


class TextMatchingForBert(BertPreTrainedModel):
    def __init__(self, config, pool_type="cls", add_pooling_layer=False, is_train=True, *model_args, **model_kargs):
        super().__init__(config)
        self.config = config
        self.is_train = is_train
        self.pooling_type = pool_type
        self.add_pooling_layer=add_pooling_layer

        self.bert = BertModel(config, add_pooling_layer)
        self.mypooler = MyPooler(self.pooling_type)

        self.init_weights()

    def pdists(self, a, squared = False, eps = 1e-8):
        prod = torch.mm(a, a.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res if squared else res.clamp(min = eps).sqrt()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]  # the last layer's output
        #if self.add_pooling_layer:
        #    pooled_output = outputs[1]
        #else:
        pooled_output = self.mypooler(sequence_output, attention_mask)
        loss = None
        if self.is_train:
            # using contrastive loss
            device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda") 
            batch_size = input_ids.size()[0] # -> 4
            
            pooled_output = pooled_output
            query_output = pooled_output[0::2,]
            doc_output= pooled_output[1::2,]
            
            query_output_norm = query_output / query_output.norm(dim=1)[:,None]
            doc_output_norm = doc_output / doc_output.norm(dim=1)[:,None]

            scores = torch.matmul(query_output_norm, doc_output_norm.transpose(0, 1))*20.0

            target = torch.arange(query_output_norm.size()[0], dtype=torch.long, device=device)

            loss = F.cross_entropy(scores, target) 
            output = (sequence_output, pooled_output) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        else:
            return pooled_output
