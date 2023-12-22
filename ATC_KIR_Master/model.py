import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from typing import Optional, Tuple
import torch.nn.functional as F
from ernie import ErnieModel, ErniePreTrainedModel


@dataclass
class UIEModelOutput(ModelOutput):
    start_prob: torch.FloatTensor = None
    end_prob: torch.FloatTensor = None
    discriminator_prob: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalAttention(nn.Module):
    def __init__(self, hidden_size, local_context_size=2):
        super(LocalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.local_context_size = local_context_size  # 设置局部上下文大小
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, sequence_output):
        seq_len = sequence_output.size(1)
        new_outputs = []

        for i in range(seq_len):
            # 根据步长获取局部区域
            start_idx = max(0, i - self.local_context_size)
            end_idx = min(seq_len, i + self.local_context_size + 1)
            local_region = sequence_output[:, start_idx:end_idx, :]

            Q = self.query(local_region)
            K = self.key(local_region)
            V = self.value(local_region)

            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size ** 0.5)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, V)
            new_output = attn_output[:, i - start_idx, :]
            new_outputs.append(new_output)

        new_sequence_output = torch.stack(new_outputs, dim=1)
        return new_sequence_output

class MultiHeadLocalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=2):
        super(MultiHeadLocalAttention, self).__init__()
        # 为每个头指定不同的局部上下文大小
        context_sizes = [1,2]
        self.heads = nn.ModuleList([LocalAttention(hidden_size, local_context_size=context_sizes[i % num_heads]) for i in range(num_heads)])

    def forward(self, sequence_output):
        head_outputs = [head(sequence_output) for head in self.heads]
        # 对所有头的输出进行平均
        combined_output = sum(head_outputs) / len(self.heads)
        return combined_output



class UIE(ErniePreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super(UIE, self).__init__(config)
        self.encoder = ErnieModel(config)
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.linear_start = nn.Linear(self.hidden_size, 1)
        self.linear_end = nn.Linear(self.hidden_size, 1)
        self.discriminator = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        # # 添加注意力融合层
        self.local_attention = MultiHeadLocalAttention(self.hidden_size)
        self.adaptive_fusion = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.post_init()
    def forward(self, input_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_output= outputs[0]

        # 使用注意力融合层
        fused_sequence_output = self.local_attention(sequence_output)

        concatenated_output = torch.cat((sequence_output, fused_sequence_output), dim=-1)
        concatenated_output=self.adaptive_fusion(concatenated_output)


        start_logits = self.linear_start(concatenated_output)
        start_logits = torch.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)


        end_logits = self.linear_end(concatenated_output)
        end_logits = torch.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)

        # # #新增判别任务部分
        content_embedding =sequence_output[:, 0, :]#取出CLS向量
        discriminator_logits = self.discriminator(content_embedding)
        discriminator_prob = self.sigmoid(discriminator_logits)
        discriminator_prob = torch.squeeze(discriminator_prob, -1)

        return UIEModelOutput(
            start_prob=start_prob,
            end_prob=end_prob,
            discriminator_prob=discriminator_prob,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )