from typing import Optional, Tuple, Union

import torch
from torch import nn

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput

class W2V2RobertaForCTC(PreTrainedModel):
    def __init__(self, encoder, decoder):
        super(W2V2RobertaForCTC, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.encoder_output_dim = getattr(encoder.config, "output_hidden_size", encoder.config.hidden_size)
        if (
            self.encoder_output_dim != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            # encoder outputs might need to be projected to different dimension for decoder
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        
    def forward(self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None):

        encoder_hidden_states = self.encoder(input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)

        if (
            self.encoder_output_dim != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        decoder_outputs = self.decoder(inputs_embeds=encoder_hidden_states,
                                       labels=labels,
                                       return_dict=return_dict)

        return decoder_outputs


