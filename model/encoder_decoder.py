from typing import Optional

import torch
from torch import nn

from transformers import AutoModelForCTC, SpeechEncoderDecoderConfig, Wav2Vec2Processor
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from .mlm_model import RobertaForCTCDecoding

class CustomMLMProcessor(Wav2Vec2Processor):
    
    feature_extractor_class = "Wav2Vec2FeatureExtractor"
    tokenizer_class = "RobertaTokenizer"
    
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

class W2V2RobertaForCTC(PreTrainedModel):

    config_class = SpeechEncoderDecoderConfig
    base_model_prefix = "speech_encoder_decoder"

    def __init__(self, config, encoder=None, decoder=None):
        super().__init__(config)

        if encoder is None:
            encoder = AutoModelForCTC.from_config(config.encoder)

        if decoder is None:
            decoder = RobertaForCTCDecoding.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        
    def forward(self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None):

        encoder_outputs = self.encoder(input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)


        if isinstance(encoder_outputs, tuple):
            if labels is not None:
                encoder_outputs = encoder_outputs[0]
            else:
                encoder_outputs = encoder_outputs[1]
        else:
            encoder_outputs = encoder_outputs.logits

        decoder_outputs = self.decoder(inputs_embeds=encoder_outputs,                                       
                                       labels=labels,
                                       return_dict=return_dict)

        return decoder_outputs
        
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder of the speech encoder so
        that its parameters will not be updated during training.
        """
        self.encoder.freeze_feature_encoder()

