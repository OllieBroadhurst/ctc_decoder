from typing import Optional

import torch

from transformers import AutoModelForCTC, SpeechEncoderDecoderConfig, Wav2Vec2Processor
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import _HIDDEN_STATES_START_POSITION

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
                return_dict=False,
                labels=labels)

        loss = None

        if labels is not None:
            encoder_loss = encoder_outputs[0]
            decoder_inputs = encoder_outputs[1]
        else:
            decoder_inputs = encoder_outputs[0]

        decoder_outputs = self.decoder(inputs_embeds=decoder_inputs,                                       
                                       labels=labels,
                                       return_dict=False)

        if labels is not None:
            loss = decoder_outputs[0] * 0.8 + encoder_loss * 0.2
            decoder_output = decoder_outputs[1:]

        if not return_dict:
            return ((loss,) + decoder_output) if loss is not None else decoder_output

        return MaskedLMOutput(
            loss=loss,
            logits=decoder_output[1],
            hidden_states=None,
            attentions=None,
        )

        
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder of the speech encoder so
        that its parameters will not be updated during training.
        """
        self.encoder.freeze_feature_encoder()

