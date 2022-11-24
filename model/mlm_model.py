from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.functional import gelu

from transformers import RobertaModel
from transformers.utils import logging
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Adapter
from transformers.modeling_outputs import MaskedLMOutput


logger = logging.get_logger(__name__)


class RobertaCTCLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.output_hidden_size, config.output_hidden_size)
        self.layer_norm = nn.LayerNorm(config.output_hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.output_hidden_size, config.decoder_vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.decoder_vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

class RobertaForCTCDecoding(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.adapter = Wav2Vec2Adapter(config)
        self.lm_head = RobertaCTCLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        down_proj = self.adapter(sequence_output)
        prediction_scores = self.lm_head(down_proj)

        ctc_loss = None
        if labels is not None:

            if labels.max() >= self.config.decoder_vocab_size:
                    raise ValueError(f"Label values must be <= vocab_size: {self.config.decoder_vocab_size}")
            
            input_shape = input_ids.shape if inputs_embeds is None else inputs_embeds.argmax(-1).shape

            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones(input_shape, dtype=torch.long)
            )

            input_lengths = attention_mask.sum(-1)

            for _ in range(self.config.num_adapter_layers):
                input_lengths = torch.div(input_lengths - 1, self.config.adapter_stride, rounding_mode="floor") + 1

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(prediction_scores, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((ctc_loss,) + output) if ctc_loss is not None else output

        return MaskedLMOutput(
            loss=ctc_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )