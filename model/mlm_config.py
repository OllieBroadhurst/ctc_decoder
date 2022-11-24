from transformers import BertConfig

class CTCDecoderConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a [`RobertaModel`] or a [`TFRobertaModel`]. It is
    used to instantiate a RoBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the RoBERTa
    [roberta-base](https://huggingface.co/roberta-base) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    The [`RobertaConfig`] class directly inherits [`BertConfig`]. It reuses the same defaults. Please check the parent
    class for more information.
    Examples:
    ```python
    >>> from transformers import RobertaConfig, RobertaModel
    >>> # Initializing a RoBERTa configuration
    >>> configuration = RobertaConfig()
    >>> # Initializing a model (with random weights) from the configuration
    >>> model = RobertaModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "roberta"

    def __init__(self, 
                pad_token_id=1,
                bos_token_id=0, 
                eos_token_id=2,
                layerdrop=0.1,
                decoder_vocab_size=32,
                output_hidden_size=None,
                num_adapter_layers=2,
                adapter_stride=2,
                adapter_kernel_size=3,
                ctc_loss_reduction="mean",
                ctc_zero_infinity=True,
                is_decoder=False,                
                **kwargs):
        """Constructs RobertaConfig."""
        self.layerdrop = layerdrop
        self.decoder_vocab_size = decoder_vocab_size        
        self.num_adapter_layers = num_adapter_layers
        self.adapter_stride = adapter_stride
        self.adapter_kernel_size = adapter_kernel_size
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.is_decoder = is_decoder
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.output_hidden_size = self.output_hidden_size = output_hidden_size or self.hidden_size