class PretrainedConfig:
    model_type=""
    base_config_key=""
    sub_configs={}
    has_no_defaults_at_init=None
    attribute_map={}
    base_model_tp_plan=None
    base_model_pp_plan=None
    
    def __init__(self,
                output_hidden_state=False,
                output_attentions=False,
                return_dict=True,
                torchscript=False,
                torch_dtype=False,
                purned_heads=None,
                tie_word_embeddings=True,
                chunk_size_feed_forward=0,
                is_encoder_decoder=False,
                is_decoder=False,
                cross_attention_hidden_size=None,
                and_cross_attention=False,
                tie_encoder_decoder=False,
                tokenizer_class=None,
                prefix=None,
                bos_token_id=None,
                pad_token_id=None,
                eos_token_id=None,
                sep_token_id=None,
                decoder_start_token_id=None
                ):
        self.return_dict=return_dict
        self.output_hidden_states=output_hidden_state
        self.torchscript=torchscript
        self.torch_dtype=torch_dtype
        self._output_attention-output_attentions

        self.pruned_heads=purned_heads if purned_heads is not None else {}
        self.tie_word_embeddings=tie_word_embeddings
        self.chunk_size_feed_forward=chunk_size_feed_forward
        self.is_encoder_decoder=is_encoder_decoder
        self.is_decoder=is_decoder
        self.cross_attention_hidden_size=cross_attention_hidden_size
        self.tie_encoder_decoder=tie_encoder_decoder

        

class Grok2_config(PretrainedConfig):
    def __init__(self):
        self.embedding_multiplier_scale = 90.50966799187809
        self.output_multiplier_scale = 0.5
        self.vocab_size = 131072
        self.hidden_size = 8192
        self.intermediate_size = 32768
        self.moe_intermediate_size = 16384
        self.max_position_embeddings = 131072
        self.num_experts_per_tok = 2
        self.num_local_experts = 8
        self.residual_moe = True
        self.num_attention_heads = 64
        self.num_key_value_heads = 8
        self.num_hidden_layers = 64
        self.head_dim = 128
        self.rms_norm_eps = 1e-05
        self.final_logit_softcapping = 50
        self.attn_logit_softcapping = 30.0
        self.router_logit_softcapping = 30.0
        self.rope_theta = 208533496
        self.attn_temperature_len = 1024
        self.sliding_window_size = -1
        self.global_attn_every_n = 1
        self.original_max_position_embeddings = 8192
        self.scaling_factor = 16.0
        self.extrapolation_factor = 1.0
        self.attn_factor = 1.0
        self.beta_fast = 8
        self.beta_slow = 1


class Grok2_config_small(PretrainedConfig):
    def __init__(self):
        self.embedding_multiplier_scale = 90.50966799187809
        self.output_multiplier_scale = 0.5
        self.vocab_size = 10000
        self.hidden_size = 8192
        self.intermediate_size = 32768
        self.moe_intermediate_size = 16384
        self.max_position_embeddings = 10000
        self.context_len = 10000
        self.num_experts_per_tok = 2
        self.num_local_experts = 8
        self.residual_moe = True
        self.num_attention_heads = 64
        self.num_key_value_heads = 8
        self.num_hidden_layers = 1
        self.head_dim = 128
        self.rms_norm_eps = 1e-05
        self.final_logit_softcapping = 50
        self.attn_logit_softcapping = 30.0
        self.router_logit_softcapping = 30.0
        self.rope_theta = 208533496
        self.attn_temperature_len = 1024
        self.sliding_window_size = -1
        self.global_attn_every_n = 1
        self.original_max_position_embeddings = 8192
        self.scaling_factor = 16.0
        self.extrapolation_factor = 1.0
        self.attention_dropout = 0.0
        self.attn_factor = 1.0
        self.beta_fast = 8
        self.beta_slow = 1
        self.bias = False
        self.attention_bias = False
        self.layer_types = [
            "sliding_attention" if bool((i+1)%2) else "full_attention" for i in range(self.num_hidden_layers)
        ]
        self.sliding_window=128
        self.pad_token_idx=0
        self.rope_scaling={
            "rope_type":"yarn",
            "factor":32.0,
            "beta_fast":32.0,
            "beta_slow":1.0,
            "truncate":False
        }

        self.rope_theta=150000.0        
