"""ESM-2 + LoRA model for affinity regression.

:
 shared ESM-2 + LoRA encoder
 heavylightantigen  ESM-2 

:
    heavy sequence  -> shared ESM-2 -> mean pooling -> heavy embedding
    light sequence  -> shared ESM-2 -> mean pooling -> light embedding
    antigen sequence -> shared ESM-2 -> mean pooling -> antigen embedding

 embedding , regression head, scalar
"""

import torch
from torch import nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class SeqProFTAffinityRegressor(nn.Module):
    """Shared ESM-2 + LoRA affinity regression model.

    :
    shared backbone :
    heavy / light / antigen  ESM-2 encoder

    ?
    ,ESM-2 
     ESM-2, 3 ,Stage 1 
    """

    def __init__(self, config: dict):
        super().__init__()

        # AutoModel  Hugging Face  ESM-2 backbone
        # add_pooling_layer=False  mean pooling, pooling
        self.esm = AutoModel.from_pretrained(
            config["model_name"],
            add_pooling_layer=False,
        )

        # LoRA  adapter
        #  ESM-2 , LoRA  regression head
        # /,
        lora_config = LoraConfig(
            r=int(config["lora_r"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            target_modules=["query", "value"],
            bias="none",
        )
        self.esm = get_peft_model(self.esm, lora_config)

        hidden_size = self.esm.config.hidden_size

        #  pooled embedding , hidden_size * 3
        self.regression_head = nn.Linear(hidden_size * 3, 1)

        # MSELoss = mean squared error
        #  prediction  label , loss 
        self.loss_fn = nn.MSELoss()

    def mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Average token embeddings while ignoring padding tokens.

        :
        ESM-2  token  embedding,:
            [batch_size, sequence_length, hidden_size]

         regression head 
        mean pooling  token embedding ,:
            [batch_size, hidden_size]

        :padding token  token,
         attention_mask  padding  0
        """

        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        summed_embeddings = (token_embeddings * mask).sum(dim=1)
        token_counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed_embeddings / token_counts

    def encode_sequence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode one sequence type with the shared ESM-2 model.

         heavylight  antigen
        :token ids -> ESM token embeddings -> pooled sequence embedding
        """

        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        return self.mean_pool(outputs.last_hidden_state, attention_mask)

    def forward(
        self,
        heavy_input_ids,
        heavy_attention_mask,
        light_input_ids,
        light_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
        labels=None,
    ) -> dict:
        """Run one forward pass.

        Prediction shape is [batch_size], matching labels shape [batch_size].

        :
        forward 
         labels , loss
        / labels , predictions
        """

        #  ESM-2 encoder, embedding
        heavy_embedding = self.encode_sequence(heavy_input_ids, heavy_attention_mask)
        light_embedding = self.encode_sequence(light_input_ids, light_attention_mask)
        antigen_embedding = self.encode_sequence(antigen_input_ids, antigen_attention_mask)

        # , antibody heavylight  antigen 
        combined_embedding = torch.cat(
            [heavy_embedding, light_embedding, antigen_embedding],
            dim=-1,
        )

        # Linear(..., 1)  [batch_size, 1]
        # squeeze(-1) , [batch_size], labels 
        predictions = self.regression_head(combined_embedding).squeeze(-1)
        output = {"predictions": predictions}

        if labels is not None:
            output["loss"] = self.loss_fn(predictions, labels)

        return output
