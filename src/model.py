""":ESM-2 + LoRA + mean pooling + classification head

:

:SeqProFTMiniClassifier

:

protein sequence tokens
-> ESM-2
-> LoRA adapter
-> mean pooling
-> linear classifier
-> logits

:
, contact mapCM-MAH benchmark
"""

import torch
from torch import nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class SeqProFTMiniClassifier(nn.Module):
    """ ESM-2 + LoRA 

    :
    ESM-2  token 
    

     mean pooling:
     token ,
    """

    def __init__(self, config: dict):
        super().__init__()

        #  num_labels = 2
        # :class 0 class 1 
        self.num_labels = int(config["num_labels"])

        # AutoModel  ESM-2  backbone
        #  protein sequence  hidden states,
        #
        # add_pooling_layer=False:
        #  mean pooling, Hugging Face  pooler
        self.esm = AutoModel.from_pretrained(
            config["model_name"],
            add_pooling_layer=False,
        )

        # LoRA :
        #  ESM-2 , adapter
        # ,
        lora_config = LoraConfig(
            # r  LoRA , adapter 
            # r ,,
            r=int(config["lora_r"]),

            # lora_alpha , LoRA 
            lora_alpha=int(config["lora_alpha"]),

            # dropout 
            lora_dropout=float(config["lora_dropout"]),

            # ESM-2  attention  query/key/value
            #  query  value  LoRA,
            target_modules=["query", "value"],

            #  bias, MVP 
            bias="none",
        )
        self.esm = get_peft_model(self.esm, lora_config)

        # ESM-2  token 
        # facebook/esm2_t6_8M_UR50D  hidden_size  320
        hidden_size = self.esm.config.hidden_size

        # :
        # : pooled vector, [batch_size, hidden_size]
        # : logits, [batch_size, num_labels]
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        # CrossEntropyLoss / loss
        #  softmax, softmax
        self.loss_fn = nn.CrossEntropyLoss()

    def mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """ token embeddings , padding

        :
        ESM-2  token :

            token_embeddings shape = [batch_size, sequence_length, hidden_size]

        :

            pooled_features shape = [batch_size, hidden_size]

        attention_mask :
        - 1  token
        - 0  padding

         padding, padding 
        """

        # attention_mask  [batch_size, sequence_length]
        # unsqueeze(-1)  [batch_size, sequence_length, 1],
        #  token_embeddings 
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)

        # padding  mask  0, padding  0
        summed_embeddings = (token_embeddings * mask).sum(dim=1)

        #  token 
        # clamp  0
        token_counts = mask.sum(dim=1).clamp(min=1e-9)

        # : token  /  token 
        return summed_embeddings / token_counts

    def forward(self, input_ids, attention_mask, labels=None) -> dict:
        """: token, logits, loss

        :
        forward 

        :
        -  labels
        -  logits  loss

        :
        -  labels
        -  logits
        """

        # 1.  token ids  ESM-2, token  hidden vector
        esm_outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = esm_outputs.last_hidden_state

        # 2.  token-level  protein-level 
        pooled_features = self.mean_pool(token_embeddings, attention_mask)

        # 3.  logits
        logits = self.classifier(pooled_features)

        output = {"logits": logits}

        # 4.  labels, loss
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)

        return output
