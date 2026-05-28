""": accuracy

:
 loss,
 accuracy
"""

import torch


def evaluate_accuracy(model, dataloader, device) -> float:
    """

    :
    accuracy =  / 

     4 , 3 :
    accuracy = 3 / 4 = 0.75

     MVP  accuracy 
    , F1AUROC 
    """

    # eval()  dropout 
    # 
    model.eval()
    # 
    correct = 0
    total = 0

    # no_grad :,
    # :
    with torch.no_grad():
        for batch in dataloader:
            # 
            #  mps, mps; cpu, cpu
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # :
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # logits 
            # argmax ,
            predictions = outputs["logits"].argmax(dim=-1)

            # 
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # : dataloader , 0
    if total == 0:
        return 0.0

    return correct / total
