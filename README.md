There are some problems with the default attention implementation.
We should use flash attention2 instead of the default one.
Besides, keep in mind that the shape of the tensor for labels should be in correspondence with the tensor for text.

These datasets contains labels in the form of 0/1

Remember when saving models to local, keep the precision identical(load model in bf16 to save it to local)

For sequence classification tasks, torch.eq(outputs.logits.argmax(dim=1),batch['lm_labels']).float().sum() could be used to calculate accurate number
