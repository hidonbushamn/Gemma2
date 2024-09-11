There are some problems with the default attention implementation.
We should use flash attention2 instead of the default one.
Besides, keep in mind that the shape of the tensor for labels should be in correspondence with the tensor for text.
