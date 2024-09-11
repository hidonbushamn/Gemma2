from transformers import AutoModel, AutoTokenizer
import huggingface_hub
huggingface_hub.login("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") # token 从 https://huggingface.co/settings/tokens 获取
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
