from huggingface_hub import hf_hub_download

print("Downloading…")
p = hf_hub_download("moka-ai/m3e-base","config.json")
print("OK:", p)
