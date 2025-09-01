from setuptools import setup, find_packages

setup(
    name="flow-grpo",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio",
        "transformers==4.55.2",
        "accelerate==1.10.0",
        "diffusers==0.35.1", 
        
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.15.2",
        "scikit-learn==1.6.1",
        "scikit-image==0.25.2",
        
        "albumentations==1.4.10",  
        "opencv-python==4.11.0.86",
        "pillow==10.4.0",
        
        "tqdm==4.67.1",
        "wandb==0.18.7",
        "pydantic==2.10.6",  
        "requests==2.32.3",
        "matplotlib==3.10.0",
        
        # "flash-attn==2.7.4.post1",
        "deepspeed==0.16.4",  
        "peft==0.17.1",       
        "bitsandbytes==0.45.3",
        
        "aiohttp==3.11.18",
        "fastapi==0.115.11", 
        "uvicorn==0.34.0",
        
        "huggingface-hub==0.34.4",  
        "datasets==3.6.0",
        "tokenizers==0.21.4",
        "timm==1.0.19",
        
        "einops==0.8.1",
        "nvidia-ml-py==12.570.86",
        "xformers",
        "absl-py",
        "ml_collections",
        "sentencepiece",
        "openai",

        "gpytorch==1.14",
    ],
    extras_require={
        "dev": [
            "ipython==8.34.0",
            "black==24.2.0",
            "pytest==8.2.0"
        ]
    }
)
