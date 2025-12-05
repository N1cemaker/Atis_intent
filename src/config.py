# src/config.py

class Config:

    train_path = "data/processed/train.txt"
    valid_path = "data/processed/valid.txt"
    test_path  = "data/processed/test.txt"

    max_len = 50
    batch_size = 64
    emb_dim = 100
    hidden_dim = 128
    num_epochs = 20
    lr = 1e-3
    min_freq = 2

    device = "cuda"  
