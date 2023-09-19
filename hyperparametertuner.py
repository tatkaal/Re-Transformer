from sklearn.model_selection import ParameterGrid
from retransformer import ReTransformer
import torch.nn as nn
import torch.optim as optim
import torch

def main(SRC_vocab, TGT_vocab, train_dataloader, valid_dataloader, device, epochs):
    print("Source vocab size:", len(SRC_vocab))
    print("Target vocab size:", len(TGT_vocab))
    # Define hyperparameter grid
    param_grid = {
        'd_model': [512],
        'num_heads': [4, 8],
        'num_encoder_layers': [4, 6],
        'num_decoder_layers': [2, 4],
        'dropout': [0.1, 0.2, 0.4],
        'lr': [0.001, 0.0007, 0.0001],
        'weight_decay': [0.0001, 0.00001, 0.000001],
    }

    # Initialize best validation loss to infinity and best params to None
    best_val_loss = float('inf')
    best_params = None

    # Loop over hyperparameter combinations
    for params in ParameterGrid(param_grid):
        print(f"Training with params: {params}")

        # Initialize the Re-Transformer model, Loss, and Optimizer
        model = ReTransformer(
            SRC_vocab, TGT_vocab, params['d_model'], params['num_heads'],
            params['num_encoder_layers'], params['num_decoder_layers'], params['dropout']
        )
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        # Initialize early stopping variables
        patience = 5
        epochs_no_improve = 0

        for epoch in range(epochs):  # Number of epochs
            model.train()
            total_train_loss = 0

            for i, (src, tgt) in enumerate(train_dataloader):
                src, tgt = src.to(device), tgt.to(device)
                optimizer.zero_grad()

                # Forward pass
                output = model(src, tgt)

                # Compute loss
                # Check 1: Output Dimension
                output_dim = output.shape[-1]
                # assert output_dim == len(TGT_vocab), f"Output dimension {output_dim} does not match target vocab size {len(TGT_vocab)}"

                # # Check 2: Target Tensor
                # assert tgt.max().item() < len(TGT_vocab), "Target tensor contains an out-of-bounds index"

                output = output[:-1].view(-1, output_dim)
                tgt = tgt[1:].view(-1)
                loss = criterion(output, tgt)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            # Validation
            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for i, (src, tgt) in enumerate(valid_dataloader):
                    src, tgt = src.to(device), tgt.to(device)

                    # Forward pass
                    output = model(src, tgt)

                    # Compute loss
                    output_dim = output.shape[-1]
                    output = output[:-1].view(-1, output_dim)
                    tgt = tgt[1:].view(-1)
                    val_loss = criterion(output, tgt)

                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(valid_dataloader)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_params = params
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print("Early stopping!")
                    break

    print(f"Best validation loss: {best_val_loss}")
    print(f"Best hyperparameters: {best_params}")

if __name__ == "__main__":
    main()