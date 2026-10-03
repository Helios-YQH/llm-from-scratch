import torch

def run_save_checkpoint(model, optimizer, iteration, path):
    """Save a checkpoint of the model, optimizer, and scheduler state."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, path)

def run_load_checkpoint(src, model, optimizer) -> int:
    """Load a checkpoint of the model, optimizer, and scheduler state."""
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]