import torch
import torch.nn.functional as F

def check_accuracy(loader, model, device="cuda"):
    """Calculates Dice score for the model on the provided data loader."""
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            # Get model predictions
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float() # Binarize predictions
            
            # Calculate intersection and union for Dice score
            intersection = (preds * y).sum()
            union = preds.sum() + y.sum()
            
            # Update Dice score
            dice_score += (2. * intersection) / (union + 1e-8) # Add epsilon to avoid division by zero

    model.train() # Set model back to training mode
    
    # Return the average Dice score over all batches
    return dice_score / len(loader)
