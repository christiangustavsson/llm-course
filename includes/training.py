import torch
from tqdm import tqdm
import gc

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Set gradient clipping and accumulation
    max_grad_norm = 1.0
    gradient_accumulation_steps = 8
    optimizer.zero_grad()

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Progress bar for this epoch
        pbar = tqdm(train_loader, desc=f"Training")
        
        for i, (input_batch, target_batch) in enumerate(pbar):
            try:
                # Forward pass
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                
                # Scale loss by gradient accumulation steps
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                tokens_seen += input_batch.numel()
                global_step += 1

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.3f}"})

                # Update weights and clear gradients every accumulation steps
                if (i + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    # Update weights
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Clear cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"\nStep {global_step:06d}: Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("\nWARNING: Out of memory. Skipping this batch.")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e

        # Handle remaining gradients
        if (i + 1) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # Print a sample text after each epoch
        print("\nGenerating sample text...")
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen 