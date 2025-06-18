import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_student(policy_model, dataloader, epochs=10, lr=1e-4, device='cuda'):
    policy_model.to(device)
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        policy_model.train()
        total_loss = 0

        for i, (images, actions) in enumerate(dataloader):
            images = images.to(device)
            actions = actions.to(device)

            preds = policy_model(images)
            loss = criterion(preds, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"[✓] Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")