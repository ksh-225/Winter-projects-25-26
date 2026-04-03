import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os


# 1. Define the Neural Network (3 layers, 64 neurons)

class WirelessNet(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, output_size=64):
        super(WirelessNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def functional_forward(x, weights):
    x = F.linear(x, weights['layer1.weight'], weights['layer1.bias'])
    x = F.relu(x)
    x = F.linear(x, weights['layer2.weight'], weights['layer2.bias'])
    x = F.relu(x)
    x = F.linear(x, weights['layer3.weight'], weights['layer3.bias'])
    return x


# 2. MAML Meta-Training Loop

def train_maml():
    print("\n--- Starting MAML Meta-Training ---")
    data = np.load("wireless_dataset.npz")
    train_x_supp = torch.tensor(data['train_x_support'])
    train_y_supp = torch.tensor(data['train_y_support'])
    train_x_query = torch.tensor(data['train_x_query'])
    train_y_query = torch.tensor(data['train_y_query'])

    num_tasks = train_x_supp.shape[0]
    meta_model = WirelessNet()
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001) 
    
    iterations = 500
    meta_batch_size = 4  
    inner_steps = 5      
    inner_lr = 0.01      
    
    meta_losses = []

    for iteration in range(iterations):
        meta_optimizer.zero_grad()
        meta_batch_loss = 0.0
        task_indices = np.random.choice(num_tasks, meta_batch_size, replace=False)
        
        for task_idx in task_indices:
            x_supp, y_supp = train_x_supp[task_idx], train_y_supp[task_idx]
            x_query, y_query = train_x_query[task_idx], train_y_query[task_idx]

            fast_weights = {name: param for name, param in meta_model.named_parameters()}

            for _ in range(inner_steps):
                preds = functional_forward(x_supp, fast_weights)
                loss = F.mse_loss(preds, y_supp)
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                fast_weights = {name: param - inner_lr * grad for ((name, param), grad) in zip(fast_weights.items(), grads)}

            query_preds = functional_forward(x_query, fast_weights)
            query_loss = F.mse_loss(query_preds, y_query)
            meta_batch_loss += query_loss

        meta_batch_loss = meta_batch_loss / meta_batch_size
        meta_batch_loss.backward()  
        meta_optimizer.step()
        meta_losses.append(meta_batch_loss.item())

        if (iteration + 1) % 50 == 0:
            print(f"Iteration {iteration + 1}/{iterations} | MAML Meta-Loss: {meta_batch_loss.item():.4f}")

    torch.save(meta_model.state_dict(), "meta_model.pth")
    print("MAML Meta-training complete. Saved to meta_model.pth.")
    return meta_losses


# 3. Baseline Training Loop

def train_baseline_on_test_tasks():
    print("\n--- Starting Baseline Training on Test Tasks ---")
    data = np.load("wireless_dataset.npz")
    test_x_supp = torch.tensor(data['test_x_support'])
    test_y_supp = torch.tensor(data['test_y_support'])
    test_x_query = torch.tensor(data['test_x_query'])
    test_y_query = torch.tensor(data['test_y_query'])

    num_test_tasks = test_x_supp.shape[0]
    total_steps = 200
    inner_lr = 0.01

    # Array to track average error at every step from 0 to 200
    baseline_errors = np.zeros(total_steps + 1)

    for task_idx in range(num_test_tasks):
        x_supp, y_supp = test_x_supp[task_idx], test_y_supp[task_idx]
        x_query, y_query = test_x_query[task_idx], test_y_query[task_idx]

        # Initialize a regular network from scratch for EACH test task
        baseline_model = WirelessNet()
        optimizer = optim.Adam(baseline_model.parameters(), lr=inner_lr)
        
        # Record Step 0 error (no training yet)
        with torch.no_grad():
            baseline_errors[0] += F.mse_loss(baseline_model(x_query), y_query).item()

        # Train for 200 steps on the support set
        for step in range(1, total_steps + 1):
            optimizer.zero_grad()
            loss = F.mse_loss(baseline_model(x_supp), y_supp)
            loss.backward()
            optimizer.step()
            
            # Record error on the query set
            with torch.no_grad():
                baseline_errors[step] += F.mse_loss(baseline_model(x_query), y_query).item()

    # Average over all 20 test tasks
    baseline_errors /= num_test_tasks
    

    np.save("baseline_errors.npy", baseline_errors)
    print(f"Baseline trained on {num_test_tasks} tasks for {total_steps} steps.")
    print("Baseline errors saved to baseline_errors.npy.")


# 4. Generate Plot 1: Training Loss Curve

def plot_training_loss(losses):
    smoothed_losses = [np.mean(losses[max(0, i-10):i+1]) for i in range(len(losses))]
    os.makedirs("results", exist_ok=True) 
    
    plt.figure(figsize=(8, 5))
    plt.plot(smoothed_losses, label="Query Loss (Smoothed)", color='blue')
    plt.title("Plot 1: Meta-Training Loss Curve (MAML)")
    plt.xlabel("Meta-Training Iteration")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("results/plot_loss.png")
    print("\nPlot 1 saved to results/plot_loss.png.")

if __name__ == "__main__":
    # 1. Meta-train MAML
    maml_losses = train_maml()
    
    # 2. Train baseline on test tasks for 200 steps
    train_baseline_on_test_tasks()
    
    # 3. Plot MAML loss
    plot_training_loss(maml_losses)
