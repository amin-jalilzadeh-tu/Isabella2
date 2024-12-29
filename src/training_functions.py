# src/training_functions.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cvxpy as cp  # for MGDA and CAGrad if needed

############################################################
# Early Stopping
############################################################
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=1e-4, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f'Validation loss decreased. Saving model to {self.path}')


############################################################
# Path Helper Functions
############################################################
def get_model_path(method, model_type):
    """Generate a unique path for saving the model."""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    filename = f'model_{method}_{model_type}.pt'
    return os.path.join(models_dir, filename)

def get_scaler_path(method, model_type, scaler_type='X'):
    """Generate a unique path for saving the scaler."""
    scalers_dir = 'scalers'
    if not os.path.exists(scalers_dir):
        os.makedirs(scalers_dir)
    filename = f'scaler_{scaler_type}_{method}_{model_type}.joblib'
    return os.path.join(scalers_dir, filename)

def get_dynamic_model_path(method, model_type):
    return get_model_path(method, model_type)


############################################################
# Custom Loss
############################################################
def custom_loss(output, target):
    """
    Basic MSE + penalty for negative predictions (if needed).
    """
    mse_loss = nn.MSELoss()(output, target)
    penalty = torch.mean(torch.relu(-output))  # Penalize negative outputs
    return mse_loss + penalty


############################################################
# Weighted Sum Training
############################################################
def train_weighted_sum(
    model, train_loader, val_loader, num_epochs, learning_rate, 
    weights, method='weighted_sum', model_type='shared'
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model_path = get_dynamic_model_path(method, model_type)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=model_path)

    history = {
        'train_total_loss': [],
        'train_task_losses': [[] for _ in weights],
        'val_total_loss': [],
        'val_task_losses': [[] for _ in weights]
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0
        task_losses_sum = [0.0 for _ in weights]
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)  # outputs is a tuple/list of length = number of tasks
            losses = []
            for i in range(len(outputs)):
                loss = custom_loss(outputs[i], targets[:, i].unsqueeze(1))
                losses.append(loss)
            total_loss_batch = sum(w * l for w, l in zip(weights, losses))
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()
            for i, loss in enumerate(losses):
                task_losses_sum[i] += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_task_losses = [tl / len(train_loader) for tl in task_losses_sum]
        history['train_total_loss'].append(avg_train_loss)
        for i, tl in enumerate(avg_train_task_losses):
            history['train_task_losses'][i].append(tl)

        # Validation
        model.eval()
        val_total_loss = 0.0
        val_task_losses_sum = [0.0 for _ in weights]
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                losses = []
                for i in range(len(outputs)):
                    loss = custom_loss(outputs[i], targets[:, i].unsqueeze(1))
                    losses.append(loss)
                total_loss_batch = sum(w * l for w, l in zip(weights, losses))
                val_total_loss += total_loss_batch.item()
                for i, loss in enumerate(losses):
                    val_task_losses_sum[i] += loss.item()

        avg_val_loss = val_total_loss / len(val_loader)
        avg_val_task_losses = [tl / len(val_loader) for tl in val_task_losses_sum]
        history['val_total_loss'].append(avg_val_loss)
        for i, tl in enumerate(avg_val_task_losses):
            history['val_task_losses'][i].append(tl)

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Total Loss: {avg_train_loss:.4f}")
        for i, tl in enumerate(avg_train_task_losses):
            print(f"    Task {i+1} Loss: {tl:.4f}")
        print(f"  Val Total Loss: {avg_val_loss:.4f}")
        for i, tl in enumerate(avg_val_task_losses):
            print(f"    Task {i+1} Loss: {tl:.4f}")

        # Early Stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load best model
    model.load_state_dict(torch.load(model_path))
    return history


############################################################
# MGDA Training
############################################################
def train_mgda(
    model, train_loader, val_loader, num_epochs, learning_rate,
    method='mgda', model_type='shared'
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model_path = get_dynamic_model_path(method, model_type)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=model_path)

    num_tasks = 4  # adjust if needed
    history = {
        'train_total_loss': [],
        'train_task_losses': [[] for _ in range(num_tasks)],
        'val_total_loss': [],
        'val_task_losses': [[] for _ in range(num_tasks)]
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        task_losses_sum = [0.0 for _ in range(num_tasks)]

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            task_losses = [
                custom_loss(outputs[i], targets[:, i].unsqueeze(1)) for i in range(num_tasks)
            ]

            # Compute gradients for each task separately
            task_gradients = []
            for loss in task_losses:
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                grads = []
                for param in model.shared_layers.parameters():
                    if param.grad is not None:
                        grads.append(param.grad.view(-1))
                if grads:
                    grad_vector = torch.cat(grads).cpu().numpy()
                else:
                    grad_vector = np.zeros(sum(p.numel() for p in model.shared_layers.parameters()))
                task_gradients.append(grad_vector)
                optimizer.zero_grad()  # reset for next

            G = np.stack(task_gradients)

            # Solve QP to find alphas
            try:
                alpha = cp.Variable(num_tasks)
                objective = cp.Minimize(0.5 * cp.quad_form(alpha, G @ G.T))
                constraints = [cp.sum(alpha) == 1, alpha >= 0]
                prob = cp.Problem(objective, constraints)
                prob.solve()
                alphas = alpha.value
                if alphas is None:
                    alphas = np.ones(num_tasks) / num_tasks
            except:
                alphas = np.ones(num_tasks) / num_tasks

            combined_grad = np.zeros_like(task_gradients[0])
            for i in range(num_tasks):
                combined_grad += alphas[i] * task_gradients[i]
            combined_grad_tensor = torch.from_numpy(combined_grad).float().to(device)

            # Assign combined gradient
            index = 0
            for param in model.shared_layers.parameters():
                param_num = param.numel()
                param.grad = combined_grad_tensor[index : index + param_num].view_as(param).clone()
                index += param_num

            # Compute total loss
            total_loss_batch = sum(task_losses)
            total_loss_batch.backward()  # for task-specific layers
            optimizer.step()

            for i in range(num_tasks):
                task_losses_sum[i] += task_losses[i].item()
            total_loss += sum(t.item() for t in task_losses)

        avg_train_loss = total_loss / len(train_loader)
        avg_train_task_losses = [tl / len(train_loader) for tl in task_losses_sum]
        history['train_total_loss'].append(avg_train_loss)
        for i, tl in enumerate(avg_train_task_losses):
            history['train_task_losses'][i].append(tl)

        # Validation
        model.eval()
        val_total_loss = 0.0
        val_task_losses_sum = [0.0 for _ in range(num_tasks)]
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                losses = [
                    custom_loss(outputs[i], targets[:, i].unsqueeze(1)).item() for i in range(num_tasks)
                ]
                val_total_loss += sum(losses)
                for i in range(num_tasks):
                    val_task_losses_sum[i] += losses[i]

        avg_val_loss = val_total_loss / len(val_loader)
        avg_val_task_losses = [tl / len(val_loader) for tl in val_task_losses_sum]
        history['val_total_loss'].append(avg_val_loss)
        for i, tl in enumerate(avg_val_task_losses):
            history['val_task_losses'][i].append(tl)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Total Loss: {avg_train_loss:.4f}")
        for i, tl in enumerate(avg_train_task_losses):
            print(f"    Task {i+1} Loss: {tl:.4f}")
        print(f"  Val Total Loss: {avg_val_loss:.4f}")
        for i, tl in enumerate(avg_val_task_losses):
            print(f"    Task {i+1} Loss: {tl:.4f}")

        # EarlyStopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model.load_state_dict(torch.load(model_path))
    return history


############################################################
# Uncertainty Training
############################################################
def train_uncertainty(
    model, train_loader, val_loader, num_epochs, learning_rate,
    method='uncertainty', model_type='shared'
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model_path = get_dynamic_model_path(method, model_type)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=model_path)

    history = {
        'train_total_loss': [],
        'train_task_losses': [[] for _ in range(model.num_tasks)],
        'val_total_loss': [],
        'val_task_losses': [[] for _ in range(model.num_tasks)]
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        task_losses_sum = [0.0 for _ in range(model.num_tasks)]

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Per-task losses
            task_losses = []
            for i in range(model.num_tasks):
                loss = custom_loss(outputs[i], targets[:, i].unsqueeze(1))
                task_losses.append(loss)

            # Weighted by uncertainty
            loss_total = 0
            for i in range(model.num_tasks):
                loss_total += 0.5 * torch.exp(-2 * model.log_sigma[i]) * task_losses[i] + model.log_sigma[i]

            loss_total.backward()
            optimizer.step()

            # Clamp log_sigma
            for i in range(model.num_tasks):
                model.log_sigma.data[i] = torch.clamp(model.log_sigma.data[i], min=-10.0, max=10.0)

            total_loss += loss_total.item()
            for i, l in enumerate(task_losses):
                task_losses_sum[i] += l.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_task_losses = [tl / len(train_loader) for tl in task_losses_sum]
        history['train_total_loss'].append(avg_train_loss)
        for i, tl in enumerate(avg_train_task_losses):
            history['train_task_losses'][i].append(tl)

        # Validation
        model.eval()
        val_total_loss = 0.0
        val_task_losses_sum = [0.0 for _ in range(model.num_tasks)]
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                # Per-task losses
                task_losses = []
                for i in range(model.num_tasks):
                    loss = custom_loss(outputs[i], targets[:, i].unsqueeze(1))
                    task_losses.append(loss)
                # Weighted by uncertainty
                loss_total = 0
                for i in range(model.num_tasks):
                    loss_total += 0.5 * torch.exp(-2 * model.log_sigma[i]) * task_losses[i] + model.log_sigma[i]
                val_total_loss += loss_total.item()
                for i, l in enumerate(task_losses):
                    val_task_losses_sum[i] += l.item()

        avg_val_loss = val_total_loss / len(val_loader)
        avg_val_task_losses = [tl / len(val_loader) for tl in val_task_losses_sum]
        history['val_total_loss'].append(avg_val_loss)
        for i, tl in enumerate(avg_val_task_losses):
            history['val_task_losses'][i].append(tl)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Total Loss: {avg_train_loss:.4f}")
        for i, tl in enumerate(avg_train_task_losses):
            print(f"    Task {i+1} Loss: {tl:.4f}")
        print(f"  Val Total Loss: {avg_val_loss:.4f}")
        for i, tl in enumerate(avg_val_task_losses):
            print(f"    Task {i+1} Loss: {tl:.4f}")

        for i in range(model.num_tasks):
            print(f"  Task {i+1} log_sigma: {model.log_sigma[i].item():.4f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model.load_state_dict(torch.load(model_path))
    return history


############################################################
# CAGrad Training
############################################################
def train_cagrad(
    model, train_loader, val_loader, num_epochs, learning_rate,
    method='cagrad', model_type='shared'
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model_path = get_dynamic_model_path(method, model_type)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=model_path)

    history = {
        'train_total_loss': [],
        'train_task_losses': [[] for _ in range(model.num_tasks)],
        'val_total_loss': [],
        'val_task_losses': [[] for _ in range(model.num_tasks)]
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        task_losses_sum = [0.0 for _ in range(model.num_tasks)]

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Per-task losses
            task_losses = [custom_loss(outputs[i], targets[:, i].unsqueeze(1)) for i in range(model.num_tasks)]

            # Compute gradients for each task
            optimizer.zero_grad()
            for loss in task_losses:
                loss.backward(retain_graph=True)

            # Extract gradients for shared parameters
            shared_params = [p for p in model.shared_layers.parameters() if p.grad is not None]
            task_gradients = []
            for i in range(model.num_tasks):
                grad = []
                for param in shared_params:
                    grad.append(param.grad.detach().clone())
                task_gradients.append(grad)

            # Reset grads to zero
            optimizer.zero_grad()

            # Stack gradients
            grad_stack = []
            for grads in task_gradients:
                flat_grad = torch.cat([g.view(-1) for g in grads])
                grad_stack.append(flat_grad)
            grad_matrix = torch.stack(grad_stack)

            # Compute Gram matrix
            gram_matrix = grad_matrix @ grad_matrix.t()

            # Solve QP to find alpha
            try:
                import cvxpy as cp
                alpha = cp.Variable(model.num_tasks)
                objective = cp.Minimize(cp.sum_squares(grad_matrix.t() @ alpha))
                constraints = [cp.sum(alpha) == 1, alpha >= 0]
                prob = cp.Problem(objective, constraints)
                prob.solve()
                alphas = alpha.value
                if alphas is None:
                    alphas = np.ones(model.num_tasks) / model.num_tasks
            except:
                alphas = np.ones(model.num_tasks) / model.num_tasks

            combined_grad = torch.zeros_like(grad_matrix[0])
            for i in range(model.num_tasks):
                combined_grad += alphas[i] * grad_matrix[i]

            # Assign combined gradients
            index = 0
            for param in shared_params:
                numel = param.numel()
                param.grad = combined_grad[index : index + numel].view_as(param).clone()
                index += numel

            # Sum of task losses
            total_loss_batch = sum(task_losses)
            total_loss_batch.backward()  # for task-specific parts
            optimizer.step()

            # Accumulate
            for i in range(model.num_tasks):
                task_losses_sum[i] += task_losses[i].item()
            total_loss += total_loss_batch.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_task_losses = [tl / len(train_loader) for tl in task_losses_sum]
        history['train_total_loss'].append(avg_train_loss)
        for i, tl in enumerate(avg_train_task_losses):
            history['train_task_losses'][i].append(tl)

        # Validation
        model.eval()
        val_total_loss = 0.0
        val_task_losses_sum = [0.0 for _ in range(model.num_tasks)]
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                losses = [custom_loss(outputs[i], targets[:, i].unsqueeze(1)) for i in range(model.num_tasks)]
                total_loss_batch = sum(losses)
                val_total_loss += total_loss_batch.item()
                for i, loss in enumerate(losses):
                    val_task_losses_sum[i] += loss.item()

        avg_val_loss = val_total_loss / len(val_loader)
        avg_val_task_losses = [tl / len(val_loader) for tl in val_task_losses_sum]
        history['val_total_loss'].append(avg_val_loss)
        for i, tl in enumerate(avg_val_task_losses):
            history['val_task_losses'][i].append(tl)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Total Loss: {avg_train_loss:.4f}")
        for i, tl in enumerate(avg_train_task_losses):
            print(f"    Task {i+1} Loss: {tl:.4f}")
        print(f"  Val Total Loss: {avg_val_loss:.4f}")
        for i, tl in enumerate(avg_val_task_losses):
            print(f"    Task {i+1} Loss: {tl:.4f}")

        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model.load_state_dict(torch.load(model_path))
    return history
