import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# 1) Neural Network Definition
###############################################################################
def make_nn():
    """
    Neural network with inputs:
      S (asset price), t (time in [0,T], with t=0 => maturity),
      K, r, sigma
    Outputs:
      [call_price, put_price]
    """
    return nn.Sequential(
        nn.Linear(5, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 2)
    )

###############################################################################
# 2) PDE Loss:  -dV/dt + 0.5*sigma^2*S^2*d2V/dS^2 + r*S*dV/dS - r*V = 0
#    (Backward in time: t=0 => maturity, t=T => 'today')
###############################################################################
def black_scholes_pde_loss(model, S, t, K, r, sigma):
    """
    Enforce the backward Black-Scholes PDE for both call and put:
      -V_t + 0.5*sigma^2*S^2*V_SS + r*S*V_S - r*V = 0
    """
    S.requires_grad = True
    t.requires_grad = True

    X = torch.cat([S, t, K, r, sigma], dim=1)  # shape: [N,5]
    V = model(X)                               # shape: [N,2]
    V_call = V[:, 0:1]
    V_put  = V[:, 1:2]

    # Derivatives for the call
    dV_call_t = torch.autograd.grad(
        V_call, t, torch.ones_like(V_call), create_graph=True
    )[0]
    dV_call_S = torch.autograd.grad(
        V_call, S, torch.ones_like(V_call), create_graph=True
    )[0]
    d2V_call_SS = torch.autograd.grad(
        dV_call_S, S, torch.ones_like(dV_call_S), create_graph=True
    )[0]

    # Derivatives for the put
    dV_put_t = torch.autograd.grad(
        V_put, t, torch.ones_like(V_put), create_graph=True
    )[0]
    dV_put_S = torch.autograd.grad(
        V_put, S, torch.ones_like(V_put), create_graph=True
    )[0]
    d2V_put_SS = torch.autograd.grad(
        dV_put_S, S, torch.ones_like(dV_put_S), create_graph=True
    )[0]

    # PDE residuals
    res_call = (
        -dV_call_t
        + 0.5 * sigma**2 * S**2 * d2V_call_SS
        + r * S * dV_call_S
        - r * V_call
    )
    res_put = (
        -dV_put_t
        + 0.5 * sigma**2 * S**2 * d2V_put_SS
        + r * S * dV_put_S
        - r * V_put
    )

    pde_loss = torch.mean(res_call**2) + torch.mean(res_put**2)
    return pde_loss

###############################################################################
# 3) Boundary/Initial Conditions
###############################################################################
def boundary_condition_loss(
    model, S0, t0, K0, r_val, sigma_val, T_val, S_max=200.0
):
    """
    1) At t=0 (maturity):
       call = max(S-K,0), put = max(K-S,0)
    2) At S=0:
       call=0, put=K*exp(-r*(T-t))  (approx)
    3) At S=S_max (large):
       call ~ S - K*exp(-r*(T-t)), put ~ 0  (approx)

    Note: Here we let K vary with S (randomly), which may lead to edge cases
          (e.g. K=0 if S=0). This is done to "keep everything else the same"
          but incorporate K in [0.5S, 1.5S].
    """
    device = S0.device

    # (a) Condition at t=0 => payoff
    r_t0     = torch.full_like(S0, r_val)
    sigma_t0 = torch.full_like(S0, sigma_val)
    X_t0 = torch.cat([S0, t0, K0, r_t0, sigma_t0], dim=1)
    V_t0 = model(X_t0)  # shape: [N,2]
    call_payoff = torch.maximum(S0 - K0, torch.tensor(0.0, device=device))
    put_payoff  = torch.maximum(K0 - S0, torch.tensor(0.0, device=device))
    loss_t0 = torch.mean((V_t0[:,0:1] - call_payoff)**2) \
             +torch.mean((V_t0[:,1:2] - put_payoff)**2)

    # (b) Condition at S=0 => call=0, put=K*exp(-r*(T-t))
    n_bc = S0.shape[0]
    S_zero = torch.zeros((n_bc,1), device=device)
    t_bc   = torch.linspace(0, T_val, n_bc, device=device).unsqueeze(1)
    # For K at S=0, we also randomize in [0.5*S, 1.5*S] => effectively 0
    K_bc   = torch.rand_like(S_zero)*(1.5 - 0.5) + 0.5  # uniform in [0.5,1.5]
    K_bc   = K_bc * S_zero  # => 0 if S=0
    r_bc   = torch.full_like(S_zero, r_val)
    sigma_bc = torch.full_like(S_zero, sigma_val)
    X_s0 = torch.cat([S_zero, t_bc, K_bc, r_bc, sigma_bc], dim=1)
    V_s0 = model(X_s0)
    call_s0_target = torch.zeros_like(S_zero)
    put_s0_target  = K_bc * torch.exp(-r_bc*(T_val - t_bc))
    loss_s0 = torch.mean((V_s0[:,0:1] - call_s0_target)**2) \
             +torch.mean((V_s0[:,1:2] - put_s0_target)**2)

    # (c) Condition at S=S_max => call ~ S_max - K*exp(-r*(T-t)), put ~ 0
    S_hi = torch.full((n_bc,1), S_max, device=device)
    t_hi = t_bc  # reuse the same t values
    # Random K in [0.5*S_max, 1.5*S_max]
    K_hi = torch.rand_like(S_hi)*(1.5 - 0.5) + 0.5
    K_hi = K_hi * S_hi
    X_smax = torch.cat([S_hi, t_hi, K_hi, r_bc, sigma_bc], dim=1)
    V_smax = model(X_smax)
    call_smax_target = S_max - K_hi*torch.exp(-r_bc*(T_val - t_hi))
    put_smax_target  = torch.zeros_like(S_hi)
    loss_smax = torch.mean((V_smax[:,0:1] - call_smax_target)**2) \
               +torch.mean((V_smax[:,1:2] - put_smax_target)**2)

    return loss_t0 + loss_s0 + loss_smax

###############################################################################
# 4) Training Function (with random K in [0.5*S, 1.5*S])
###############################################################################
def train_pinn(
    r_val=0.05, sigma_val=0.2, T_val=1.0,
    S_max=200.0, num_epochs=5000, save_path="pinn_model.pth"
):
    """
    Train a PINN to solve the backward-time Black-Scholes PDE for both
    European calls and puts, with:
      t in [0, T_val], S in [0, S_max],
      K in [0.5*S, 1.5*S].
    At t=0, we impose payoff conditions. At S=0 and S=S_max, we impose
    boundary conditions for calls/puts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_nn().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4(a) Sample interior points for PDE loss
    N_int = 3000
    S_int = torch.rand(N_int,1)*(S_max)          # S in [0, S_max]
    t_int = torch.rand(N_int,1)*(T_val)          # t in [0, T_val]
    r_int = torch.full_like(S_int, r_val)
    sigma_int = torch.full_like(S_int, sigma_val)

    # K in [0.5*S, 1.5*S]
    K_int = torch.rand_like(S_int)*(1.5 - 0.5) + 0.5  # uniform in [0.5,1.5]
    K_int = K_int * S_int

    S_int, t_int, K_int, r_int, sigma_int = [
        x.to(device) for x in (S_int, t_int, K_int, r_int, sigma_int)
    ]

    # 4(b) Sample boundary/initial condition points
    N_bc = 100
    # For payoff at t=0
    S0_bc = torch.linspace(0, S_max, N_bc).unsqueeze(1)
    t0_bc = torch.zeros_like(S0_bc)
    # K also in [0.5*S, 1.5*S]
    K0_bc = torch.rand_like(S0_bc)*(1.5 - 0.5) + 0.5
    K0_bc = K0_bc * S0_bc

    S0_bc, t0_bc, K0_bc = [x.to(device) for x in (S0_bc, t0_bc, K0_bc)]

    # 4(c) Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # PDE residual
        pde_loss = black_scholes_pde_loss(model, S_int, t_int, K_int, r_int, sigma_int)

        # Boundary/initial conditions
        bc_loss = boundary_condition_loss(
            model, S0_bc, t0_bc, K0_bc, r_val, sigma_val, T_val, S_max
        )

        loss = pde_loss + bc_loss
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

###############################################################################
# 5) Inference (Predict at t=T, i.e., "today")
###############################################################################
def predict_option_prices(model, S0, K_val, T_val, r_val, sigma_val):
    """
    Evaluate the network at time t=T (the 'today' value, i.e. T years before maturity).
    Because we used t=0 as maturity in the PDE, t=T is the solution we want.
    """
    device = next(model.parameters()).device

    S0_tensor = torch.tensor([[S0]], dtype=torch.float32, device=device)
    tT_tensor = torch.tensor([[T_val]], dtype=torch.float32, device=device)  # "today"
    K_tensor  = torch.tensor([[K_val]], dtype=torch.float32, device=device)
    r_tensor  = torch.tensor([[r_val]], dtype=torch.float32, device=device)
    sigma_tensor = torch.tensor([[sigma_val]], dtype=torch.float32, device=device)

    X = torch.cat([S0_tensor, tT_tensor, K_tensor, r_tensor, sigma_tensor], dim=1)
    with torch.no_grad():
        V = model(X)  # shape [1,2]
    call_price = V[0,0].item()
    put_price  = V[0,1].item()
    return call_price, put_price

###############################################################################
# 6) Example Usage
###############################################################################
"""
# 1) Train the model with random K in [0.5*S, 1.5*S].
model = train_pinn(
    r_val=0.05, sigma_val=0.2, T_val=1.0,
    S_max=200.0, num_epochs=5000, save_path="pinn_model.pth"
)

# 2) Load the model (if already trained)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = make_nn().to(device)
# model.load_state_dict(torch.load("pinn_model.pth", map_location=device))
# model.eval()

# 3) Predict call/put prices at "today" (t=T).
call_price, put_price = predict_option_prices(model, S0=100.0, K_val=120.0,
                                              T_val=1.0, r_val=0.05, sigma_val=0.2)
print("PINN Call Price:", call_price)
print("PINN Put Price: ", put_price)
"""



