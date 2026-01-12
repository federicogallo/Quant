import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from PIL import Image






def European_Black_Scholes_Formula(S0, K, T, r, sigma, option_type="call"):

    # Calculate d1 and d2 using Black-Scholes Formula
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)


    if option_type == "call":
        # Calculate Call Option Price using Black-Scholes Formula
        option_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
    # Calculate Put Option Price using Black-Scholes Formula
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    return option_price







#def crank_nicolson_european_option(S0, K, T, r, sigma, S_max, M, N, option_type="call"):
def European_Crank_Nicolson(S0, K, T, r, sigma, S_max, M, N, option_type="call"):
    # Step sizes
    dS = S_max / M
    dt = T / N

    # Grid setup
    S = np.linspace(0, S_max, M+1)
    V = np.zeros((M+1, N+1))

    # Boundary conditions
    if option_type == "call":
        V[:, -1] = np.maximum(S - K, 0)  # Payoff for call option
        V[-1, :] = S_max - K * np.exp(-r * dt * np.arange(N+1))  # Call option boundary condition at S_max
    elif option_type == "put":
        V[:, -1] = np.maximum(K - S, 0)  # Payoff for put option
        V[0, :] = K * np.exp(-r * dt * np.arange(N+1))  # Put option boundary condition at S = 0

    # Crank-Nicolson coefficients
    alpha = 0.25 * dt * (sigma**2 * np.arange(1, M)**2 - r * np.arange(1, M))
    beta = -0.5 * dt * (sigma**2 * np.arange(1, M)**2 + r)
    gamma = 0.25 * dt * (sigma**2 * np.arange(1, M)**2 + r * np.arange(1, M))

    # Create the tridiagonal matrix
    A = np.diag(1 - beta) + np.diag(-alpha[1:], -1) + np.diag(-gamma[:-1], 1)
    B = np.diag(1 + beta) + np.diag(alpha[1:], -1) + np.diag(gamma[:-1], 1)

    # Iterate backward in time
    for j in range(N-1, -1, -1):
        V[1:M, j] = np.linalg.solve(A, B @ V[1:M, j+1])

    # Interpolating to get the option price at S0
    option_price = np.interp(S0, S, V[:, 0])
    
    return option_price













#def monte_carlo_option_pricing(S0, K, T, r, sigma, num_simulations, option_type="call"):
def European_Monte_Carlo(S0, K, T, r, sigma, num_simulations, option_type="call"):
    # Generate random standard normal variables
    Z = np.random.standard_normal(num_simulations)
    
    # Calculate stock price at maturity for each simulation
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Calculate the payoff for each simulation
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:  # option_type == "put"
        payoffs = np.maximum(K - ST, 0)
    
    # Calculate the present value of the expected payoff
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price










#def crank_nicolson_american_option(S0, K, T, r, sigma, S_max, M, N, option_type="call"):
def American_Crank_Nicolson(S0, K, T, r, sigma, S_max, M, N, option_type="call"):
    """
    Crank-Nicolson method for pricing American call/put options.
    
    Parameters:
    - S0: Current stock price
    - K: Strike price
    - T: Time to expiration
    - r: Risk-free interest rate
    - sigma: Volatility
    - S_max: Maximum stock price in the grid
    - M: Number of stock price steps
    - N: Number of time steps
    - option_type: "call" or "put"
    
    Returns:
    - Option price at S0
    """
    # Step sizes
    dS = S_max / M
    dt = T / N

    # Grid setup
    S = np.linspace(0, S_max, M+1)
    V = np.zeros((M+1, N+1))

    # Boundary conditions
    if option_type == "call":
        V[:, -1] = np.maximum(S - K, 0)  # Payoff at maturity for call
        V[-1, :] = S_max - K * np.exp(-r * dt * np.arange(N+1))  # Call boundary condition at S_max
    elif option_type == "put":
        V[:, -1] = np.maximum(K - S, 0)  # Payoff at maturity for put
        V[0, :] = K * np.exp(-r * dt * np.arange(N+1))  # Put boundary condition at S = 0

    # Crank-Nicolson coefficients
    alpha = 0.25 * dt * (sigma**2 * np.arange(1, M)**2 - r * np.arange(1, M))
    beta = -0.5 * dt * (sigma**2 * np.arange(1, M)**2 + r)
    gamma = 0.25 * dt * (sigma**2 * np.arange(1, M)**2 + r * np.arange(1, M))

    # Create the tridiagonal matrices
    A = np.diag(1 - beta) + np.diag(-alpha[1:], -1) + np.diag(-gamma[:-1], 1)
    B = np.diag(1 + beta) + np.diag(alpha[1:], -1) + np.diag(gamma[:-1], 1)

    # Iterate backward in time
    for j in range(N-1, -1, -1):
        V[1:M, j] = np.linalg.solve(A, B @ V[1:M, j+1])

        # Apply early exercise condition (for American options)
        if option_type == "call":
            V[1:M, j] = np.maximum(V[1:M, j], S[1:M] - K)
        elif option_type == "put":
            V[1:M, j] = np.maximum(V[1:M, j], K - S[1:M])

    # Interpolating to get the option price at S0
    option_price = np.interp(S0, S, V[:, 0])
    
    return option_price












#def binomial_tree_american_option(S0, K, T, r, sigma, N, option_type="call", dividend_yield=0):
def American_Binomial_Tree(S0, K, T, r, sigma, N, option_type="call", dividend_yield=0):

    """
    Binomial Tree method for pricing American call/put options.
    
    Parameters:
    - S0: Current stock price
    - K: Strike price
    - T: Time to expiration (in years)
    - r: Risk-free interest rate
    - sigma: Volatility (annualized)
    - N: Number of time steps
    - option_type: "call" or "put"
    - dividend_yield: Dividend yield (set to 0 if no dividends)
    
    Returns:
    - Option price at S0
    """
    # Time step
    dt = T / N
    # Up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    # Adjust the risk-free rate for the dividend yield
    p = (np.exp((r - dividend_yield) * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    S = np.zeros(N+1)
    S[0] = S0 * d**N
    for j in range(1, N+1):
        S[j] = S[j-1] * u / d
    
    # Initialize option values at maturity
    V = np.zeros(N+1)
    if option_type == "call":
        for j in range(N+1):
            V[j] = max(S[j] - K, 0)  # Call payoff at maturity
    elif option_type == "put":
        for j in range(N+1):
            V[j] = max(K - S[j], 0)  # Put payoff at maturity
    
    # Backward induction to calculate option value at each node
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            S[j] = S[j] * u / d  # Update the stock price at each node
            continuation_value = np.exp(-r * dt) * (p * V[j+1] + (1 - p) * V[j])
            
            if option_type == "call" and dividend_yield > 0:
                # Early exercise is possible for American call options with dividends
                V[j] = max(S[j] - K, continuation_value)
            elif option_type == "put":
                # Early exercise is possible for American put options
                V[j] = max(K - S[j], continuation_value)
            else:
                # For calls without dividends, use the continuation value only
                V[j] = continuation_value
    
    return V[0]  # Option value at root node















#def monte_carlo_american_option(S0, K, T, r, sigma, N, M, option_type="call"):
def American_Monte_Carlo(S0, K, T, r, sigma, N, M, option_type="call"):
    """
    Monte Carlo simulation to price American call/put options using the Least Squares Monte Carlo (LSMC) method.

    Parameters:
    - S0: Current stock price
    - K: Strike price
    - T: Time to expiration (in years)
    - r: Risk-free interest rate
    - sigma: Volatility (annualized)
    - N: Number of time steps
    - M: Number of Monte Carlo paths
    - option_type: "call" or "put"
    
    Returns:
    - Estimated price of the American option
    """
    # Time step size
    dt = T / N
    # Discount factor
    discount = np.exp(-r * dt)
    
    # Generate random paths for stock price using Geometric Brownian Motion
    Z = np.random.normal(size=(M, N))
    stock_paths = np.zeros((M, N + 1))
    stock_paths[:, 0] = S0

    for t in range(1, N + 1):
        stock_paths[:, t] = stock_paths[:, t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
    
    # Payoff at maturity (final time step)
    if option_type == "call":
        payoff = np.maximum(stock_paths[:, -1] - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - stock_paths[:, -1], 0)

    # Store the option values at the final time step
    option_values = payoff

    # Backward induction using Least Squares Monte Carlo (LSMC)
    for t in range(N-1, 0, -1):
        # Discount the option values to time t
        option_values *= discount
        
        # Get the stock prices at time t
        stock_price_at_t = stock_paths[:, t]

        # Select only the paths where the option is in the money
        if option_type == "call":
            in_the_money = stock_price_at_t > K
        elif option_type == "put":
            in_the_money = stock_price_at_t < K
        
        # Fit the continuation value using least squares regression
        X = stock_price_at_t[in_the_money]
        Y = option_values[in_the_money]

        # If there are paths in the money, continue
        if len(X) > 0:
            # Use polynomial regression (e.g., second degree)
            poly = np.polyfit(X, Y, 2)
            continuation_value = np.polyval(poly, stock_price_at_t[in_the_money])
            
            # Calculate the immediate exercise value
            if option_type == "call":
                immediate_exercise_value = stock_price_at_t[in_the_money] - K
            elif option_type == "put":
                immediate_exercise_value = K - stock_price_at_t[in_the_money]
            
            # Early exercise decision: take the maximum of immediate exercise and continuation value
            option_values[in_the_money] = np.maximum(immediate_exercise_value, continuation_value)

    # Discount the option values to time 0
    option_values *= discount
    
    # Return the estimated option price as the average of the option values at time 0
    option_price = np.mean(option_values)
    return option_price



