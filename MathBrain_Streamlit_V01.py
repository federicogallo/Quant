import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from PIL import Image

import MathBrain_Options_Traditional as mb
import MathBrain_Options_PINN as mbPINN

import torch
import torch.nn as nn
import torch.optim as optim











# Set up the Streamlit page and sidebar
st.sidebar.title("Interactive Book | Derivatives Pricing")
st.sidebar.write("")
page = st.sidebar.radio(
    "Select option:",
    ["Pioneering the Future in Financial Derivatives Pricing", 
     "European Options Pricing Models", 
     "American Options Pricing Models",
     "Annex - Black-Scholes Equation"]
)





















if page == "Pioneering the Future in Financial Derivatives Pricing":
    # Title
    st.title("Pioneering the Future in Financial Derivatives Pricing")

    # Load the image
    image = Image.open("5.png")

    # Display the image
    st.image(image, caption="Image1", use_column_width=True)

    # Introduction content
    st.markdown("""
                
    **Experience the transformative power of Physics-Informed Neural Networks (PINNs)** in pricing intricate derivatives. By uniting deep learning with advanced mathematical models, PINNs slash valuation times for exotic options and empower traders to detect mispricings, seize arbitrage opportunities, and maximize profits in today’s fast-paced markets. Dive in to discover how harnessing PINNs can supercharge your trading strategies and open up new avenues for wealth creation.
    


    ### Transforming Complex Derivatives Valuation

    In today’s rapidly evolving derivatives market, Physics-Informed Neural Networks (PINNs) are revolutionizing option pricing. By embedding differential equations directly into the training of neural networks, PINNs offer a robust methodology to tackle complex financial models with exceptional speed and precision.

    ### Real-Time Arbitrage and Enhanced Trading Strategies

    Traditional numerical approaches—such as finite difference methods, Monte Carlo simulations, and binomial trees—often fall short when valuing exotic options. PINNs, on the other hand, excel in pricing multifaceted instruments like basket options, barrier options, and Asian options. Their ability to capture higher-dimensional dynamics and intricate path dependencies allows for instantaneous valuations, enabling traders to react swiftly to market fluctuations.

    ### A Dual-Faceted Approach to Market Leadership

    Our interactive platform is built on a comprehensive research strategy:
    - **Benchmarking Excellence:** We rigorously compare PINN-based valuations with classical methods—including analytic solutions, finite difference techniques, Monte Carlo simulations, and binomial trees—to validate their accuracy.
    - **Identifying Competitive Edges:** By pinpointing cases where PINNs clearly outperform traditional methods, we uncover trading opportunities that capitalize on rapid mispricing detection and dynamic portfolio optimization.

    ### Empowering Options Traders in a Complex Market

    Designed specifically for options traders and market professionals, this exploration aims to unlock the transformative potential of PINNs. By continuously refining our portfolio of complex derivatives, we offer innovative tools that enable faster, more informed trading decisions and open new avenues for capturing market inefficiencies.
    """)





























if page == "European Options Pricing Models":
    # Title
    st.title("European Options Pricing Models")



    # User Inputs for Parameters
    S0 = st.number_input("Current Price of Underlying Asset (S₀)", min_value=0.0, step=0.1, value=100.0)
    K = st.number_input("Strike Price (K)", min_value=0.0, step=0.1, value=100.0)
    T = st.number_input("Time to Expiration (T) in years", min_value=0.0, step=0.01, value=1.0)
    r = st.number_input("Risk-Free Interest Rate (r) in decimal (e.g., 0.05 for 5%)", min_value=0.0, max_value=1.0, step=0.01, value=0.05)
    sigma = st.number_input("Volatility (σ) in decimal (e.g., 0.2 for 20%)", min_value=0.0, max_value=1.0, step=0.01, value=0.2)









    # Display the calculated Call and Put Option Prices for all methods
    st.write("## Calculated Option Prices:")


    # Calculate Call and Put prices using Black_Schole_Formula
    call_price_bs = mb.European_Black_Scholes_Formula(S0, K, T, r, sigma, option_type="call")
    put_price_bs = mb.European_Black_Scholes_Formula(S0, K, T, r, sigma, option_type="put")

    st.write("### Black-Scholes Formula:")
    st.write(f"Call Option Price (C) = ${call_price_bs:.2f}")
    st.write(f"Put Option Price (P) = ${put_price_bs:.2f}")





    # Crank-Nicolson parameters
    S_max = 200  # Maximum stock price in the grid
    M = 100  # Number of stock price steps
    N = 100  # Number of time steps

    # Calculate Call and Put prices using Crank-Nicolson Method
    call_price_cn = mb.European_Crank_Nicolson(S0, K, T, r, sigma, S_max, M, N, option_type="call")
    put_price_cn = mb.European_Crank_Nicolson(S0, K, T, r, sigma, S_max, M, N, option_type="put")

    st.write("### Crank-Nicolson Method:")
    st.write(f"Call Option Price (C) = ${call_price_cn:.2f}")
    st.write(f"Put Option Price (P) = ${put_price_cn:.2f}")





    # Monte Carlo Simulation parameters
    num_simulations = 10000

    # Calculate Call and Put prices using Monte Carlo Simulation
    call_price_mc = mb.European_Monte_Carlo(S0, K, T, r, sigma, num_simulations, option_type="call")
    put_price_mc = mb.European_Monte_Carlo(S0, K, T, r, sigma, num_simulations, option_type="put")

    st.write("### Monte Carlo Simulation:")
    st.write(f"Call Option Price (C) = ${call_price_mc:.2f}")
    st.write(f"Put Option Price (P) = ${put_price_mc:.2f}")




    # Calculate Call and Put prices using PINNs

    # 2) Load the model (if already trained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mbPINN.make_nn() #.to(device)
    model.load_state_dict(torch.load("pinn_model.pth", map_location=device))
    model.eval()

    # 3) Predict call/put prices at "today" (t=T).
    call_price_PINN, put_price_PINN = mbPINN.predict_option_prices(model, S0=S0, K_val=K,
                                                T_val=T, r_val=r, sigma_val=sigma)
    #print("PINN Call Price:", call_price)
    #print("PINN Put Price: ", put_price)

    st.write("### PINN Inference:")
    st.write(f"Call Option Price (C) = ${call_price_PINN:.2f}")
    st.write(f"Put Option Price (P) = ${put_price_PINN:.2f}")











    




    st.markdown("---")  # Separator line for visual distinction



     # Create a button labeled "Theory"
    if st.button("Theory"):
        

        st.header("European Options Theory")






        # Definition
        st.write("""
        A **European option** is a type of financial derivative that gives the holder the **right, but not the obligation**, 
        to buy (in the case of a **call option**) or sell (in the case of a **put option**) an underlying asset at a specified price (called the **strike price**) on a specific future date (called the **expiration date** or **maturity date**).
        Unlike an **American option**, which can be exercised at any time before the expiration date, a European option can only be exercised **on the expiration date**.
        """)

        # Key Features of European Options
        st.subheader("Key Features of European Options:")
        st.write("""
        1. **Exercise Date**: Can only be exercised at the expiration date, not before.
        2. **Underlying Asset**: Can be any tradable asset such as a stock, index, commodity, or currency.
        3. **Call Option**: Gives the holder the right to **buy** the underlying asset at the strike price.
        4. **Put Option**: Gives the holder the right to **sell** the underlying asset at the strike price.
        5. **Strike Price (K)**: The predetermined price at which the asset can be bought (for a call) or sold (for a put).
        6. **Expiration Date (T)**: The specific date on which the option can be exercised.
        """)

        # Payoff formulas
        st.subheader("Payoff Formulas:")
        st.write("""
        - **Call Option Payoff**:
        """)
        st.latex(r"\text{Payoff} = \max(S_T - K, 0)")
        st.write("""
        where \( S_T \) is the asset's price at expiration and \( K \) is the strike price.
        """)
        st.write("""
        - **Put Option Payoff**:
        """)
        st.latex(r"\text{Payoff} = \max(K - S_T, 0)")

        # Explanation
        st.write("""
        European options are typically easier to price mathematically compared to American options because they have a single exercise date, 
        allowing closed-form solutions such as the **Black-Scholes formula** to be applied.
        """)







        # Theoretical Annex on the Crank-Nicolson Method
        st.markdown("---")  # Separator line for visual distinction
        st.header("Theoretical Annex 1: Black-Scholes Formula")


        # Call Option
        st.write("**Call Option**:")
        st.latex(r"C = S_0 N(d_1) - K e^{-rT} N(d_2)")

        # Put Option
        st.write("**Put Option**:")
        st.latex(r"P = K e^{-rT} N(-d_2) - S_0 N(-d_1)")

        # d1 and d2
        st.write("Where:")
        st.latex(r"d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{1}{2} \sigma^2\right) T}{\sigma \sqrt{T}}")
        st.latex(r"d_2 = d_1 - \sigma \sqrt{T}")

        # Parameters
        st.subheader("2. Parameters:")
        st.write(r"$S_0$: Current price of the underlying asset.")
        st.write(r"$K$: Strike price of the option.")
        st.write(r"$T$: Time to expiration (in years).")
        st.write(r"$r$: Risk-free interest rate (annualized).")
        st.write(r"$\sigma$: Volatility of the underlying asset (annualized).")
        st.write(r"$N(\cdot)$: Cumulative distribution function (CDF) of the standard normal distribution.")
        st.write(r"$C$: Price of the call option.")
        st.write(r"$P$: Price of the put option.")



















        # Theoretical Annex on the Crank-Nicolson Method
        st.markdown("---")  # Separator line for visual distinction
        st.header("Theoretical Annex 2: Crank-Nicolson Method")

        st.write("""
        The **Crank-Nicolson Method** is a popular finite difference technique used to numerically solve partial differential equations (PDEs). It is particularly effective for parabolic PDEs like the **Black-Scholes Equation**, which is used in financial mathematics to model the price of derivatives such as European options.
        """)

        st.subheader("1. Background and Overview")
        st.write("""
        The Crank-Nicolson method is a **semi-implicit** method, which combines the Forward Euler (explicit) method and the Backward Euler (implicit) method. It is unconditionally stable, meaning that it can provide accurate solutions even with larger time steps, as long as the spatial grid resolution is chosen appropriately. The method is known for its **second-order accuracy** in both space and time.
        """)

        st.subheader("2. Derivation of the Method")
        st.write("""
        For a general parabolic PDE of the form:
        """)
        st.latex(r"""
        \frac{\partial V}{\partial t} = \alpha \frac{\partial^2 V}{\partial S^2} + \beta \frac{\partial V}{\partial S} + \gamma V,
        """)
        st.write("""
        the Crank-Nicolson method approximates the time derivative at the midpoint between the current time step \( t \) and the next time step \( t + \Delta t \). The idea is to take the average of the explicit and implicit discretizations, which leads to:
        """)
        st.latex(r"""
        \frac{V_{i}^{n+1} - V_{i}^{n}}{\Delta t} = \frac{1}{2} \left[ \alpha \frac{V_{i+1}^{n} - 2V_{i}^{n} + V_{i-1}^{n}}{(\Delta S)^2} + \beta \frac{V_{i+1}^{n} - V_{i-1}^{n}}{2 \Delta S} + \gamma V_{i}^{n} \right] + 
        \frac{1}{2} \left[ \alpha \frac{V_{i+1}^{n+1} - 2V_{i}^{n+1} + V_{i-1}^{n+1}}{(\Delta S)^2} + \beta \frac{V_{i+1}^{n+1} - V_{i-1}^{n+1}}{2 \Delta S} + \gamma V_{i}^{n+1} \right].
        """)
        st.write("""
        This equation forms a system of linear equations, which can be solved at each time step to find the option prices at all spatial grid points. The coefficients \( \alpha \), \( \beta \), and \( \gamma \) depend on the specific PDE being solved. In the case of the **Black-Scholes Equation**, these coefficients are given by:
        """)
        st.latex(r"""
        \alpha = \frac{1}{2} \sigma^2 S^2, \quad \beta = r S, \quad \gamma = -r.
        """)

        st.subheader("3. Application to the Black-Scholes Equation")
        st.write("""
        When applying the Crank-Nicolson method to the Black-Scholes equation, we discretize both the stock price \( S \) and time \( t \) dimensions. The resulting finite difference scheme leads to two tridiagonal matrices, which can be solved efficiently using numerical linear algebra techniques. The option payoff is used as the terminal condition, and the boundary conditions are set based on the specific type of option (e.g., call or put).
        """)

        st.write("""
        The Crank-Nicolson method is a powerful technique because it balances accuracy and stability. It is widely used in financial applications due to its reliability and efficiency in solving PDEs like the Black-Scholes equation for European options.
        """)

        st.subheader("4. Advantages of the Crank-Nicolson Method")
        st.write("""
        - **Second-Order Accuracy**: The method is accurate to the second order in both time and space.
        - **Stability**: It is unconditionally stable for the Black-Scholes equation, meaning it does not require small time steps to maintain stability.
        - **Versatility**: The method can be applied to various types of financial derivatives beyond European options, with appropriate modifications for boundary conditions.
        """)

        st.write("The Crank-Nicolson method, due to its blend of accuracy, stability, and computational efficiency, is a standard technique for solving the **Black-Scholes Equation** numerically.")








        # Theoretical Annex on the Monte Carlo Simulation
        st.markdown("---")  # Separator line for visual distinction
        st.header("Theoretical Annex 3: Monte Carlo Simulation for Option Pricing")

        st.write("""
        The **Monte Carlo Simulation** is a widely used numerical technique to estimate the price of European call and put options. This method relies on simulating the random paths of the underlying asset’s price based on a **stochastic process**, specifically the Geometric Brownian Motion (GBM) model.
        """)

        st.subheader("1. Geometric Brownian Motion Model")
        st.write("""
        The GBM model assumes that the stock price follows a stochastic differential equation (SDE) of the form:
        """)
        st.latex(r"""
        dS_t = r S_t dt + \sigma S_t dW_t,
        """)
        st.write("""
        where:
        - \( S_t \) is the stock price at time \( t \),
        - \( r \) is the risk-free interest rate,
        - \( \sigma \) is the volatility of the stock’s returns,
        - \( dW_t \) is a Wiener process (a random variable representing Brownian motion).
        """)

        st.subheader("2. Simulating Stock Price Paths")
        st.write("""
        The GBM model leads to a discretized version of the stock price at a future time \( T \) given by:
        """)
        st.latex(r"""
        S_T = S_0 \exp \left\{ \left( r - \frac{\sigma^2}{2} \right) T + \sigma \sqrt{T} Z \right\},
        """)
        st.write("""
        where \( Z \) is a random variable drawn from a standard normal distribution (\( Z \sim N(0, 1) \)).
        """)

        st.subheader("3. Option Payoff and Discounting")
        st.write("""
        The option payoff at maturity depends on whether it is a call or put option:
        - For a **call option**, the payoff is \( \max(S_T - K, 0) \).
        - For a **put option**, the payoff is \( \max(K - S_T, 0) \).
        The average of the discounted payoffs over all simulated paths gives the present value of the option:
        """)
        st.latex(r"""
        \text{Option Price} = e^{-r T} \cdot \frac{1}{N} \sum_{i=1}^{N} \text{Payoff}_i,
        """)
        st.write("""
        where \( N \) is the number of simulated paths.
        """)

        st.subheader("4. Advantages of Monte Carlo Simulation")
        st.write("""
        - **Flexibility**: Monte Carlo simulations can handle complex payoffs and multiple sources of uncertainty.
        - **Scalability**: This method can be scaled to handle high-dimensional problems and can be easily parallelized.
        - **Intuitive Approach**: The simulation is conceptually straightforward and can be applied to a wide range of derivative pricing problems.
        """)

        st.write("Monte Carlo simulations are a powerful tool for estimating the price of European options, particularly when the analytical solutions are difficult to obtain or do not exist.")
































if page == "American Options Pricing Models":

    # Title of the section
    st.title("American Options Pricing Models")






    # Integrate this with the existing user input setup
    S0 = st.number_input("Current Price of Underlying Asset (S₀)", min_value=0.0, step=0.1, value=100.0)
    K = st.number_input("Strike Price (K)", min_value=0.0, step=0.1, value=100.0)
    T = st.number_input("Time to Expiration (T) in years", min_value=0.0, step=0.01, value=1.0)
    r = st.number_input("Risk-Free Interest Rate (r) in decimal (e.g., 0.05 for 5%)", min_value=0.0, max_value=1.0, step=0.01, value=0.05)
    sigma = st.number_input("Volatility (σ) in decimal (e.g., 0.2 for 20%)", min_value=0.0, max_value=1.0, step=0.01, value=0.2)

    # Grid parameters
    S_max = 200  # Maximum stock price in the grid
    M = 100  # Number of stock price steps
    #N = 100  # Number of time steps
    N = st.number_input("Number of Time Steps (N)", min_value=1, step=1, value=100)
    dividend_yield = st.number_input("Dividend Yield (if any)", min_value=0.0, max_value=1.0, step=0.01, value=0.0)
    MC = st.number_input("Number of Monte Carlo Paths (M)", min_value=1000, step=1000, value=10000)




    st.write("## Calculated Option Prices:")

    # Calculate American option prices using Crank-Nicolson method for both call and put options
    call_price_american_cn = mb.American_Crank_Nicolson(S0, K, T, r, sigma, S_max, M, N, option_type="call")
    put_price_american_cn = mb.American_Crank_Nicolson(S0, K, T, r, sigma, S_max, M, N, option_type="put")

    # Display the calculated American call and put option prices
    st.write("### Crank-Nicolson Method:")
    st.write(f"Call Option Price (C) = ${call_price_american_cn:.2f}")
    st.write(f"Put Option Price (P) = ${put_price_american_cn:.2f}")



    # Calculate American option prices using Binomial Tree method for both call and put options
    call_price_american_binomial = mb.American_Binomial_Tree(S0, K, T, r, sigma, N, option_type="call", dividend_yield=dividend_yield)
    put_price_american_binomial = mb.American_Binomial_Tree(S0, K, T, r, sigma, N, option_type="put", dividend_yield=dividend_yield)

    # Display the calculated American call and put option prices
    st.write("### Binomial Tree Method:")
    st.write(f"Call Option Price (C) = ${call_price_american_binomial:.2f}")
    st.write(f"Put Option Price (P) = ${put_price_american_binomial:.2f}")




    # Calculate American option prices for both call and put options
    call_price_american_mc = mb.American_Monte_Carlo(S0, K, T, r, sigma, N, MC, option_type="call")
    put_price_american_mc = mb.American_Monte_Carlo(S0, K, T, r, sigma, N, MC, option_type="put")

    # Display the calculated American call and put option prices
    st.write("### Monte Carlo Simulation:")
    st.write(f"Call Option Price (C) = ${call_price_american_mc:.2f}")
    st.write(f"Put Option Price (P) = ${put_price_american_mc:.2f}")









    # Theoretical Annex 2: Crank-Nicolson Method for American Options
    st.markdown("---")  # Separator line for visual distinction






     # Create a button labeled "Theory"
    if st.button("Theory"):
        

        st.header("American Options Theory")










        st.header("American Options Overview")















        # Definition
        st.write("""
        An **American option** is a type of financial derivative that gives the holder the **right, but not the obligation**, 
        to buy (in the case of a **call option**) or sell (in the case of a **put option**) an underlying asset at a specified price (called the **strike price**) at **any time before or on the expiration date**. 
        Unlike a **European option**, which can only be exercised on the expiration date, American options provide more flexibility since they can be exercised at **any point up to and including the expiration date**.
        """)

        # Key Features of American Options
        st.subheader("Key Features of American Options:")
        st.write("""
        1. **Exercise Date**: Can be exercised **at any time** before or on the expiration date.
        2. **Underlying Asset**: Can be any tradable asset such as a stock, index, commodity, or currency.
        3. **Call Option**: Gives the holder the right to **buy** the underlying asset at the strike price.
        4. **Put Option**: Gives the holder the right to **sell** the underlying asset at the strike price.
        5. **Strike Price (K)**: The predetermined price at which the asset can be bought (for a call) or sold (for a put).
        6. **Expiration Date (T)**: The last date on which the option can be exercised. 
        7. **Flexibility**: American options offer the holder more flexibility compared to European options, allowing them to capitalize on favorable movements in the underlying asset price at any time.
        """)

        # Payoff formulas
        st.subheader("Payoff Formulas:")
        st.write("""
        - **Call Option Payoff**:
        """)
        st.latex(r"\text{Payoff} = \max(S_T - K, 0)")
        st.write("""
        where \( S_T \) is the asset's price at expiration and \( K \) is the strike price.
        """)
        st.write("""
        - **Put Option Payoff**:
        """)
        st.latex(r"\text{Payoff} = \max(K - S_T, 0)")

        # Explanation
        st.write("""
        Unlike European options, the flexibility of American options makes them more complex to price because the holder can choose to exercise them at any time.
        This flexibility often results in higher premiums for American options compared to European options.
        Because of this complexity, **closed-form pricing models** like the Black-Scholes formula cannot be applied, and **numerical methods** such as the **binomial tree model** or **finite difference methods** are typically used to price American options.
        """)







































        # Theoretical Annex 2: Crank-Nicolson Method for American Options
        st.markdown("---")  # Separator line for visual distinction
        st.header("Theoretical Annex 1: Crank-Nicolson Method for American Options")

        st.write("""
        The **Crank-Nicolson Method** is a powerful finite difference method used to numerically solve the **Black-Scholes Partial Differential Equation (PDE)**. 
        For **American options**, the challenge lies in the possibility of **early exercise**, which means the option holder can choose to exercise the option at any point before the expiration date. 
        This makes American options more complex to price than European options, which can only be exercised at expiration.
        """)

        st.subheader("1. Early Exercise Condition")
        st.write("""
        To account for the early exercise feature in American options, the Crank-Nicolson method imposes a condition at each time step that checks whether exercising the option early is more valuable than holding it. 
        This condition is defined as follows:
        """)
        st.latex(r"""
        V(S, t) = \max(V_{\text{continue}}(S, t), V_{\text{exercise}}(S, t)),
        """)
        st.write("""
        Where:
        - \( V_{\text{continue}}(S, t) \) is the value of the option if it is held (continuation value),
        - \( V_{\text{exercise}}(S, t) \) is the value of the option if exercised immediately (payoff from exercise).
        """)

        st.subheader("2. Application of the Crank-Nicolson Method")
        st.write("""
        For American options, the Crank-Nicolson method follows the same steps as for European options, but with the addition of the **early exercise condition** at each step. 
        The PDE is discretized in both the **stock price** and **time** dimensions. At each time step, the value of the option is computed by solving the discretized system of equations, and the early exercise condition is applied.
        """)

        st.latex(r"""
        \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0
        """)

        st.write("""
        At each time step, the Crank-Nicolson scheme solves this equation to compute the continuation value. Then, the option price at each grid point is updated using:
        """)
        st.latex(r"""
        V(S, t) = \max\left( V_{\text{continue}}(S, t), \text{Payoff from early exercise} \right)
        """)

        st.subheader("3. Payoff Functions")
        st.write("""
        The payoff functions for American options are similar to those for European options, but with the early exercise condition applied:
        """)
        st.write("""
        - **Call Option Payoff**:
        """)
        st.latex(r"""
        \text{Payoff} = \max(S - K, 0)
        """)
        st.write("""
        - **Put Option Payoff**:
        """)
        st.latex(r"""
        \text{Payoff} = \max(K - S, 0)
        """)

        st.subheader("4. Advantages of the Crank-Nicolson Method for American Options")
        st.write("""
        - **Second-Order Accuracy**: The method is accurate to the second order in both time and space.
        - **Stability**: The method is unconditionally stable, making it suitable for pricing American options, even with large time steps.
        - **Ability to Handle Early Exercise**: The early exercise condition is naturally incorporated into the Crank-Nicolson method, making it well-suited for American options.
        """)

        st.write("""
        The Crank-Nicolson method, when properly adapted to handle early exercise, is a robust and widely used numerical technique for pricing **American options**.
        """)







        # Theoretical Annex 2: Binomial Tree Method for American Options
        st.markdown("---")  # Separator line for visual distinction
        st.header("Theoretical Annex 2: Binomial Tree Method for American Options")

        st.write("""
        The **Binomial Tree Method** is a popular and intuitive numerical approach for pricing options, including **American options**, 
        which allow early exercise at any point before or on the expiration date. 
        This method works by discretizing time into small intervals and constructing a tree that represents all possible price paths the underlying asset can take. 
        The option is then priced by working backward from the expiration date to the present, taking into account both the possibility of early exercise and the option's payoff at each node.
        """)

        st.subheader("Key Concepts of the Binomial Tree Method:")
        st.write("""
        - **Asset Price Tree**: The binomial tree method models the underlying asset's price evolution over time using a binomial distribution. 
        At each step in time, the asset price can either move **up** or **down** by a certain factor, creating a lattice of possible asset prices.
        """)

        st.write("""
        - **Risk-Neutral Valuation**: The method assumes a risk-neutral world, meaning that investors expect to earn the risk-free rate of return, regardless of the asset's risk.
        """)

        st.write("""
        - **Backward Induction**: The option price is calculated by working backward through the tree, starting from the terminal payoff at expiration and discounting the option value at each earlier step.
        """)

        st.subheader("Steps in the Binomial Tree Method for American Options:")
        st.write("1. **Define the Tree Parameters**")
        st.write("""
        The tree consists of time steps where each step represents a discrete moment in time. 
        The price of the underlying asset can either move up or down at each step:
        - Let \( N \) be the number of time steps.
        - Define the time step \( \Delta t = \\frac{T}{N} \), where \( T \) is the time to expiration.
        The asset price can move up or down based on two factors:
        """)
        st.write("- **Up factor**:")
        st.latex(r"u = e^{\sigma \sqrt{\Delta t}}")
        st.write("- **Down factor**:")
        st.latex(r"d = e^{-\sigma \sqrt{\Delta t}}")

        st.write("2. **Risk-Neutral Probability**")
        st.write("""
        To ensure that the binomial tree is consistent with risk-neutral pricing, the probability of an upward movement is:
        """)
        st.latex(r"p = \frac{e^{r \Delta t} - d}{u - d}")
        st.write("""
        Where:
        - \( r \) is the risk-free interest rate,
        - \( u \) and \( d \) are the up and down factors.
        The probability of a downward movement is \( 1 - p \).
        """)

        st.write("3. **Build the Asset Price Tree**")
        st.write("""
        At each node \( i,j \) in the binomial tree, where \( i \) represents the time step and \( j \) represents the node index at that time step, the asset price at that node is given by:
        """)
        st.latex(r"S_{i,j} = S_0 u^j d^{i-j}")
        st.write("""
        Where \( S_0 \) is the initial price of the underlying asset.
        """)

        st.write("4. **Option Payoff at Expiration**")
        st.write("""
        At the final time step \( N \), calculate the payoff of the option at each node:
        - **Call Option Payoff**:
        """)
        st.latex(r"V_{N,j} = \max(S_{N,j} - K, 0)")
        st.write("- **Put Option Payoff**:")
        st.latex(r"V_{N,j} = \max(K - S_{N,j}, 0)")

        st.write("5. **Backward Induction**")
        st.write("""
        Starting from the final time step \( N \), calculate the option value at each earlier node by working backward through the tree. 
        The value of the option at each node is determined by the expected value of holding the option (i.e., the discounted value of the option in the next step) and the payoff from early exercise. 
        For American options, you must check whether early exercise is optimal:
        """)
        st.latex(r"V_{i,j} = \max\left( \text{Payoff from early exercise}, e^{-r \Delta t} \left[ p V_{i+1,j+1} + (1 - p) V_{i+1,j} \right] \right)")
        st.write("""
        The value of the option is the maximum of:
        - The value of **immediate exercise** (for a call option, \( \max(S_{i,j} - K, 0) \)),
        - The value of holding the option and letting it move to the next time step.
        """)

        st.write("6. **Result**")
        st.write("""
        The **present value** of the option at the root of the tree \( V_{0,0} \) is the price of the option.
        """)

        st.subheader("Equations Summary:")
        st.write("""
        - **Time step**:
        """)
        st.latex(r"\Delta t = \frac{T}{N}")
        st.write("""
        - **Up and down factors**:
        """)
        st.latex(r"u = e^{\sigma \sqrt{\Delta t}}, \quad d = e^{-\sigma \sqrt{\Delta t}}")
        st.write("""
        - **Risk-neutral probability**:
        """)
        st.latex(r"p = \frac{e^{r \Delta t} - d}{u - d}")
        st.write("""
        - **Option Value at each node**:
        """)
        st.latex(r"V_{i,j} = \max\left( \text{Payoff from early exercise}, e^{-r \Delta t} \left[ p V_{i+1,j+1} + (1 - p) V_{i+1,j} \right] \right)")

        st.subheader("Advantages of the Binomial Tree Method:")
        st.write("""
        - **Flexibility**: Can handle complex option features like early exercise for American options, as well as path-dependent options like barrier options.
        - **Intuitive**: The binomial model is easy to understand and implement, making it one of the most widely used option pricing methods.
        """)

        st.subheader("Disadvantages:")
        st.write("""
        - **Computationally Intensive**: For large numbers of time steps, the binomial tree method becomes computationally expensive.
        - **Accuracy**: The accuracy of the method depends on the number of time steps. More time steps improve accuracy but increase computational cost.
        """)




        # Theoretical Annex 3: Monte Carlo Simulation for American Options
        st.markdown("---")  # Separator line for visual distinction
        st.header("Theoretical Annex 3: Monte Carlo Simulation for American Options")

        st.write("""
        The **Monte Carlo Simulation** is a flexible and powerful numerical technique used to estimate the price of American options. 
        Unlike European options, which can only be exercised at expiration, American options allow the holder to exercise the option at any point before or on the expiration date.
        This feature requires the Monte Carlo method to account for the possibility of early exercise, which adds complexity to the simulation process.
        """)

        st.subheader("1. Geometric Brownian Motion Model")
        st.write("""
        Like European options, the underlying asset's price for American options is modeled using **Geometric Brownian Motion (GBM)**. 
        The GBM model assumes the stock price follows a stochastic differential equation (SDE):
        """)
        st.latex(r"""
        dS_t = r S_t dt + \sigma S_t dW_t,
        """)
        st.write("""
        Where:
        - \( S_t \) is the stock price at time \( t \),
        - \( r \) is the risk-free interest rate,
        - \( \sigma \) is the volatility of the stock’s returns,
        - \( dW_t \) is a Wiener process (representing Brownian motion).
        """)

        st.subheader("2. Simulating Stock Price Paths and Early Exercise")
        st.write("""
        For American options, the key challenge is determining the optimal time to exercise the option. 
        After simulating the stock price paths using the GBM model, we need to evaluate the option value at each point and check if it is optimal to exercise the option early.
        The stock price at maturity \( T \) is simulated as follows:
        """)
        st.latex(r"""
        S_T = S_0 \exp \left\{ \left( r - \frac{\sigma^2}{2} \right) T + \sigma \sqrt{T} Z \right\},
        """)
        st.write("""
        Where \( Z \sim N(0,1) \) is a random variable drawn from a standard normal distribution.

        At each time step along the simulated path, the value of the option is calculated by comparing:
        - The value of **immediate exercise** (for a call option, \( \max(S_t - K, 0) \)),
        - The **continuation value**, which is the discounted expected payoff from holding the option and continuing to the next step.
        """)

        st.subheader("3. Early Exercise Condition and Backward Induction")
        st.write("""
        After simulating all the paths, the **Least Squares Monte Carlo (LSMC)** method is typically used to determine the optimal early exercise strategy. 
        The LSMC method applies backward induction along the simulated paths to estimate the continuation value at each step.

        At each time step, the option holder must decide whether to exercise the option early or to continue holding it based on the maximum value between the two:
        """)
        st.latex(r"""
        V(S_t) = \max(V_{\text{continue}}(S_t), V_{\text{exercise}}(S_t)).
        """)
        st.write("""
        Where:
        - \( V_{\text{continue}}(S_t) \) is the continuation value (holding the option),
        - \( V_{\text{exercise}}(S_t) \) is the immediate exercise payoff.

        For American call options, early exercise is typically optimal when the value of the stock price exceeds a critical threshold. For put options, early exercise is favored when the stock price is below a certain level.
        """)

        st.subheader("4. Discounting and Averaging the Payoff")
        st.write("""
        Once the early exercise strategy is determined, the payoff for each simulated path is calculated by either exercising early or holding until expiration.
        The option's present value is then computed as the **average discounted payoff** across all simulated paths:
        """)
        st.latex(r"""
        \text{Option Price} = e^{-r T} \cdot \frac{1}{N} \sum_{i=1}^{N} \text{Payoff}_i,
        """)
        st.write("""
        Where \( N \) is the number of simulated paths.

        Monte Carlo simulations are especially useful for American options when closed-form solutions like the Black-Scholes formula cannot be applied.
        """)

        st.subheader("5. Differences between European and American Monte Carlo Simulations")
        st.write("""
        The main difference between Monte Carlo simulations for **European** and **American options** lies in the handling of **early exercise**. 
        While European options are only exercised at expiration, American options require the Monte Carlo method to check for optimal exercise at each time step, adding complexity to the simulation.
        """)

        st.write("""
        - **European Options**: Payoff is calculated only at expiration.
        - **American Options**: Early exercise must be considered, making backward induction and regression techniques (like the **Least Squares Monte Carlo** method) necessary to estimate the continuation value.
        """)

        st.subheader("6. Advantages of Monte Carlo Simulation for American Options")
        st.write("""
        - **Flexibility**: Monte Carlo simulations can handle complex early exercise features and multiple sources of uncertainty.
        - **Versatility**: This method can be applied to various types of American options and options with path-dependent features.
        - **Scalability**: Monte Carlo simulations can be parallelized to improve performance, making them suitable for high-dimensional problems.
        """)

        st.write("""
        Monte Carlo simulations provide a flexible and powerful approach to pricing American options, especially when the early exercise feature makes closed-form solutions impractical.
        """)





















if page == "Annex - Black-Scholes Equation":
    # Title
    st.title("Black-Scholes Summary of Equations")

    # Black-Scholes Differential Equation
    st.subheader("1. Black-Scholes Differential Equation:")
    st.latex(r"\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0")

    # Parameters
    st.subheader("2. Parameters:")
    st.write(r"$V$: The price of the derivative (option) as a function of the stock price $S$ and time $t$.")
    st.write(r"$S$: Current price of the underlying asset (stock price).")
    st.write(r"$t$: Current time (in years).")
    st.write(r"$\sigma$: Volatility of the underlying asset (annualized standard deviation of the asset’s returns).")
    st.write(r"$r$: Risk-free interest rate (annualized).")



