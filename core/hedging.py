from core.pricing import bs_price_delta

def hedge_option_path(S_path, K, sigma, r, dt, option_type='call', position='short', hedge_every_n=1, transaction_cost=0.0):
    steps = len(S_path) - 1
    T = steps * dt
    price0, delta0 = bs_price_delta(S_path[0], K, T, sigma, r, option_type)

    cash = price0 if position == 'short' else -price0
    shares = delta0 if position == 'short' else -delta0
    cash -= shares * S_path[0]

    pnl_series = []

    for i in range(1, steps + 1):
        t_remaining = T - i * dt
        S = S_path[i]

        if i % hedge_every_n == 0 or i == steps:
            price, new_delta = bs_price_delta(S, K, t_remaining, sigma, r, option_type)
            desired_shares = new_delta if position == 'short' else -new_delta
            delta_diff = desired_shares - shares

            if delta_diff != 0:
                cost = delta_diff * S
                tc = abs(cost) * transaction_cost
                cash -= cost + tc
                shares += delta_diff

        portfolio = cash + shares * S

        if t_remaining > 0:
            price, _ = bs_price_delta(S, K, t_remaining, sigma, r, option_type)
            liability = price if position == 'short' else -price
        else:
            payoff = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
            liability = payoff if position == 'short' else -payoff

        pnl_series.append(portfolio - liability)

    return pnl_series[-1], pnl_series
