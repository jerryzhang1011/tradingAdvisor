from order import Order

# Order(ticker, quantity, price, filled_datetime=None)
myOrders = [
    Order("AAPL", 1.1749, 38.3, "2025-11-03", "CAD"),
    Order("MSFT", 1.2149, 37.04, "2025-11-03", "CAD"),
    Order("NEE", 0.4604, 27.15, "2025-11-05", "CAD"),
    Order("NVDA", 0.8924, 44.82, "2025-11-06", "CAD"),
    Order("GOOG", 0.888, 45.05, "2025-11-07", "CAD"),
    Order("TSLA", 2, 426, "2025-11-07", "USD"),
]

