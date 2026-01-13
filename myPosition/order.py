class Order:
    def __init__(self, ticker, quantity, price, filled_date="", currency="USD"):
        """
        Initialize a position with a ticker, quantity, and price.
        Args:
            ticker:                 string - ticker symbol
            quantity:               float  - number of shares (can be decimal, e.g., 1.2, 1.3)
            price:                  float  - filled price per share
            filled_datetime(Opt):   string - (e.g., "2024-01-15")
        """
        self.ticker = ticker
        self.quantity = quantity
        self.price = price
        self.filled_date = filled_date
        self.currency = currency
    
    def get_filled_date(self) -> str:
        return self.filled_date

    def total_value(self) -> float:
        return self.quantity * self.price

    def __str__(self):
        return f"{self.ticker}: {self.quantity} shares at ${self.price} {self.currency} at {self.filled_date}"

    def __repr__(self):
        return f"Order(ticker={self.ticker}, quantity={self.quantity}, price={self.price}, currency={self.currency}, filled_date={self.filled_date})"
