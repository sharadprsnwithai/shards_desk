"""sector_momentum_desk/stock_scorer.py — stub (replaced in Task 4)"""
from datetime import datetime


class StockScorer:
    def score_and_select(self, sector: str, bias: str, as_of: datetime):
        return []

    def _fetch_stock_data(self, yf_ticker: str, as_of: datetime):
        return (0.0, 1)

    def _fetch_levels(self, yf_ticker: str, as_of: datetime):
        return (0.0, 0.0, 0.0, 0.0)
