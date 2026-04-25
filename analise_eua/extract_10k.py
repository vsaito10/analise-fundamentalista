from pathlib import Path
import sys

import pandas as pd

from funcoes_sec import (
    annual_balance_sheet_from_companyfacts,
    annual_cash_flow_from_companyfacts,
    annual_income_statement_from_companyfacts,
)

SCALE_DIVISOR = 1000000


def _prepare_statement(df: pd.DataFrame) -> pd.DataFrame:
    statement = df.copy()
    statement = statement.dropna(axis=0, how="all")
    statement = statement.dropna(axis=1, how="all")
    statement = statement.sort_index(axis=1)
    statement = statement / SCALE_DIVISOR
    return statement


def get_balance_sheet(ticker: str) -> pd.DataFrame:
    return _prepare_statement(annual_balance_sheet_from_companyfacts(ticker.upper()))


def get_income_statement(ticker: str) -> pd.DataFrame:
    return _prepare_statement(annual_income_statement_from_companyfacts(ticker.upper()))


def get_cash_flow(ticker: str) -> pd.DataFrame:
    return _prepare_statement(annual_cash_flow_from_companyfacts(ticker.upper()))


def build_output_path(ticker: str, output_path: Path | None = None) -> Path:
    if output_path is not None:
        return output_path

    ticker = ticker.lower()
    return Path(__file__).resolve().parent / f"{ticker}_10k_complete.xlsx"


def save_10k(ticker: str, output_path: Path | None = None) -> Path:
    final_output_path = build_output_path(ticker, output_path)
    ticker = ticker.upper()

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_balance_sheet = get_balance_sheet(ticker)
    df_income_statement = get_income_statement(ticker)
    df_cash_flow = get_cash_flow(ticker)

    with pd.ExcelWriter(final_output_path) as writer:
        df_balance_sheet.to_excel(writer, sheet_name="balance_sheet")
        df_income_statement.to_excel(writer, sheet_name="income_statement")
        df_cash_flow.to_excel(writer, sheet_name="cash_flow")

    return final_output_path


if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    saved_path = save_10k(ticker)
    print(saved_path)
