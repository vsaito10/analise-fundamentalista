"""
Calculadora de Beta de uma ação.

Calcula o beta de um ativo em relação a um índice de mercado, via regressão
linear (OLS) dos retornos do ativo contra os retornos do índice, reutilizando a
função `calculate_beta` de `funcoes_eua`.

Exemplos de uso (linha de comando):
    python beta_calculator.py AAPL
    python beta_calculator.py AAPL --index ^GSPC --period 5y --interval 1mo
    python beta_calculator.py PETR4.SA --index ^BVSP --period 2y --interval 1wk
"""

import argparse

import funcoes_eua as f_eua


def calcular_beta(
    stock: str,
    index: str = "^GSPC",
    period: str = "5y",
    interval: str = "1mo",
) -> dict:
    """
    Calcula o beta de uma ação e estatísticas da regressão.

    Parameters
    ----------
    stock: str
        Ticker da ação (ex.: 'AAPL', 'PETR4.SA').
    index: str
        Ticker do índice de mercado (default '^GSPC' = S&P 500).
    period: str
        Período do histórico de preços (ex.: '1y', '5y', 'max').
    interval: str
        Intervalo dos preços (ex.: '1d', '1wk', '1mo').

    Returns
    -------
    dict
        Dicionário com beta, alfa, R², p-valor do beta e nº de observações.
    """
    beta, ols = f_eua.calculate_beta(
        index=index,
        stock=stock,
        period=period,
        interval=interval,
        just_beta=False,
    )

    return {
        "stock": stock,
        "index": index,
        "period": period,
        "interval": interval,
        "beta": float(beta),
        "alpha": float(ols.params[0]),
        "r_squared": float(ols.rsquared),
        "beta_p_value": float(ols.pvalues[1]),
        "n_obs": int(ols.nobs),
    }


def interpretar_beta(beta: float) -> str:
    """Retorna uma descrição textual do beta."""
    if beta < 0:
        return "move-se na direção oposta ao mercado"
    if beta < 1:
        return "menos volátil que o mercado (defensiva)"
    if beta > 1:
        return "mais volátil que o mercado (agressiva)"
    return "acompanha o mercado"


def imprimir_resultado(resultado: dict) -> None:
    """Imprime o resultado do cálculo de beta de forma legível."""
    print("=" * 60)
    print(f"Beta de {resultado['stock']} vs {resultado['index']}")
    print("=" * 60)
    print(f"Período / intervalo : {resultado['period']} / {resultado['interval']}")
    print(f"Observações         : {resultado['n_obs']}")
    print(f"Beta                : {resultado['beta']:.4f}  "
          f"({interpretar_beta(resultado['beta'])})")
    print(f"Alfa                : {resultado['alpha']:.6f}")
    print(f"R²                  : {resultado['r_squared']:.4f}")
    print(f"P-valor (beta)      : {resultado['beta_p_value']:.4g}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcula o beta de uma ação em relação a um índice de mercado.",
    )
    parser.add_argument("stock", help="Ticker da ação (ex.: AAPL, PETR4.SA).")
    parser.add_argument(
        "--index",
        default="^GSPC",
        help="Ticker do índice de mercado (default: ^GSPC = S&P 500).",
    )
    parser.add_argument(
        "--period",
        default="5y",
        help="Período do histórico (ex.: 1y, 5y, max). Default: 5y.",
    )
    parser.add_argument(
        "--interval",
        default="1mo",
        help="Intervalo dos preços (ex.: 1d, 1wk, 1mo). Default: 1mo.",
    )
    args = parser.parse_args()

    resultado = calcular_beta(
        stock=args.stock,
        index=args.index,
        period=args.period,
        interval=args.interval,
    )
    imprimir_resultado(resultado)


if __name__ == "__main__":
    main()
