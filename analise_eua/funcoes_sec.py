from bs4 import BeautifulSoup
import calendar
import logging
import numpy as np
import os
import requests
import pandas as pd


statement_keys_map = {
    "balance_sheet": [
        "balance sheet",
        "balance sheets",
        "statement of financial position",
        "consolidated balance sheets",
        "consolidated balance sheet",
        "consolidated condensed balance sheets",
        "consolidated financial position",
        "consolidated balance sheets - southern",
        "consolidated statements of financial position",
        "consolidated statement of financial position",
        "consolidated statements of financial condition",
        "combined and consolidated balance sheet",
        "condensed consolidated balance sheets",
        "condensed consolidated balance sheets - unaudited",
        "consolidated balance sheets, as of december 31",
        "dow consolidated balance sheets",
        "consolidated balance sheets (unaudited)",
        "condensed consolidated balance sheets (unaudited)"
    ],

    "income_statement": [
        "income statement",
        "income statements",
        "statement of earnings (loss)",
        "statements of consolidated income",
        "consolidated condensed statements of income",
        "consolidated statements of operations",
        "consolidated statement of operations",
        "consolidated statements of earnings",
        "consolidated statement of earnings",
        "consolidated statements of income",
        "consolidated statement of income",
        "consolidated income statements",
        "consolidated income statement",
        "condensed consolidated statements of operations",
        "condensed consolidated statements of earnings",
        "condensed consolidated statements of income",
        "condensed consolidated statements of operations - unaudited",
        "consolidated results of operations",
        "consolidated statements of income (loss)",
        "consolidated statements of income - southern",
        "consolidated statements of operations and comprehensive income",
        "consolidated statements of comprehensive income",
        "condensed consolidated statements of operations (unaudited)"
    ],

    "income_statement_2": [
        "consolidated statements of operations consolidated statements of operations"
    ],

    "cash_flow_statement": [
        "cash flows statement",
        "cash flows statements",
        "statement of cash flows",
        "statements of consolidated cash flows",
        "consolidated condensed statements of cash flows",
        "consolidated statements of cash flows",
        "consolidated statement of cash flows",
        "consolidated statement of cash flow",
        "consolidated cash flows statements",
        "consolidated cash flow statements",
        "condensed consolidated statements of cash flows",
        "condensed consolidated statements of cash flows - unaudited",
        "consolidated statements of cash flows (unaudited)",
        "consolidated statements of cash flows - southern",
        "condensed consolidated statements of cash flows (unaudited)"
    ],

    "cross_holding": [
        # AAPL -> 'Note 6 – Consolidated Financial Statement Details' -> 'Other Non-Current Assets'
        "consolidated financial statement details - other non-current assets (details)",  
        # NVDA -> 'Note 9 - Balance Sheet Components' -> 'Other Assets (Long Term)' -> 'Investments in non-affiliated entities'
        "balance sheet components - other assets (details)",
    ],

    "leases": [
        # AAPL -> 'Note 8 – Leases' -> 'Total lease liabilities'
        "leases - rou assets and lease liabilities (details)",
        "commitments and contingencies - future minimum lease payments under noncancelable operating leases (details)",
        # GOOGL -> 'Note 7. Supplemental Financial Statement Information' -> 'Accrued Expenses and Other Current Liabilities' -> 'Current operating lease liabilities'
        "supplemental financial statement information - accrued expenses and other current liabilities (details)",
        "supplemental financial statement information (accrued expenses and other current liabilities) (details)",
        # MSFT -> 'Note 13 — Leases'
        "supplemental balance sheet information related to leases (detail)",
        # META -> 'Note 8. Leases'
        "leases - schedule of maturities of lease liabilities (details)",
        "leases  - schedule of maturities of lease liabilities (details)",
        "commitments and contingencies (details)",
        # NVDA -> 'Note 17 - Leases'
        "leases - schedule of future minimum lease payments (details)",
        "leases - schedule of future minimum lease obligations (details)",
        "leases - schedule of future minimum payments (details)",
    ],

    "rsu": [
        # AAPL -> 'Note 11 – Share-Based Compensation' -> 'Number of RSUs'
        "share-based compensation - restricted stock unit activity and related information (details)",
        # GOOGL -> 'Note 12. Net Income Per Share' -> tabela que contém as informações dos nº da 'Class A', 'Class B', 'Class C' e 'Restricted stock units'
        "net income per share - schedule of earnings per share (details)",
        "net income per share (details)",
        "net income per share (schedule of earnings per share) (details)",
        # META -> 'Note 4. Earnings per Share'
        "earnings per share - schedule of numerators and denominators of basic and diluted eps computations for common stock (details)"
    ],

    "interest_expense": [
        # AAPL -> 'Note 6 – Consolidated Financial Statement Details' -> 'Other Income/(Expense), Net' -> 'Interest expense'
        "consolidated financial statement details - other income/(expense), net (details)",
        # GOOGL -> 'Other Income (Expense), Net' -> 'Interest expense'
        "supplemental financial statement information - components of other income (expense), net (details)",
        "supplemental financial statement information - schedule of other income (expense), net (details)",
        # MSFT -> 'Other Income (Expense), Net' -> 'Interest expense'
        "components of other income (expense), net (detail)",
        # META -> 'Note 14. Interest and Other Income (Expense), Net'
        "interest and other income (expense), net (details)",
        "interest and other income, net (details)"
    ],

    "current_portion_lease": [
        # GOOGL -> 'Note 6. Debt' -> 'Long-Term Debt' -> 'Current portion of future finance lease payments'
        "debt - long-term debt (details)",
        "debt (long-term debt) (details)",
        # AMZN -> 'Note 4 — Leases'
        "leases - operating and finance lease reconciliation (details)",
        "leases - operating and finance lease liability reconciliation (details)",
        "leases operating and finance lease reconciliation (details)",
    ],

    "current_portion_capital_lease":[
        # AMZN -> 'Note 6 — OTHER LONG-TERM LIABILITIES' -> 'Capital and Finance Leases'
        # Nos anos de 2018 e 2017, o 'current portion lease' está separado em dois ('current portion capital lease' e 'current portion finance lease')
        "other long-term liabilities - long term capital lease obligation (details)"  
    ],

    "current_portion_finance_lease":[
        # AMZN -> 'Note 6 — OTHER LONG-TERM LIABILITIES' -> 'Capital and Finance Leases'
        # Nos anos de 2018 e 2017, o 'current portion lease' está separado em dois ('current portion capital lease' e 'current portion finance lease')
        "other long-term liabilities - long term finance lease obligation (details)"  
    ],

    "current_portion_debt":[
        # AMZN -> 'Note 6 — Debt'
        "debt - long-term debt obligations (details)",
        "long-term debt - long-term debt obligations (details)",
    ],

    "geographic_revenue": [
        # AAPL -> 'Segment Operating Performance'
        "segment information and geographic data - information by reportable segment (details)",
        # NVDA -> 'Note 16 - Segment Information' -> 'Geographic Revenue based upon Customer Billing Location'
        "segment information - revenue and long-lived assets by region (details)"
    ],

    "restructuring_pretax_charges": [
        # META -> 'Note 3. Restructuring'
        "restructuring - narrative (details)"
    ]
}

header = {
  "User-Agent": "vitorsaito95@email.com"
}


def cik_matching_ticker(ticker, headers=header):

    ticker = ticker.upper().replace('.', '')

    ticker_json = requests.get('https://www.sec.gov/files/company_tickers.json', headers=header).json()
    
    for company in ticker_json.values():
        if company['ticker'] == ticker:
            cik = str(company['cik_str']).zfill(10)
            return cik
    
    raise ValueError(f'Ticker {ticker} not found in SEC database')


def get_submission_data_for_ticker(ticker, headers=header, only_filings_df=False):

    cik = cik_matching_ticker(ticker)

    url = f'https://data.sec.gov/submissions/CIK{cik}.json'

    company_json = requests.get(url, headers=header).json()

    if only_filings_df:
        return pd.DataFrame(company_json['filings']['recent'])
    
    return company_json


def get_filtered_filings(ticker, ten_k=True, just_accession_numbers=False, headers=header):

    company_filings_df = get_submission_data_for_ticker(
        ticker, only_filings_df=True, headers=headers
    )
    if ten_k:
        df = company_filings_df[company_filings_df["form"] == "10-K"]
    else:
        df = company_filings_df[company_filings_df["form"] == "10-Q"]
    if just_accession_numbers:
        df = df.set_index("reportDate")
        accession_df = df["accessionNumber"]
        return accession_df
    else:
        return df
    

def get_facts(ticker, headers=header):

    cik = cik_matching_ticker(ticker)

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    company_facts = requests.get(url, headers=headers).json()
    
    return company_facts


def facts_DF(ticker, headers=header):

    facts = get_facts(ticker, headers)
    us_gaap_data = facts["facts"]["us-gaap"]
    df_data = []
    for fact, details in us_gaap_data.items():
        for unit in details["units"]:
            for item in details["units"][unit]:
                row = item.copy()
                row["fact"] = fact
                df_data.append(row)

    df = pd.DataFrame(df_data)
    df["end"] = pd.to_datetime(df["end"])
    df["start"] = pd.to_datetime(df["start"])
    df = df.drop_duplicates(subset=["fact", "end", "val"])
    df.set_index("end", inplace=True)
    labels_dict = {fact: details["label"] for fact, details in us_gaap_data.items()}
    
    return df, labels_dict


def annual_facts(ticker, headers=header):
    accession_nums = get_filtered_filings(
        ticker, ten_k=True, just_accession_numbers=True
    )
    df, label_dict = facts_DF(ticker, headers)
    ten_k = df[df["accn"].isin(accession_nums)]
    ten_k = ten_k[ten_k.index.isin(accession_nums.index)]
    pivot = ten_k.pivot_table(values="val", columns="fact", index="end")
    pivot.rename(columns=label_dict, inplace=True)

    return pivot.T


def quarterly_facts(ticker, headers=header):
    accession_nums = get_filtered_filings(
        ticker, ten_k=False, just_accession_numbers=True
    )
    df, label_dict = facts_DF(ticker, headers)
    ten_q = df[df["accn"].isin(accession_nums)]
    ten_q = ten_q[ten_q.index.isin(accession_nums.index)].reset_index(drop=False)
    ten_q = ten_q.drop_duplicates(subset=["fact", "end"], keep="last")
    pivot = ten_q.pivot_table(values="val", columns="fact", index="end")
    pivot.rename(columns=label_dict, inplace=True)

    return pivot.T


def save_dataframe_to_csv(dataframe, folder_name, ticker, statement_name, frequency):
    directory_path = os.path.join(folder_name, ticker)
    os.makedirs(directory_path, exist_ok=True)
    file_path = os.path.join(directory_path, f"{statement_name}_{frequency}.csv")
    dataframe.to_csv(file_path)
    
    return None


def _get_file_name(report):
    html_file_name_tag = report.find("HtmlFileName")
    xml_file_name_tag = report.find("XmlFileName")

    if html_file_name_tag:

        return html_file_name_tag.text
    
    elif xml_file_name_tag:

        return xml_file_name_tag.text
    
    else:

        return ""


def _is_statement_file(short_name_tag, long_name_tag, file_name):
    return (
        short_name_tag is not None
        and long_name_tag is not None
        and file_name  # Check if file_name is not an empty string
        and "Statement" in long_name_tag.text
        or "Disclosure" in long_name_tag.text
    )


def get_statement_file_names_in_filing_summary(ticker, accession_number, headers=header):
    try:
        session = requests.Session()
        cik = cik_matching_ticker(ticker)
        base_link = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}"
        filing_summary_link = f"{base_link}/FilingSummary.xml"
        filing_summary_response = session.get(
            filing_summary_link, headers=headers
        ).content.decode("utf-8")

        filing_summary_soup = BeautifulSoup(filing_summary_response, "lxml-xml")
        statement_file_names_dict = {}

        for report in filing_summary_soup.find_all("Report"):
            file_name = _get_file_name(report)
            short_name, long_name = report.find("ShortName"), report.find("LongName")

            if _is_statement_file(short_name, long_name, file_name):
                statement_file_names_dict[short_name.text.lower()] = file_name

        return statement_file_names_dict

    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return {}


def get_statement_soup(
    ticker,
    accession_number,
    statement_name,
    headers,
    statement_keys_map,
):
    """
    the statement_name should be one of the following:
    'balance_sheet'
    'income_statement'
    'cash_flow_statement'
    """
    session = requests.Session()

    cik = cik_matching_ticker(ticker)
    base_link = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}"

    statement_file_name_dict = get_statement_file_names_in_filing_summary(
        ticker, accession_number, headers
    )

    statement_link = None
    for possible_key in statement_keys_map.get(statement_name.lower(), []):
        file_name = statement_file_name_dict.get(possible_key.lower())
        if file_name:
            statement_link = f"{base_link}/{file_name}"
            break

    if not statement_link:
        raise ValueError(f"Could not find statement file name for {statement_name}")

    try:
        statement_response = session.get(statement_link, headers=headers)
        statement_response.raise_for_status()  # Check if the request was successful

        if statement_link.endswith(".xml"):
            return BeautifulSoup(
                statement_response.content, "lxml-xml", from_encoding="utf-8"
            )
        else:
            return BeautifulSoup(statement_response.content, "lxml")

    except requests.RequestException as e:
        raise ValueError(f"Error fetching the statement: {e}")
    

def extract_columns_values_and_dates_from_statement(soup):
    """
    Extracts columns, values, and dates from an HTML soup object representing a financial statement.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object of the HTML document.

    Returns:
        tuple: Tuple containing columns, values_set, and date_time_index.
    """
    columns = []
    values_set = []
    date_time_index = get_datetime_index_dates_from_statement(soup)

    for table in soup.find_all("table"):
        unit_multiplier = 1
        special_case = False

        # Check table headers for unit multipliers and special cases
        table_header = table.find("th")
        if table_header:
            header_text = table_header.get_text()
            # Determine unit multiplier based on header text
            if "in Thousands" in header_text:
                unit_multiplier = 1
            elif "in Millions" in header_text:
                unit_multiplier = 1  # Era 1000, mas coloquei 1 para ficar no formato normal do arquivos SEC
            # Check for special case scenario
            if "unless otherwise specified" in header_text:
                special_case = True

        # Process each row of the table
        for row in table.select("tr"):
            onclick_elements = row.select("td.pl a, td.pl.custom a")
            if not onclick_elements:
                continue

            # Extract column title from 'onclick' attribute
            onclick_attr = onclick_elements[0]["onclick"]
            column_title = onclick_attr.split("defref_")[-1].split("',")[0]
            columns.append(column_title)

            # Initialize values array with NaNs
            values = [np.nan] * len(date_time_index)

            # Process each cell in the row
            for i, cell in enumerate(row.select("td.text, td.nump, td.num")):
                if "text" in cell.get("class"):
                    continue

                # Clean and parse cell value
                value = keep_numbers_and_decimals_only_in_string(
                    cell.text.replace("$", "")
                    .replace(",", "")
                    .replace("(", "")
                    .replace(")", "")
                    .strip()
                )
                if value:
                    value = float(value)
                    # Adjust value based on special case and cell class
                    if special_case:
                        value /= 1000
                    else:
                        if "nump" in cell.get("class"):
                            values[i] = value * unit_multiplier
                        else:
                            values[i] = -value * unit_multiplier

            values_set.append(values)

    return columns, values_set, date_time_index


def get_datetime_index_dates_from_statement(soup: BeautifulSoup) -> pd.DatetimeIndex:
    """
    Extracts datetime index dates from the HTML soup object of a financial statement.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object of the HTML document.

    Returns:
        pd.DatetimeIndex: A Pandas DatetimeIndex object containing the extracted dates.
    """
    table_headers = soup.find_all("th", {"class": "th"})
    dates = [str(th.div.string) for th in table_headers if th.div and th.div.string]
    dates = [standardize_date(date).replace(".", "") for date in dates]
    index_dates = pd.to_datetime(dates)
    return index_dates


def standardize_date(date: str) -> str:
    """
    Standardizes date strings by replacing abbreviations with full month names.

    Args:
        date (str): The date string to be standardized.

    Returns:
        str: The standardized date string.
    """
    for abbr, full in zip(calendar.month_abbr[1:], calendar.month_name[1:]):
        date = date.replace(abbr, full)
    return date


def keep_numbers_and_decimals_only_in_string(mixed_string: str):
    """
    Filters a string to keep only numbers and decimal points.

    Args:
        mixed_string (str): The string containing mixed characters.

    Returns:
        str: String containing only numbers and decimal points.
    """
    num = "1234567890."
    allowed = list(filter(lambda x: x in num, mixed_string))
    return "".join(allowed)


def create_dataframe_of_statement_values_columns_dates(
    values_set, columns, index_dates
) -> pd.DataFrame:
    """
    Creates a DataFrame from statement values, columns, and index dates.

    Args:
        values_set (list): List of values for each column.
        columns (list): List of column names.
        index_dates (pd.DatetimeIndex): DatetimeIndex for the DataFrame index.

    Returns:
        pd.DataFrame: DataFrame constructed from the given data.
    """
    transposed_values_set = list(zip(*values_set))
    df = pd.DataFrame(transposed_values_set, columns=columns, index=index_dates)
    return df


def process_one_statement(ticker, accession_number, statement_name):
    """
    Processes a single financial statement identified by ticker, accession number, and statement name.

    Args:
        ticker (str): The stock ticker.
        accession_number (str): The SEC accession number.
        statement_name (str): Name of the financial statement.

    Returns:
        pd.DataFrame or None: DataFrame of the processed statement or None if an error occurs.
    """
    try:
        # Fetch the statement HTML soup
        soup = get_statement_soup(
            ticker,
            accession_number,
            statement_name,
            headers=header,
            statement_keys_map=statement_keys_map,
        )
    except Exception as e:
        logging.error(
            f"Failed to get statement soup: {e} for accession number: {accession_number}"
        )
        return None

    if soup:
        try:
            # Extract data and create DataFrame
            columns, values, dates = extract_columns_values_and_dates_from_statement(
                soup
            )
            df = create_dataframe_of_statement_values_columns_dates(
                values, columns, dates
            )

            if not df.empty:
                # Remove duplicate columns
                df = df.T.drop_duplicates()
            else:
                logging.warning(
                    f"Empty DataFrame for accession number: {accession_number}"
                )
                return None

            return df
        except Exception as e:
            logging.error(f"Error processing statement: {e}")
            return None


def get_label_dictionary(ticker, headers):
    facts = get_facts(ticker, headers)
    us_gaap_data = facts["facts"]["us-gaap"]
    labels_dict = {fact: details["label"] for fact, details in us_gaap_data.items()}
    # Alguns itens tem a sua 'label' igual a None. Se for None, ele será substituido pela respectiva chave
    labels_dict = {key: (value if value is not None else key) for key, value in labels_dict.items()}
    return labels_dict


def rename_statement(statement, label_dictionary):
    # Extract the part after the first "_" and then map it using the label dictionary
    statement.index = statement.index.map(
        lambda x: label_dictionary.get(x.split("_", 1)[-1], x)
    )
    return statement


def ajustar_mes_02_05_08(dt):
    """
    Ajusta o mês de acordo com o mapeamento:
      - 01 -> 02
      - 04 -> 05
      - 07 -> 08
    """
    # Dicionário de substituição de meses
    map_months = {1: 2, 4: 5, 7: 8}
    
    if not pd.isna(dt):
        try:
            dt = pd.to_datetime(dt)  # garante que é datetime
            novo_mes = map_months.get(dt.month, dt.month)
            return dt.replace(month=novo_mes)
        except Exception:
            raise ValueError('O valor fornecido não é uma data válida.')
    return dt


def ajustar_mes_03_06_09(dt):
    """
    Ajusta o mês de acordo com o mapeamento:
      - 04 -> 03
      - 07 -> 06
      - 10 -> 09
    """
    # Dicionário de substituição de meses
    map_months = {4: 3, 7: 6, 10: 9}
    
    if not pd.isna(dt):
        try:
            dt = pd.to_datetime(dt)  # garante que é datetime
            novo_mes = map_months.get(dt.month, dt.month)
            return dt.replace(month=novo_mes)
        except Exception:
            raise ValueError('O valor fornecido não é uma data válida.')
    return dt


def ajustar_mes_04_07_10(dt):
    """
    Ajusta o mês de acordo com o mapeamento:
      - 05 -> 04
      - 08 -> 07
      - 11 -> 10
    """
    # Dicionário de substituição de meses
    map_months = {5: 4, 8: 7, 11: 10}
    
    if not pd.isna(dt):
        try:
            dt = pd.to_datetime(dt)  # garante que é datetime
            novo_mes = map_months.get(dt.month, dt.month)
            return dt.replace(month=novo_mes)
        except Exception:
            raise ValueError('O valor fornecido não é uma data válida.')
    return dt


def ajustar_mes_12_03_06(dt):
    """
    Ajusta o mês de acordo com o mapeamento:
      - 11 -> 12
      - 02 -> 03
      - 05 -> 06
    """
    # Dicionário de substituição de meses
    map_months = {11: 12, 2: 3, 5: 6}
    
    if not pd.isna(dt):
        try:
            dt = pd.to_datetime(dt)  # garante que é datetime
            novo_mes = map_months.get(dt.month, dt.month)
            return dt.replace(month=novo_mes)
        except Exception:
            raise ValueError('O valor fornecido não é uma data válida.')
    return dt


def ajustar_mes_12_03_06_aapl(dt):
    """
    Ajusta o mês de acordo com o mapeamento:
      - 11 -> 12
      - 07 -> 06
      - 04 -> 03
    """
    # Dicionário de substituição de meses
    map_months = {11: 12, 7: 6, 4: 3}
    
    if not pd.isna(dt):
        try:
            dt = pd.to_datetime(dt)  # garante que é datetime
            novo_mes = map_months.get(dt.month, dt.month)
            return dt.replace(month=novo_mes)
        except Exception:
            raise ValueError('O valor fornecido não é uma data válida.')
    return dt