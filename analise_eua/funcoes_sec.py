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
        "condensed consolidated balance sheets (unaudited)",
        "condensed balance sheets",
        "condensed balance sheet",
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
        "condensed consolidated statements of operations (unaudited)",
        "statements of operations",
        "statement of operations",
        "condensed statements of operations"
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
        "condensed consolidated statements of cash flows (unaudited)",
        "statements of cash flows",
        "statement of cash flows"
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

TEN_K_FORMS = {"10-K"}
TEN_Q_FORMS = {"10-Q"}


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


def _accession_numbers_for_forms(
    ticker,
    forms,
    headers=header,
    empty_message=None,
):
    company_filings_df = get_submission_data_for_ticker(
        ticker, only_filings_df=True, headers=headers
    )
    filings_df = company_filings_df[company_filings_df["form"].isin(forms)]

    if filings_df.empty:
        if empty_message is None:
            forms_text = ", ".join(sorted(forms))
            empty_message = f"No filings found for ticker {ticker} and forms {forms_text}"
        raise ValueError(empty_message)

    filings_df = filings_df.set_index("reportDate")
    accession_nums = filings_df["accessionNumber"]
    accession_nums.index = pd.to_datetime(accession_nums.index)
    return accession_nums
    

def get_facts(ticker, headers=header):

    cik = cik_matching_ticker(ticker)

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    company_facts = requests.get(url, headers=headers).json()
    
    return company_facts


def _get_accounting_facts_block(facts):
    facts_root = facts.get("facts", {})

    if "us-gaap" in facts_root:
        return facts_root["us-gaap"], "us-gaap"

    if "ifrs-full" in facts_root:
        return facts_root["ifrs-full"], "ifrs-full"

    available_taxonomies = ", ".join(sorted(facts_root.keys()))
    raise KeyError(
        "No supported accounting taxonomy found in companyfacts. "
        f"Available taxonomies: {available_taxonomies}"
    )


def facts_DF(ticker, headers=header):

    facts = get_facts(ticker, headers)
    accounting_data, _ = _get_accounting_facts_block(facts)
    df_data = []
    for fact, details in accounting_data.items():
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
    labels_dict = {fact: details["label"] for fact, details in accounting_data.items()}
    
    return df, labels_dict


def annual_facts(ticker, headers=header):
    raw_df, label_dict = _annual_facts_raw(ticker, headers)
    pivot = raw_df.T
    pivot.rename(columns=label_dict, inplace=True)
    return pivot.T


BALANCE_SHEET_FACT_MAP = {
    "Cash and Cash Equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndCashEquivalents",
    ],
    "Marketable Securities": [
        "AvailableForSaleSecuritiesCurrent",
        "AvailableForSaleDebtSecuritiesCurrent",
        "DebtSecuritiesAvailableForSaleCurrent",
        "ShortTermInvestments",
        "AvailableForSaleDebtSecurities",
        "MarketableSecuritiesCurrent",
        "MarketableSecuritiesNoncurrent",
        "AvailableForSaleSecuritiesNoncurrent",
        "AvailableForSaleDebtSecuritiesNoncurrent",
        "DebtSecuritiesAvailableForSaleNoncurrent",
        "EquitySecuritiesFvNiCurrentAndNoncurrent",
        "EquitySecuritiesWithoutReadilyDeterminableFairValueAmount",
        "MarketableSecurities",
        "nvda_MarketableSecuritiesAndEquitySecuritiesFVNI",
    ],
    "Accounts Receivable, Net, Current": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
        "TradeAndOtherCurrentReceivables",
        "CurrentTradeReceivables",
        "TradeReceivablesCurrent",
    ],
    "Inventory, Net": [
        "InventoryNet",
        "InventoriesNetOfReserves",
        "Inventories",
    ],
    "Other Assets, Current": [
        "OtherAssetsCurrent",
        "OtherCurrentAssets",
    ],
    "Assets, Current": [
        "AssetsCurrent",
        "CurrentAssets",
    ],
    "Property, Plant and Equipment, Net": [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
        "PropertyPlantAndEquipment",
    ],
    "Operating Lease Right-of-Use Asset": [
        "OperatingLeaseRightOfUseAsset",
        "OperatingLeaseRightOfUseAssetNet",
    ],
    "Intangible Assets, Net": [
        "FiniteLivedIntangibleAssetsNet",
        "FiniteLivedIntangibleAssetsNetExcludingGoodwill",
        "IndefiniteLivedIntangibleAssetsExcludingGoodwill",
        "IntangibleAssetsNetExcludingGoodwill",
        "IntangibleAssetsOtherThanGoodwill",
    ],
    "Goodwill": [
        "Goodwill",
    ],
    "Other Assets, Noncurrent": [
        "OtherAssetsNoncurrent",
        "OtherNoncurrentAssets",
    ],
    "Assets": [
        "Assets",
    ],
    "Accounts Payable, Current": [
        "AccountsPayableCurrent",
        "AccountsPayableTradeCurrent",
        "AccountsPayableAndAccruedLiabilitiesCurrent",
        "AccountsPayableOtherCurrent",
        "TradeAndOtherCurrentPayables",
        "CurrentTradePayables",
        "TradePayablesCurrent",
    ],
    "Accrued Liabilities, Current": [
        "AccruedLiabilitiesCurrent",
        "EmployeeRelatedLiabilitiesCurrent",
        "OperatingLeaseLiabilityCurrentAndNoncurrent",
        "CurrentAccrualsAndCurrentDeferredIncomeIncludingCurrentContractLiabilities",
        "OtherCurrentLiabilities",
    ],
    "Contract with Customer, Liability, Current": [
        "ContractWithCustomerLiabilityCurrent",
        "DeferredRevenueCurrent",
        "CustomerAdvancesCurrent",
        "fb_ContractWithCustomerLiabilityAndUnusedDeposits",
        "meta_ContractWithCustomerLiabilityAndUnusedDeposits",
    ],
    "Operating Lease Liability, Current": [
        "OperatingLeaseLiabilityCurrent",
        "OperatingLeaseLiability",
        "LesseeOperatingLeaseLiabilityCurrent",
        "OperatingLeaseLiabilitiesCurrent",
        "OperatingLeaseLiabilities",
        "OperatingLeaseCurrent",
        "LeaseLiabilitiesCurrent",
    ],
    "Finance Lease Liability, Current": [
        "FinanceLeaseLiabilityCurrent",
        "LesseeFinanceLeaseLiabilityCurrent",
        "FinanceLeaseCurrent",
    ],
    "Lease Liabilities, Current": [
        "LeaseLiabilitiesCurrent",
    ],
    "Liabilities, Current": [
        "LiabilitiesCurrent",
        "CurrentLiabilities",
    ],
    "Short-term Debt": [
        "DebtCurrent",
        "ShortTermBorrowings",
        "CommercialPaper",
        "ShortTermBankLoans",
        "BankOverdrafts",
        "LongTermDebtCurrent",
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
        "CurrentPortionOfLongTermDebt",
        "LongTermDebtCurrentMaturities",
        "LineOfCreditFacilityAmountOutstanding",
        "ShortTermDebt",
        "CurrentBorrowings",
        "CurrentPortionOfBorrowings",
    ],
    "Long-term Liabilities - Current Portion": [
        "LongTermDebtCurrent",
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
        "CurrentPortionOfLongTermDebt",
        "LongTermDebtCurrentMaturities",
        "CurrentPortionOfBorrowings",
        "CurrentPortionOfNoncurrentBorrowings",
        "CurrentPortionOfBankLoans",
    ],
    "Long-term Debt, Noncurrent": [
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndFinanceLeaseObligations",
        "LongTermDebt",
        "LongTermDebtFairValue",
        "NoncurrentBorrowings",
        "BorrowingsNoncurrent",
    ],
    "Long-term Bank Loans": [
        "LongTermBorrowings",
        "LongTermBankLoans",
        "BankLoansNoncurrent",
        "BorrowingsNoncurrent",
        "NoncurrentBorrowings",
    ],
    "Deferred Tax Liabilities, Net, Noncurrent": [
        "DeferredTaxLiabilitiesNetNoncurrent",
        "DeferredTaxLiabilitiesNoncurrent",
    ],
    "Contract with Customer, Liability, Noncurrent": [
        "ContractWithCustomerLiabilityNoncurrent",
        "DeferredRevenueNoncurrent",
        "CustomerAdvancesNoncurrent",
    ],
    "Operating Lease Liability, Noncurrent": [
        "OperatingLeaseLiabilityNoncurrent",
        "OperatingLeaseLiabilitiesNoncurrent",
        "OperatingLeaseLiability",
        "LesseeOperatingLeaseLiabilityNoncurrent",
        "OperatingLeaseLiabilities",
        "LongTermOperatingLeaseLiabilities",
        "OperatingLeaseNoncurrent",
        "LeaseLiabilitiesNoncurrent",
    ],
    "Finance Lease Liability, Noncurrent": [
        "FinanceLeaseLiabilityNoncurrent",
        "LesseeFinanceLeaseLiabilityNoncurrent",
        "FinanceLeaseNoncurrent",
    ],
    "Lease Liabilities, Noncurrent": [
        "LeaseLiabilitiesNoncurrent",
    ],
    "Other Liabilities, Noncurrent": [
        "OtherLiabilitiesNoncurrent",
        "OtherNoncurrentLiabilities",
    ],
    "Liabilities": [
        "Liabilities",
    ],
    "Common Stocks, Including Additional Paid in Capital": [
        "CommonStocksIncludingAdditionalPaidInCapital",
        "AdditionalPaidInCapitalCommonStock",
        "CommonStocksIncludingAdditionalPaidInCapitalAndRetainedEarnings",
        "IssuedCapital",
        "ShareCapital",
        "OtherReserves",
    ],
    "Retained Earnings (Accumulated Deficit)": [
        "RetainedEarningsAccumulatedDeficit",
        "RetainedEarnings",
    ],
    "Accumulated Other Comprehensive Income (Loss), Net of Tax": [
        "AccumulatedOtherComprehensiveIncomeLossNetOfTax",
        "OtherReserves",
    ],
    "Stockholders' Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "Equity",
        "EquityAttributableToOwnersOfParent",
    ],
}


INCOME_STATEMENT_FACT_MAP = {
    "Revenue": [
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueServicesNet",
        "AdvertisingRevenue",
        "CloudRevenue",
        "SalesRevenueGoodsNet",
        "Revenue",
    ],
    "Cost of Revenue": [
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostOfSales",
        "CostOfGoodsAndServicesSold",
        "CostOfServices",
        "msft_CostOfServicesAndOther",
        "CostOfSales",
    ],
    "Gross Profit": [
        "GrossProfit",
    ],
    "Research and Development Expense": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentAssetAcquiredOtherThanThroughBusinessCombinationWrittenOff",
    ],
    "Selling, General and Administrative Expense": [
        "SellingGeneralAndAdministrativeExpense",
        "MarketingExpense",
    ],
    "Operating Expenses": [
        "OperatingExpenses",
        "CostsAndExpenses",
        "OperatingCostsAndExpenses",
        "OtherOperatingExpenses",
    ],
    "Operating Income": [
        "OperatingIncomeLoss",
        "IncomeLossFromOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "ProfitLossFromOperatingActivities",
    ],
    "Interest Expense": [
        "InterestExpense",
        "InterestAndDebtExpense",
        "InterestExpenseNonoperating",
    ],
    "Other Nonoperating Income (Expense)": [
        "OtherNonoperatingIncomeExpense",
        "NonoperatingIncomeExpense",
        "InterestAndOtherIncomeExpenseNet",
        "OtherIncomeExpenseNet",
        "GainsLossesOnSalesOfOtherAssets",
        "msft_GainLossOnInvestmentsAndDerivativeInstruments",
        "OtherGainsLosses",
        "FinanceIncome",
        "FinanceCosts",
    ],
    "Income before income tax": [
        "IncomeBeforeTaxExpenseBenefit",
        "PretaxIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "ProfitLossBeforeTax",
    ],
    "Income tax expense": [
        "IncomeTaxExpenseBenefit",
        "CurrentTaxExpenseBenefit",
        "DeferredTaxExpenseBenefit",
        "IncomeTaxes",
        "IncomeTaxExpenseContinuingOperations",
    ],
    "Net Income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLossAttributableToOwnersOfParent",
    ],
    "Basic EPS": [
        "EarningsPerShareBasic",
        "IncomeLossFromContinuingOperationsPerBasicShare",
    ],
    "Diluted EPS": [
        "EarningsPerShareDiluted",
        "IncomeLossFromContinuingOperationsPerDilutedShare",
    ],
    "Weighted Average Shares Outstanding, Basic": [
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    ],
    "Weighted Average Shares Outstanding, Diluted": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    ],
}


CASH_FLOW_FACT_MAP = {
    "Net Cash Provided by (Used in) Operating Activities": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInContinuingOperations",
        "CashFlowsFromUsedInOperatingActivities",
    ],
    "Depreciation and Amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "Depreciation",
        "DepreciationExpense",
        "DepreciationPropertyPlantAndEquipment",
        "DepreciationAndImpairmentOfPropertyPlantAndEquipment",
        "DepreciationPropertyRightofuseAsset",
        "AmortizationOfIntangibleAssets",
        "AmortisationExpense",
        "AmortizationExpense",
        "msft_DepreciationAmortizationAndOther",
        "DepreciationAndAmortisationExpense",
    ],
    "Share-based Compensation": [
        "ShareBasedCompensation",
        "AllocatedShareBasedCompensationExpense",
    ],
    "Change in Accounts Receivable": [
        "IncreaseDecreaseInAccountsReceivable",
        "IncreaseDecreaseInReceivables",
        "IncreaseDecreaseInTradeAndOtherReceivables",
    ],
    "Change in Inventory": [
        "IncreaseDecreaseInInventories",
        "IncreaseDecreaseInInventory",
        "IncreaseDecreaseInInventories",
    ],
    "Change in Accounts Payable": [
        "IncreaseDecreaseInAccountsPayable",
        "IncreaseDecreaseInAccountsPayableAndAccruedLiabilities",
        "IncreaseDecreaseInTradeAndOtherPayables",
    ],
    "Net Cash Provided by (Used in) Investing Activities": [
        "NetCashProvidedByUsedInInvestingActivities",
        "CashFlowsFromUsedInInvestingActivities",
    ],
    "Capital Expenditures": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquirePropertyPlantEquipmentAndOtherProductiveAssets",
        "nvda_PurchasesOfPropertyAndEquipmentAndIntangibleAssets",
        "nvda_PaymentsForFinancedPropertyPlantAndEquipmentAndIntangibleAssetsFinancingActivities",
        "nvda_PaymentsForFinancedPropertyPlantAndEquipmentFinancingActivities",
        "PurchaseOfPropertyPlantAndEquipment",
        "PaymentsToAcquireIntangibleAssets",
    ],
    "Purchases of Marketable Securities": [
        "PaymentsToAcquireAvailableForSaleSecuritiesDebt",
        "PaymentsToAcquireShortTermInvestments",
        "PaymentsToAcquireAvailableForSaleSecurities",
        "PaymentsToAcquireMarketableSecurities",
        "PaymentsToAcquireInvestments",
    ],
    "Sales/Maturities of Marketable Securities": [
        "ProceedsFromSaleOfAvailableForSaleSecuritiesDebt",
        "ProceedsFromMaturitiesPrepaymentsAndCallsOfAvailableForSaleSecurities",
        "ProceedsFromSaleMaturityAndCollectionsOfShortTermInvestments",
        "ProceedsFromSaleOfAvailableForSaleSecurities",
        "ProceedsFromSaleAndMaturityOfMarketableSecurities",
        "ProceedsFromInvestments",
        "msft_ProceedsFromInvestments",
    ],
    "Net Cash Provided by (Used in) Financing Activities": [
        "NetCashProvidedByUsedInFinancingActivities",
        "CashFlowsFromUsedInFinancingActivities",
    ],
    "Proceeds from Issuance of Long-term Debt": [
        "ProceedsFromIssuanceOfLongTermDebt",
        "ProceedsFromLongTermDebtAndOther",
        "ProceedsFromIssuanceOfDebt",
        "ProceedsFromIssuanceOfOtherLongTermDebt",
        "ProceedsFromIssuanceOfSeniorLongTermDebt",
        "ProceedsFromOtherDebt",
        "ProceedsFromBorrowings",
    ],
    "Repayments of Debt": [
        "RepaymentsOfLongTermDebt",
        "RepaymentsOfDebt",
        "RepaymentsOfShortTermDebt",
        "RepaymentsOfLongTermDebtAndCapitalSecurities",
        "RepaymentsOfOtherDebt",
        "RepaymentsOfBorrowings",
    ],
    "Dividends Paid": [
        "PaymentsOfDividends",
        "PaymentsOfOrdinaryDividends",
        "DividendsPaid",
    ],
    "Repurchases of Common Stock": [
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfEquity",
    ],
    "Proceeds from Stock Plans": [
        "ProceedsFromStockOptionsExercised",
        "ExcessTaxBenefitFromShareBasedCompensationFinancingActivities",
        "nvda_Netproceedspaymentsrelatedtoemployeestockplans",
        "nvda_NetProceedsPaymentsRelatedToEmployeeStockPlans",
    ],
}

STATEMENT_LABEL_ALIASES = {
    "Cash and Cash Equivalents": [
        "cash and cash equivalents",
        "cash and cash equivalents at end of period",
    ],
    "Marketable Securities": [
        "financial assets at fair value through profit or loss",
        "financial assets at fair value through other comprehensive income",
        "current financial assets at amortised cost",
        "current financial assets at amortized cost",
        "non-current financial assets at amortised cost",
        "non-current financial assets at amortized cost",
        "other financial assets current",
        "other financial assets noncurrent",
    ],
    "Accounts Receivable, Net, Current": [
        "trade receivables",
        "trade receivables net",
        "trade and other receivables",
        "notes and accounts receivable",
        "notes receivable",
    ],
    "Inventory, Net": [
        "inventories",
    ],
    "Other Assets, Current": [
        "other current assets",
        "prepayments",
    ],
    "Assets, Current": [
        "current assets",
    ],
    "Property, Plant and Equipment, Net": [
        "property plant and equipment",
        "property plant and equipment net",
    ],
    "Intangible Assets, Net": [
        "intangible assets",
        "intangible assets net",
        "other intangible assets",
    ],
    "Goodwill": [
        "goodwill",
    ],
    "Other Assets, Noncurrent": [
        "other non-current assets",
        "other noncurrent assets",
    ],
    "Assets": [
        "assets",
        "total assets",
    ],
    "Accounts Payable, Current": [
        "accounts payable",
        "trade payables",
        "trade and other payables",
        "payables",
    ],
    "Accrued Liabilities, Current": [
        "accrued liabilities",
        "other current liabilities",
        "accruals",
    ],
    "Contract with Customer, Liability, Current": [
        "contract liabilities current",
        "contract liabilities",
        "deferred revenue current",
        "customer advances",
    ],
    "Liabilities, Current": [
        "current liabilities",
    ],
    "Short-term Debt": [
        "short-term borrowings",
        "current borrowings",
        "current portion of long-term borrowings",
        "current portion of bonds payable",
    ],
    "Long-term Debt, Noncurrent": [
        "long-term borrowings",
        "non-current borrowings",
        "noncurrent borrowings",
        "bonds payable",
    ],
    "Other Liabilities, Noncurrent": [
        "other non-current liabilities",
        "other noncurrent liabilities",
    ],
    "Liabilities": [
        "liabilities",
        "total liabilities",
    ],
    "Common Stocks, Including Additional Paid in Capital": [
        "issued capital",
        "share capital",
        "capital stock",
        "capital surplus",
        "additional paid-in capital",
    ],
    "Retained Earnings (Accumulated Deficit)": [
        "retained earnings",
        "unappropriated earnings",
        "accumulated deficit",
    ],
    "Accumulated Other Comprehensive Income (Loss), Net of Tax": [
        "other equity",
        "other reserves",
        "accumulated other comprehensive income",
    ],
    "Stockholders' Equity": [
        "equity",
        "equity attributable to owners of the parent",
        "total equity",
    ],
    "Revenue": [
        "revenue",
        "revenues",
        "net revenue",
        "net revenues",
        "sales",
    ],
    "Cost of Revenue": [
        "cost of sales",
        "cost of revenue",
        "cost of goods sold",
    ],
    "Gross Profit": [
        "gross profit",
        "gross profit from operations",
    ],
    "Research and Development Expense": [
        "research and development expenses",
        "research and development expense",
    ],
    "Selling, General and Administrative Expense": [
        "selling general and administrative expenses",
        "general and administrative expenses",
        "administrative expenses",
    ],
    "Operating Expenses": [
        "operating expenses",
        "total operating expenses",
    ],
    "Operating Income": [
        "income from operations",
        "operating income",
        "profit from operations",
        "profit from operating activities",
    ],
    "Interest Expense": [
        "interest expense",
        "finance costs",
    ],
    "Other Nonoperating Income (Expense)": [
        "other income expenses",
        "other gains and losses",
        "other income",
        "finance income",
    ],
    "Income before income tax": [
        "profit before tax",
        "income before tax",
        "profit before income tax",
    ],
    "Income tax expense": [
        "income tax expense",
        "income tax expense benefit",
    ],
    "Net Income": [
        "profit for the year",
        "profit",
        "net income",
        "profit attributable to owners of the parent",
        "profit attributable to shareholders of the parent",
    ],
    "Basic EPS": [
        "basic earnings per share",
        "basic eps",
    ],
    "Diluted EPS": [
        "diluted earnings per share",
        "diluted eps",
    ],
    "Weighted Average Shares Outstanding, Basic": [
        "weighted average number of shares outstanding basic",
        "weighted average ordinary shares basic",
    ],
    "Weighted Average Shares Outstanding, Diluted": [
        "weighted average number of diluted shares outstanding",
        "weighted average ordinary shares diluted",
    ],
    "Net Cash Provided by (Used in) Operating Activities": [
        "net cash generated from operating activities",
        "net cash provided by operating activities",
        "cash flows from used in operating activities",
    ],
    "Depreciation and Amortization": [
        "depreciation and amortisation expense",
        "depreciation and amortization expense",
        "depreciation amortisation and amortization",
        "depreciation expense",
        "depreciation of property plant and equipment",
        "depreciation of right-of-use assets",
        "amortisation expense",
        "amortization expense",
    ],
    "Share-based Compensation": [
        "share-based compensation",
    ],
    "Change in Accounts Receivable": [
        "increase decrease in trade and other receivables",
        "decrease increase in trade receivables",
    ],
    "Change in Inventory": [
        "increase decrease in inventories",
    ],
    "Change in Accounts Payable": [
        "increase decrease in trade and other payables",
        "increase decrease in accounts payable",
    ],
    "Net Cash Provided by (Used in) Investing Activities": [
        "net cash used in investing activities",
        "net cash provided by investing activities",
        "cash flows from used in investing activities",
    ],
    "Capital Expenditures": [
        "acquisition of property plant and equipment",
        "purchase of property plant and equipment",
        "payments to acquire property plant and equipment",
        "capital expenditures",
    ],
    "Net Cash Provided by (Used in) Financing Activities": [
        "net cash generated from financing activities",
        "net cash used in financing activities",
        "cash flows from used in financing activities",
    ],
    "Proceeds from Issuance of Long-term Debt": [
        "proceeds from borrowings",
        "proceeds from long-term debt",
    ],
    "Repayments of Debt": [
        "repayments of borrowings",
        "repayment of bonds",
        "repayment of long-term debt",
    ],
    "Dividends Paid": [
        "dividends paid",
        "cash dividends paid",
    ],
    "Repurchases of Common Stock": [
        "repurchase of treasury shares",
        "repurchases of common stock",
    ],
}


def _facts_raw_for_forms(
    ticker,
    forms,
    headers=header,
    prefer_full_year=False,
    prefer_quarter=False,
    prefer_fy_match=False,
    empty_message=None,
):
    accession_nums = _accession_numbers_for_forms(
        ticker=ticker,
        forms=forms,
        headers=headers,
        empty_message=empty_message,
    )
    df, label_dict = facts_DF(ticker, headers)
    facts_df = df[df["accn"].isin(accession_nums)]
    facts_df = facts_df[facts_df.index.isin(accession_nums.index)].reset_index()

    duration_days = (facts_df["end"] - facts_df["start"]).dt.days
    sort_columns = ["fact", "end"]

    if prefer_fy_match:
        fy_numeric = pd.to_numeric(facts_df["fy"], errors="coerce")
        facts_df["matches_end_fy"] = fy_numeric.eq(facts_df["end"].dt.year)
        sort_columns.append("matches_end_fy")

    if prefer_full_year:
        facts_df["is_full_year_duration"] = (
            facts_df["start"].notna() & duration_days.ge(300)
        )
        sort_columns.append("is_full_year_duration")

    if prefer_quarter:
        facts_df["is_quarter_duration"] = (
            facts_df["start"].notna() & duration_days.between(60, 120)
        )
        sort_columns.append("is_quarter_duration")

    facts_df["is_instant_fact"] = facts_df["start"].isna()
    sort_columns.extend(["is_instant_fact", "filed", "accn"])

    facts_df = facts_df.sort_values(sort_columns)
    facts_df = facts_df.drop_duplicates(subset=["fact", "end"], keep="last")
    pivot = facts_df.pivot(index="end", columns="fact", values="val")

    return pivot.T, label_dict


def _annual_facts_raw(ticker, headers=header):
    # Companyfacts can include the current year's 10-K values together with
    # comparative prior-year columns and quarter-length periods for the same
    # fact/end pair. Rank rows so we keep the filing that actually corresponds
    # to the reported fiscal year-end instead of averaging incompatible values.
    return _facts_raw_for_forms(
        ticker=ticker,
        forms=TEN_K_FORMS,
        headers=headers,
        prefer_full_year=True,
        prefer_fy_match=True,
    )


def annual_facts_raw(ticker, headers=header):
    raw_df, _ = _annual_facts_raw(ticker, headers)
    return raw_df


def _quarterly_facts_raw(ticker, headers=header):
    # 10-Q filings can mix instant facts with year-to-date values for the same
    # end date. Prefer quarter-length durations while keeping instant facts for
    # balance-sheet style metrics.
    return _facts_raw_for_forms(
        ticker=ticker,
        forms=TEN_Q_FORMS,
        headers=headers,
        prefer_quarter=True,
    )


def quarterly_facts_raw(ticker, headers=header):
    raw_df, _ = _quarterly_facts_raw(ticker, headers)
    return raw_df


def _normalize_fact_label(value):
    if value is None:
        return ""

    normalized = str(value).strip().lower()
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("(", "")
    normalized = normalized.replace(")", "")
    normalized = normalized.replace("-", " ")
    normalized = " ".join(normalized.split())
    return normalized


def _build_label_to_facts_map(label_dict):
    label_to_facts = {}

    for fact_name, label in label_dict.items():
        normalized_label = _normalize_fact_label(label)
        if not normalized_label:
            continue
        label_to_facts.setdefault(normalized_label, []).append(fact_name)

    return label_to_facts


def _candidate_fact_names(raw_df, candidate_fact, label_to_facts, output_label=None):
    candidate_names = []

    if candidate_fact in raw_df.index:
        candidate_names.append(candidate_fact)

    label_candidates = [candidate_fact]
    if output_label is not None:
        label_candidates.append(output_label)
        label_candidates.extend(STATEMENT_LABEL_ALIASES.get(output_label, []))

    for label_candidate in label_candidates:
        normalized_candidate = _normalize_fact_label(label_candidate)
        for fact_name in label_to_facts.get(normalized_candidate, []):
            if fact_name in raw_df.index and fact_name not in candidate_names:
                candidate_names.append(fact_name)

    return candidate_names


def _build_statement_from_fact_map(raw_df, fact_map, label_dict=None):
    statement_rows = {}
    label_to_facts = _build_label_to_facts_map(label_dict or {})

    for output_label, candidate_facts in fact_map.items():
        merged_row = None

        for fact_name in candidate_facts:
            resolved_fact_names = _candidate_fact_names(
                raw_df=raw_df,
                candidate_fact=fact_name,
                label_to_facts=label_to_facts,
                output_label=output_label,
            )

            if not resolved_fact_names:
                continue

            for resolved_fact_name in resolved_fact_names:
                row = raw_df.loc[resolved_fact_name]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                if row.dropna().empty:
                    continue

                if merged_row is None:
                    merged_row = row.copy()
                else:
                    merged_row = merged_row.combine_first(row)

        if merged_row is not None and not merged_row.dropna().empty:
            statement_rows[output_label] = merged_row

    if not statement_rows:
        return pd.DataFrame(columns=raw_df.columns)

    statement_df = pd.DataFrame(statement_rows).T
    statement_df = statement_df.dropna(axis=0, how="all")
    statement_df = statement_df.dropna(axis=1, how="all")
    return statement_df


def _fill_row_from_formula(df, target_row, left_row, right_row, operation="subtract"):
    if left_row not in df.index or right_row not in df.index:
        return df

    derived_series = None
    if operation == "subtract":
        derived_series = df.loc[left_row] - df.loc[right_row]
    elif operation == "add":
        derived_series = df.loc[left_row] + df.loc[right_row]
    else:
        raise ValueError("Unsupported operation")

    if target_row not in df.index:
        df.loc[target_row] = derived_series
        return df

    df.loc[target_row] = df.loc[target_row].combine_first(derived_series)
    return df


def _postprocess_income_statement(statement_df):
    df = statement_df.copy()

    df = _fill_row_from_formula(
        df,
        target_row="Operating Expenses",
        left_row="Gross Profit",
        right_row="Operating Income",
        operation="subtract",
    )
    df = _fill_row_from_formula(
        df,
        target_row="Income before income tax",
        left_row="Net Income",
        right_row="Income tax expense",
        operation="add",
    )

    return df


def _postprocess_balance_sheet(statement_df):
    df = statement_df.copy()

    if "Short-term Debt" in df.index and "Short-term debt" not in df.index:
        df.loc["Short-term debt"] = df.loc["Short-term Debt"]

    if (
        "Long-term Liabilities - Current Portion" in df.index
        and "Long-term liabilities - current portion" not in df.index
    ):
        df.loc["Long-term liabilities - current portion"] = df.loc[
            "Long-term Liabilities - Current Portion"
        ]

    if "Long-term Debt, Noncurrent" in df.index and "Long-term debt" not in df.index:
        df.loc["Long-term debt"] = df.loc["Long-term Debt, Noncurrent"]

    if "Long-term Bank Loans" in df.index and "Long-term bank loans" not in df.index:
        df.loc["Long-term bank loans"] = df.loc["Long-term Bank Loans"]

    if "Lease Liabilities, Current" in df.index and "Lease liabilities, current" not in df.index:
        df.loc["Lease liabilities, current"] = df.loc["Lease Liabilities, Current"]

    if (
        "Lease Liabilities, Noncurrent" in df.index
        and "Lease liabilities, noncurrent" not in df.index
    ):
        df.loc["Lease liabilities, noncurrent"] = df.loc[
            "Lease Liabilities, Noncurrent"
        ]

    if "Operating Lease Liability, Current" in df.index and "Operating lease liability, current" not in df.index:
        df.loc["Operating lease liability, current"] = df.loc[
            "Operating Lease Liability, Current"
        ]

    if (
        "Operating Lease Liability, Noncurrent" in df.index
        and "Operating lease liability, noncurrent" not in df.index
    ):
        df.loc["Operating lease liability, noncurrent"] = df.loc[
            "Operating Lease Liability, Noncurrent"
        ]

    if "Finance Lease Liability, Current" in df.index and "Finance lease liability, current" not in df.index:
        df.loc["Finance lease liability, current"] = df.loc[
            "Finance Lease Liability, Current"
        ]

    if (
        "Finance Lease Liability, Noncurrent" in df.index
        and "Finance lease liability, noncurrent" not in df.index
    ):
        df.loc["Finance lease liability, noncurrent"] = df.loc[
            "Finance Lease Liability, Noncurrent"
        ]

    return df


def _build_standardized_financial_statements(raw_df, label_dict=None):
    income_statement_df = _build_statement_from_fact_map(
        raw_df, INCOME_STATEMENT_FACT_MAP, label_dict=label_dict
    )
    income_statement_df = _postprocess_income_statement(income_statement_df)
    balance_sheet_df = _build_statement_from_fact_map(
        raw_df, BALANCE_SHEET_FACT_MAP, label_dict=label_dict
    )
    balance_sheet_df = _postprocess_balance_sheet(balance_sheet_df)
    cash_flow_df = _build_statement_from_fact_map(
        raw_df, CASH_FLOW_FACT_MAP, label_dict=label_dict
    )

    return income_statement_df, balance_sheet_df, cash_flow_df


def _facts_to_labeled_pivot(raw_df, label_dict):
    pivot = raw_df.T
    pivot.rename(columns=label_dict, inplace=True)
    return pivot.T


def _balance_sheet_from_raw_df(raw_df, label_dict=None):
    statement_df = _build_statement_from_fact_map(
        raw_df, BALANCE_SHEET_FACT_MAP, label_dict=label_dict
    )
    return _postprocess_balance_sheet(statement_df)


def _income_statement_from_raw_df(raw_df, label_dict=None):
    statement_df = _build_statement_from_fact_map(
        raw_df, INCOME_STATEMENT_FACT_MAP, label_dict=label_dict
    )
    return _postprocess_income_statement(statement_df)


def _cash_flow_from_raw_df(raw_df, label_dict=None):
    return _build_statement_from_fact_map(
        raw_df, CASH_FLOW_FACT_MAP, label_dict=label_dict
    )


def annual_balance_sheet_from_companyfacts(ticker, headers=header):
    raw_df, label_dict = _annual_facts_raw(ticker, headers)
    return _balance_sheet_from_raw_df(raw_df, label_dict=label_dict)


def annual_income_statement_from_companyfacts(ticker, headers=header):
    raw_df, label_dict = _annual_facts_raw(ticker, headers)
    return _income_statement_from_raw_df(raw_df, label_dict=label_dict)


def annual_cash_flow_from_companyfacts(ticker, headers=header):
    raw_df, label_dict = _annual_facts_raw(ticker, headers)
    return _cash_flow_from_raw_df(raw_df, label_dict=label_dict)


def annual_facts_standardized_separated(ticker, headers=header):
    raw_df, label_dict = _annual_facts_raw(ticker, headers)
    return _build_standardized_financial_statements(raw_df, label_dict=label_dict)


def quarterly_balance_sheet_from_companyfacts(ticker, headers=header):
    raw_df, label_dict = _quarterly_facts_raw(ticker, headers)
    return _balance_sheet_from_raw_df(raw_df, label_dict=label_dict)


def quarterly_income_statement_from_companyfacts(ticker, headers=header):
    raw_df, label_dict = _quarterly_facts_raw(ticker, headers)
    return _income_statement_from_raw_df(raw_df, label_dict=label_dict)


def quarterly_cash_flow_from_companyfacts(ticker, headers=header):
    raw_df, label_dict = _quarterly_facts_raw(ticker, headers)
    return _cash_flow_from_raw_df(raw_df, label_dict=label_dict)


def quarterly_facts_standardized_separated(ticker, headers=header):
    raw_df, label_dict = _quarterly_facts_raw(ticker, headers)
    return _build_standardized_financial_statements(raw_df, label_dict=label_dict)


def _find_candidate_rows_for_missing_columns(raw_df, statement_row, current_facts):
    missing_columns = statement_row[statement_row.isna()].index
    if len(missing_columns) == 0:
        return []

    candidates = []

    for fact_name in raw_df.index:
        if fact_name in current_facts:
            continue

        row = raw_df.loc[fact_name]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        available_count = row.loc[missing_columns].notna().sum()
        if available_count == 0:
            continue

        candidates.append((fact_name, int(available_count)))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def diagnose_missing_companyfacts_rows(ticker, fact_map, headers=header, top_n=10):
    raw_df, _ = _annual_facts_raw(ticker, headers)
    statement_df = _build_statement_from_fact_map(raw_df, fact_map)
    diagnostics = {}

    for output_label, candidate_facts in fact_map.items():
        if output_label not in statement_df.index:
            continue

        statement_row = statement_df.loc[output_label]
        if statement_row.notna().all():
            continue

        diagnostics[output_label] = {
            "missing_columns": list(statement_row[statement_row.isna()].index),
            "mapped_facts": candidate_facts,
            "candidate_rows": _find_candidate_rows_for_missing_columns(
                raw_df=raw_df,
                statement_row=statement_row,
                current_facts=candidate_facts,
            )[:top_n],
        }

    return diagnostics


def diagnose_missing_balance_sheet_rows(ticker, headers=header, top_n=10):
    return diagnose_missing_companyfacts_rows(
        ticker=ticker,
        fact_map=BALANCE_SHEET_FACT_MAP,
        headers=headers,
        top_n=top_n,
    )


def diagnose_missing_income_statement_rows(ticker, headers=header, top_n=10):
    return diagnose_missing_companyfacts_rows(
        ticker=ticker,
        fact_map=INCOME_STATEMENT_FACT_MAP,
        headers=headers,
        top_n=top_n,
    )


def diagnose_missing_cash_flow_rows(ticker, headers=header, top_n=10):
    return diagnose_missing_companyfacts_rows(
        ticker=ticker,
        fact_map=CASH_FLOW_FACT_MAP,
        headers=headers,
        top_n=top_n,
    )


def quarterly_facts(ticker, headers=header):
    raw_df, label_dict = _quarterly_facts_raw(ticker, headers)
    return _facts_to_labeled_pivot(raw_df, label_dict)


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
        and file_name
        and (
            "Statement" in long_name_tag.text
            or "Disclosure" in long_name_tag.text
        )
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
    accounting_data, _ = _get_accounting_facts_block(facts)
    labels_dict = {fact: details["label"] for fact, details in accounting_data.items()}
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
