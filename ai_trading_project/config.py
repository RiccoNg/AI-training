def map_to_yfinance_ticker(code: str) -> str:
    if code.startswith("US."):
        return code.replace("US.", "")
    elif code.endswith(".HK"):
        return code
    elif code.endswith(".SH"):
        return code.replace(".SH", ".SS")
    elif code.endswith(".SZ"):
        return code
    else:
        raise ValueError(f"無法識別的代碼格式：{code}")

TICKER_CODE = "US.QQQ"
