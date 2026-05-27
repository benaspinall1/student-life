def format_user_id(index: int) -> str:
    """
    Maps indices 0-59 to a string with leading zeros for single digits.
    For example:
        0  -> "00"
        5  -> "05"
        12 -> "12"
        59 -> "59"
    """
    if not (0 <= index <= 59):
        raise ValueError("Index out of range (should be 0-59).")
    return f"{index:02d}"


    