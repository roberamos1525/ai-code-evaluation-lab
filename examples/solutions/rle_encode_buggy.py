def rle_encode(text: str) -> str:
    """Off-by-one: drops the final run."""
    out = []
    for i in range(1, len(text)):
        if text[i] != text[i - 1]:
            out.append(text[i - 1] + "1")
    return "".join(out)
