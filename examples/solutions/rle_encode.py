def rle_encode(text: str) -> str:
    """Compress runs of repeated characters (candidate solution for core/rle-encode)."""
    out = []
    i = 0
    while i < len(text):
        j = i
        while j < len(text) and text[j] == text[i]:
            j += 1
        out.append(f"{text[i]}{j - i}")
        i = j
    return "".join(out)
