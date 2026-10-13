# Rejected by the standard and strict AST policies (and blocked again at run time).
import os


def rle_encode(text: str) -> str:
    os.system("echo this never runs inside the sandbox")
    return text
