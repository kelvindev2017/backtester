import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "App.py"
TRANSIENT = ("_btn", "_table", "_chart", "_upload", "_draft")


def literal_key(node):
    """The text of a key=... argument: a plain string, or an f-string's constant tail."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[-1], ast.Constant):
        return node.values[-1].value
    return None


def test_button_keys_are_transient():
    # The app saves widget values under their keys and puts them back on the next run; a button's value cannot be
    # set (StreamlitValueAssignmentNotAllowedError), so every button key must end in a suffix the saver skips.
    tree = ast.parse(APP.read_text())
    wrong = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in (
                "button", "download_button", "form_submit_button", "file_uploader"):
            for keyword in node.keywords:
                if keyword.arg == "key":
                    key = literal_key(keyword.value)
                    if key is not None and not key.endswith(TRANSIENT):
                        wrong.append((node.lineno, key))
    assert not wrong, f"button keys without a transient suffix: {wrong}"
