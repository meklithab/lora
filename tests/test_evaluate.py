from fractions import Fraction

from rankalloc.evaluate import extract_flexible, extract_strict, _matches, _normalize_number


def test_extract_strict_reads_after_final_hashes():
    text = "Reasoning here.\n#### 42"
    assert extract_strict(text) == Fraction(42)


def test_extract_strict_uses_last_hashes_block():
    text = "#### 3\nmore text #### 42"
    assert extract_strict(text) == Fraction(42)


def test_extract_strict_none_without_hashes():
    assert extract_strict("The answer is 42 but no marker.") is None


def test_extract_flexible_takes_last_number():
    text = "First 3 apples, then 4 more, total 7."
    assert extract_flexible(text) == Fraction(7)


def test_extract_flexible_none_without_numbers():
    assert extract_flexible("no numbers here") is None


def test_normalize_strips_commas_and_currency():
    assert _normalize_number("$1,234") == Fraction(1234)


def test_normalize_handles_percent():
    assert _normalize_number("50%") == Fraction(50)


def test_normalize_handles_decimal():
    assert _normalize_number("3.5") == Fraction(7, 2)


def test_normalize_handles_negative():
    assert _normalize_number("-12") == Fraction(-12)


def test_normalize_trailing_zeros_equal_to_int():
    assert _normalize_number("42.00") == _normalize_number("42")


def test_normalize_invalid_returns_none():
    assert _normalize_number("not-a-number") is None
    assert _normalize_number("") is None


def test_matches_true_when_equal():
    assert _matches(Fraction(42), "42") is True


def test_matches_false_when_different():
    assert _matches(Fraction(41), "42") is False


def test_matches_false_when_pred_none():
    assert _matches(None, "42") is False


def test_matches_true_with_formatting_differences():
    assert _matches(Fraction(1234), "1,234") is True
