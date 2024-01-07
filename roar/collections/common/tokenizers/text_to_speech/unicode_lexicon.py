VALID_LOCALES = [
    "as",
    "bn",
    "en-us",
    "en",
    "gu",
    "hi",
    "kok",
    "mr",
    "or",
    "pa",
    "sd",
    "ur",
    "kn",
    "ta",
    "te",
    "ml",
]

UNICODE_CHARACTER_SETS = {
    "as": ("\u0980", "\u09FF"),
    "bn": ("\u0980", "\u09FF"),
    "gu": ("\u0A80", "\u0AFF"),
    "hi": ("\u0900", "\u097F"),
    "kok": ("\u0900", "\u097F"),
    "mr": ("\u0900", "\u097F"),
    "or": ("\u0B00", "\u0B7F"),
    "pa": ("\u0A00", "\u0A7F"),
    "sd": ("\u0600", "\u06FF"),
    "ur": ("\u0600", "\u06FF"),
    "kn": ("\u0C80", "\u0CFF"),
    "ta": ("\u0B80", "\u0BFF"),
    "te": ("\u0C00", "\u0C7F"),
    "ml": ("\u0D00", "\u0D7F"),
}


def get_unicode_character_set(locale):
    if locale not in VALID_LOCALES:
        raise ValueError(
            f"Unicode character set not found for locale '{locale}'. "
            f"Supported locales {VALID_LOCALES}"
        )
    return UNICODE_CHARACTER_SETS[locale]
