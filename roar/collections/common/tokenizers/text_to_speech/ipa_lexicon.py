SUPPORTED_LOCALES = ["en-US", "de-DE", "es-ES"]

DEFAULT_PUNCTUATION = (
    ",",
    ".",
    "!",
    "?",
    "-",
    ":",
    ";",
    "/",
    '"',
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
)

VITS_PUNCTUATION = (
    ",",
    ".",
    "!",
    "?",
    "-",
    ":",
    ";",
    '"',
    "«",
    "»",
    "“",
    "”",
    "¡",
    "¿",
    "—",
    "…",
)

GRAPHEME_CHARACTER_SETS = {
    "en-US": (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ),
    "es-ES": (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "Á",
        "É",
        "Í",
        "Ñ",
        "Ó",
        "Ú",
        "Ü",
    ),
    # ref: https://en.wikipedia.org/wiki/German_orthography#Alphabet
    "de-DE": (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "Ä",
        "Ö",
        "Ü",
        "ẞ",
    ),
}

IPA_CHARACTER_SETS = {
    "en-US": (
        "a",
        "b",
        "d",
        "e",
        "f",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "z",
        "æ",
        "ð",
        "ŋ",
        "ɐ",
        "ɑ",
        "ɔ",
        "ə",
        "ɚ",
        "ɛ",
        "ɜ",
        "ɡ",
        "ɪ",
        "ɬ",
        "ɹ",
        "ɾ",
        "ʃ",
        "ʊ",
        "ʌ",
        "ʒ",
        "ʔ",
        "ʲ",
        "̃",
        "̩",
        "θ",
        "ᵻ",
    ),
    "es-ES": (
        "a",
        "b",
        "d",
        "e",
        "f",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "r",
        "s",
        "t",
        "u",
        "w",
        "x",
        "ð",
        "ŋ",
        "ɛ",
        "ɡ",
        "ɣ",
        "ɪ",
        "ɲ",
        "ɾ",
        "ʃ",
        "ʊ",
        "ʎ",
        "ʒ",
        "ʝ",
        "β",
        "θ",
    ),
    "de-DE": (
        "1",
        "a",
        "b",
        "d",
        "e",
        "f",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "ç",
        "ø",
        "ŋ",
        "œ",
        "ɐ",
        "ɑ",
        "ɒ",
        "ɔ",
        "ə",
        "ɛ",
        "ɜ",
        "ɡ",
        "ɪ",
        "ɹ",
        "ɾ",
        "ʃ",
        "ʊ",
        "ʌ",
        "ʒ",
        "̃",
        "θ",
    ),
}

GRAPHEME_CHARACTER_CASES = ["upper", "lower", "mixed"]

# fmt: on


def validate_locale(locale):
    if locale not in SUPPORTED_LOCALES:
        raise ValueError(
            f"Unsupported locale '{locale}'. " f"Supported locales {SUPPORTED_LOCALES}"
        )


def get_grapheme_character_set(locale: str, case: str = "upper") -> str:
    if locale not in GRAPHEME_CHARACTER_SETS:
        raise ValueError(
            f"Grapheme character set not found for locale '{locale}'. "
            f"Supported locales {GRAPHEME_CHARACTER_SETS.keys()}"
        )

    charset_str_origin = "".join(GRAPHEME_CHARACTER_SETS[locale])
    if case == "upper":
        # Directly call .upper() will convert 'ß' into 'SS' according to https://bugs.python.org/issue30810.
        charset_str = charset_str_origin.replace("ß", "ẞ").upper()
    elif case == "lower":
        charset_str = charset_str_origin.lower()
    elif case == "mixed":
        charset_str = (
            charset_str_origin.replace("ß", "ẞ").upper() + charset_str_origin.lower()
        )
    else:
        raise ValueError(
            f"Grapheme character case not found: '{case}'. Supported cases are {GRAPHEME_CHARACTER_CASES}"
        )

    return charset_str


def get_ipa_character_set(locale):
    if locale not in IPA_CHARACTER_SETS:
        raise ValueError(
            f"IPA character set not found for locale '{locale}'. "
            f"Supported locales {IPA_CHARACTER_SETS.keys()}"
        )
    char_set = set(IPA_CHARACTER_SETS[locale])
    return char_set


def get_ipa_punctuation_list(locale):
    if locale is None:
        return sorted(list(DEFAULT_PUNCTUATION))

    validate_locale(locale)

    punct_set = set(DEFAULT_PUNCTUATION)
    # TODO verify potential mismatches with locale-specific punctuation sets used
    #  in nemo_text_processing.text_normalization.en.taggers.punctuation.py
    if locale in ["de-DE", "es-ES"]:
        # ref: https://en.wikipedia.org/wiki/Guillemet#Uses
        punct_set.update(["«", "»", "‹", "›"])
    if locale == "de-DE":
        # ref: https://en.wikipedia.org/wiki/German_orthography#Punctuation
        punct_set.update(
            [
                "„",  # double low-9 quotation mark, U+201E, decimal 8222
                "“",  # left double quotation mark, U+201C, decimal 8220
                "‚",  # single low-9 quotation mark, U+201A, decimal 8218
                "‘",  # left single quotation mark, U+2018, decimal 8216
                "‒",  # figure dash, U+2012, decimal 8210
                "–",  # en dash, U+2013, decimal 8211
                "—",  # em dash, U+2014, decimal 8212
            ]
        )
    elif locale == "es-ES":
        # ref: https://en.wikipedia.org/wiki/Spanish_orthography#Punctuation
        punct_set.update(["¿", "¡"])

    punct_list = sorted(list(punct_set))
    return punct_list
