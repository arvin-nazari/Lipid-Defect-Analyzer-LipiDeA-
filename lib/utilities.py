"""
Input-file parsing and value coercion.

Every LipiDeA option comes from one plain-text ``key: value`` file, by
convention ``prep.in``. This module owns reading that file and turning raw
strings into the booleans, lists, and paths the pipeline steps expect.

"""

from pathlib import Path

# ==================================================
# Input-file parser
# ==================================================
def load_input_file(path):
    """Read the input file into a dict of {lowercased key: raw string value}.

        input: input/traj.pdb
        model: CG
        neutral: TAG,DAG,TOG,DOG
        triangle_size: 5.0

    Keys are lowercased, so custom_FF and custom_ff are the same option.
    Blank lines, full-line # comments, and trailing # comments are ignored.
    Both "key: value" and "key = value" are accepted.

    Unrecognized keys are kept in the returned dict rather than rejected;
    main.py decides what to do with them (see IGNORED_CONFIG_KEYS there).

    path: str or Path to the input file.
    Returns: dict mapping key to value, both str.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input/config file not found: {path.resolve()}")
    cfg = {}

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # Remove inline comments before parsing the key/value separator.
            line_no_comment = line.split("#", 1)[0].strip()
            if not line_no_comment:
                continue

            if ":" in line_no_comment:
                key, value = line_no_comment.split(":", 1)
            elif "=" in line_no_comment:
                key, value = line_no_comment.split("=", 1)
            else:
                raise ValueError(
                    f"Bad config line {lineno} in {path}: {raw.rstrip()!r}\n"
                    "Expected format: key: value or key = value"
                )

            key = key.strip().lower()
            value = value.strip()
            cfg[key] = value

    return cfg


# ==================================================
# Value parser
# ==================================================

def get_value(cfg, *keys, default=None, required=False):
    """Return the value of the first key present and non-empty in cfg.

    Keys are tried left to right, so the canonical name goes first and any
    accepted aliases follow:  get_value(cfg, "cpu_workers", "workers")

    cfg:      dict from load_input_file.
    keys:     one or more key names, most-canonical first.
    default:  returned when no key matches.
    required: raise ValueError instead of returning default.
    """
    for key in keys:
        k = key.lower()
        if k in cfg and cfg[k] != "":
            return cfg[k]

    if required:
        joined = " / ".join(keys)
        raise ValueError(f"Missing required config value: {joined}")

    return default


def as_bool(value, default=False):
    """Parse an on/off config value into a bool.

    True:  1, true, t, yes, y, on
    False: 0, false, f, no, n, off

    Anything else raises rather than being guessed at, so a typo like
    "fitting: onn" fails loudly instead of silently turning the analysis off.

    default applies only to a missing or empty value, not an invalid one.
    """
    if value is None or value == "":
        return default

    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise ValueError(f"Expected boolean value, got: {value!r}")


def as_list(value):
    """Split a config value into a list of tokens.

    Commas, semicolons, and spaces are interchangeable separators, so
    "TAG DAG", "TAG,DAG", and "TAG;DAG" all give ["TAG", "DAG"].

    Because spaces separate, individual entries cannot contain spaces.
    That is fine for the residue and bead names this parses (res_name,
    atom_name, neutral) and is why those options take short tokens only.

    Empty or missing values give an empty list, never None.
    """
    if value is None or str(value).strip() == "":
        return []

    # Normalize every separator to a comma, then split once.
    text = str(value).replace(";", ",").replace(" ", ",")
    return [x.strip() for x in text.split(",") if x.strip()]


def cfg_path(value, base_dir):
    """Resolve a path from the input file against the input file's folder.

    value:    path string from the config, or empty.
    base_dir: folder containing the input file.
    Returns:  a Path, or None if value was empty or missing.

    Callers must handle None -- this does not raise on a missing value.
    """
    if value is None or str(value).strip() == "":
        return None

    p = Path(str(value).strip())
    if not p.is_absolute():
        p = base_dir / p
    return p


