def normalize(str_input):
    """Trim whitespace and forward the cleaned string for further processing."""
    str2 = str_input.strip()
    return forward(str2)


def forward(str_input):
    """Pass a cleaned string along to the next stage of the pipeline."""
    return str_input.lower()


def check_locale(s):
    """Check whether the given string starts with the en-US locale prefix."""
    pre = s[0:5]
    return pre == "en-US"


def perf(str_input):
    """Run a couple of performance-related actions on the input string."""
    if act("A", str_input):
        return act("B", str_input)
    return None


def act(kind, str_input):
    return f"{kind}:{str_input}"
