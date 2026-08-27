"""Half-block ANSI rendering of the praximetry mark, for the CLI startup banner.

_ICON is a 6-row half-block (each row = 2 image-pixel rows via fg/bg color)
rendering of the logo icon, downsampled to match the ansi_shadow wordmark's
6-line height -- a terminal can't scale character height the way the artifact's
CSS font-size did, so matching row-count is the only way to keep the icon and
wordmark proportioned like the approved mockup. Uses only the plain half-block
character (`\u2580`) rather than quadrant block glyphs -- those aren't in the
CP437-derived block-element set most terminal fonts actually ship, and render
as tofu. Regenerate from the SVG if the mark changes; not meant to be hand-edited.
"""

from __future__ import annotations

import pyfiglet

WHITE = (233, 238, 243)
GREEN = (44, 181, 82)

_ICON = [
    "\x1b[38;2;13;17;23;48;2;13;17;23m▀\x1b[38;2;13;17;23;48;2;13;17;23m▀\x1b[38;2;13;17;23;48;2;13;18;24m▀\x1b[38;2;13;19;24;48;2;12;11;21m▀\x1b[38;2;12;9;20;48;2;18;45;33m▀\x1b[38;2;24;74;43;48;2;45;183;83m▀\x1b[38;2;40;160;75;48;2;45;186;84m▀\x1b[38;2;40;160;75;48;2;45;186;84m▀\x1b[38;2;24;74;43;48;2;45;183;83m▀\x1b[38;2;12;9;20;48;2;18;45;33m▀\x1b[38;2;13;19;24;48;2;12;11;21m▀\x1b[38;2;13;17;23;48;2;13;18;24m▀\x1b[38;2;13;17;23;48;2;13;17;23m▀\x1b[38;2;13;17;23;48;2;13;17;23m▀\x1b[0m",
    "\x1b[38;2;13;17;23;48;2;13;17;23m▀\x1b[38;2;13;17;23;48;2;13;17;23m▀\x1b[38;2;13;20;24;48;2;13;18;23m▀\x1b[38;2;11;5;18;48;2;13;13;22m▀\x1b[38;2;26;82;47;48;2;15;38;29m▀\x1b[38;2;49;196;89;48;2;32;168;69m▀\x1b[38;2;44;175;80;48;2;44;190;84m▀\x1b[38;2;44;175;80;48;2;44;190;84m▀\x1b[38;2;49;196;89;48;2;32;168;69m▀\x1b[38;2;26;82;47;48;2;15;38;29m▀\x1b[38;2;11;5;19;48;2;13;13;22m▀\x1b[38;2;13;20;24;48;2;13;18;23m▀\x1b[38;2;13;17;23;48;2;13;17;23m▀\x1b[38;2;13;17;23;48;2;13;17;23m▀\x1b[0m",
    "\x1b[38;2;13;17;23;48;2;13;17;23m▀\x1b[38;2;13;17;23;48;2;13;17;23m▀\x1b[38;2;15;18;24;48;2;17;21;27m▀\x1b[38;2;8;12;17;48;2;0;0;5m▀\x1b[38;2;34;32;43;48;2;170;176;180m▀\x1b[38;2;170;207;187;48;2;185;185;194m▀\x1b[38;2;45;145;75;48;2;5;1;13m▀\x1b[38;2;45;145;75;48;2;5;1;13m▀\x1b[38;2;170;207;187;48;2;185;184;194m▀\x1b[38;2;34;32;43;48;2;170;176;181m▀\x1b[38;2;8;12;17;48;2;0;0;5m▀\x1b[38;2;15;18;24;48;2;17;21;27m▀\x1b[38;2;13;17;23;48;2;13;17;23m▀\x1b[38;2;13;17;23;48;2;13;17;23m▀\x1b[0m",
    "\x1b[38;2;15;19;25;48;2;5;9;15m▀\x1b[38;2;16;20;26;48;2;7;11;17m▀\x1b[38;2;0;0;4;48;2;33;36;42m▀\x1b[38;2;87;91;97;48;2;223;228;233m▀\x1b[38;2;233;238;243;48;2;100;104;110m▀\x1b[38;2;27;32;36;48;2;0;0;6m▀\x1b[38;2;8;13;18;48;2;17;21;27m▀\x1b[38;2;8;13;18;48;2;17;21;27m▀\x1b[38;2;27;32;36;48;2;0;0;6m▀\x1b[38;2;233;238;243;48;2;100;104;110m▀\x1b[38;2;87;91;97;48;2;223;228;233m▀\x1b[38;2;0;0;4;48;2;33;36;42m▀\x1b[38;2;16;20;26;48;2;7;11;17m▀\x1b[38;2;15;19;25;48;2;5;9;15m▀\x1b[0m",
    "\x1b[38;2;64;68;74;48;2;201;206;211m▀\x1b[38;2;192;197;202;48;2;197;201;207m▀\x1b[38;2;232;237;241;48;2;226;231;237m▀\x1b[38;2;184;189;194;48;2;205;210;215m▀\x1b[38;2;1;5;11;48;2;28;32;38m▀\x1b[38;2;15;19;25;48;2;10;14;20m▀\x1b[38;2;13;17;23;48;2;14;18;24m▀\x1b[38;2;13;17;23;48;2;14;18;24m▀\x1b[38;2;15;19;26;48;2;10;14;20m▀\x1b[38;2;1;5;11;48;2;28;32;38m▀\x1b[38;2;184;189;194;48;2;205;210;215m▀\x1b[38;2;232;237;241;48;2;226;231;237m▀\x1b[38;2;192;197;202;48;2;197;201;207m▀\x1b[38;2;64;68;74;48;2;201;206;211m▀\x1b[0m",
    "\x1b[38;2;193;198;203;48;2;54;57;63m▀\x1b[38;2;187;192;197;48;2;191;195;201m▀\x1b[38;2;178;182;187;48;2;207;212;217m▀\x1b[38;2;214;219;224;48;2;80;84;90m▀\x1b[38;2;24;28;34;48;2;3;6;12m▀\x1b[38;2;10;14;20;48;2;15;19;25m▀\x1b[38;2;14;18;24;48;2;13;17;23m▀\x1b[38;2;14;18;24;48;2;13;17;23m▀\x1b[38;2;10;14;20;48;2;15;19;25m▀\x1b[38;2;24;28;34;48;2;3;6;12m▀\x1b[38;2;214;219;224;48;2;80;84;90m▀\x1b[38;2;177;182;187;48;2;207;212;217m▀\x1b[38;2;187;192;197;48;2;191;196;201m▀\x1b[38;2;193;198;203;48;2;54;57;63m▀\x1b[0m",
]


def _ansi_fg(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"\x1b[38;2;{r};{g};{b}m"


def render() -> str:
    """PRAXI/METRY wordmark (white/green) with the logo icon to its left."""
    praxi = (
        pyfiglet.Figlet(font="ansi_shadow", width=200).renderText("PRAXI").rstrip("\n").split("\n")
    )
    metry = (
        pyfiglet.Figlet(font="ansi_shadow", width=200).renderText("METRY").rstrip("\n").split("\n")
    )
    praxi = [line for line in praxi if line.strip()]
    metry = [line for line in metry if line.strip()]

    reset = "\x1b[0m"
    lines = []
    for icon_row, p, m in zip(_ICON, praxi, metry):
        lines.append(f"{icon_row}  {_ansi_fg(WHITE)}{p}{_ansi_fg(GREEN)}{m}{reset}")
    return "\n".join(lines)
