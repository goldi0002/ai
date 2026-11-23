"""The CLI module must import and expose every documented subcommand."""

from runtime.cli import build_parser

DOCUMENTED = [
    ["info"],
    ["generate", "hi"],
    ["chat"],
    ["research", "stats"],
    ["memory", "list"],
    ["tools"],
    ["benchmark"],
    ["quantize"],
    ["eval-tools"],
]


def test_cli_imports_and_parses_documented_commands():
    parser = build_parser()
    for argv in DOCUMENTED:
        assert parser.parse_args(argv).func is not None
