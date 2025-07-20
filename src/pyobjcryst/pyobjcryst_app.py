import argparse

from pyobjcryst.version import __version__  # noqa


def main():
    parser = argparse.ArgumentParser(
        prog="pyobjcryst",
        description=(
            "Python bindings to the ObjCryst++ library.\n\n"
            "For more information, visit: "
            "https://github.com/diffpy/pyobjcryst/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the program's version number and exit",
    )

    args = parser.parse_args()

    if args.version:
        print(f"pyobjcryst {__version__}")
    else:
        # Default behavior when no arguments are given
        parser.print_help()


if __name__ == "__main__":
    main()
