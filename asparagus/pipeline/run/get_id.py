import argparse
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="file to extract run_id from")
    args = parser.parse_args()

    f = open(args.file)
    f = f.read()

    pattern = r"###RUN-ID=\d+###"

    matches = re.findall(pattern, f)
    if len(matches) < 1:
        print("found 0 run-IDs")
        return

    run_id = matches[0].replace("#", "").replace("RUN-ID=", "")
    print(f"found run-ID: {run_id}")


if __name__ == "__main__":
    main()
