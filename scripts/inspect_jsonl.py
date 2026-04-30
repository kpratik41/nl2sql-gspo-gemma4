import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= args.n:
                break

            obj = json.loads(line)
            print("=" * 100)
            print(f"Example {idx}")
            print("Keys:", obj.keys())

            messages = obj.get("messages", [])
            print("Num messages:", len(messages))

            for m in messages:
                print(f"\nROLE: {m.get('role')}")
                print(str(m.get("content", ""))[:1000])

            print("\ndb_id:", obj.get("db_id"))
            print("gold_sql:", obj.get("gold_sql") or obj.get("query") or obj.get("sql"))


if __name__ == "__main__":
    main()