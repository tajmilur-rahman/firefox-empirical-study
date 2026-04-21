from bugbug import bugzilla

def main():
    print("🔍 Counting history rows in bug data...")

    total_bugs = 0
    total_history_rows = 0
    max_bugs = None   # change to a number (e.g., 200) to test faster

    for bug in bugzilla.get_bugs():
        total_bugs += 1
        history_list = bug.get("history", [])
        total_history_rows += len(history_list)

        if max_bugs and total_bugs >= max_bugs:
            break

        # Print progress every 1000 bugs
        if total_bugs % 1000 == 0:
            print(f"🧩 Processed {total_bugs} bugs... ({total_history_rows} history rows so far)")

    print("\n✅ Counting complete!")
    print(f"📊 Total bugs scanned: {total_bugs}")
    print(f"📋 Total history rows to insert: {total_history_rows}")

if __name__ == "__main__":
    main()
