import json
import argparse
from collections import defaultdict

def load_events(path):
    with open(path, "r") as f:
        data = json.load(f)
    # ORT profiles are usually a list of dict events
    if isinstance(data, dict) and "traceEvents" in data:
        # chrome trace format
        return data["traceEvents"]
    if isinstance(data, list):
        return data
    raise ValueError("Unrecognized ORT profile JSON format")

def get_str(d, *keys, default=""):
    for k in keys:
        if k in d:
            return d[k]
    return default

def get_dur_us(ev):
    # chrome trace often uses "dur" in microseconds
    if "dur" in ev:
        return float(ev["dur"])
    # sometimes stored in args
    args = ev.get("args", {})
    for k in ("dur", "duration", "dur_us"):
        if k in args:
            return float(args[k])
    return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", help="Path to ORT profile JSON")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--filter", type=str, default="", help="Substring filter on name/category")
    ap.add_argument("--output", type=str, default=None, help="Write summary to file instead of stdout")
    args = ap.parse_args()

    events = load_events(args.profile)

    by_name = defaultdict(float)
    by_cat = defaultdict(float)
    count_by_name = defaultdict(int)

    total = 0.0
    kept = 0

    for ev in events:
        name = get_str(ev, "name")
        cat = get_str(ev, "cat")
        dur = get_dur_us(ev)

        if dur <= 0:
            continue

        # filter to meaningful execution events; keep broad by default
        # common cats: "Node", "Kernel", "Session", "InferenceSession"
        if args.filter:
            hay = (name + " " + cat).lower()
            if args.filter.lower() not in hay:
                continue

        total += dur
        kept += 1
        by_name[name] += dur
        by_cat[cat] += dur
        count_by_name[name] += 1

    def fmt(us):  # microseconds -> ms
        return us / 1000.0

    # Build output as string
    output_lines = []
    output_lines.append(f"Profile: {args.profile}")
    output_lines.append(f"Events used: {kept}")
    output_lines.append(f"Total (dur sum): {fmt(total):.2f} ms")
    output_lines.append("")
    output_lines.append("Top categories by time:")
    for cat, us in sorted(by_cat.items(), key=lambda x: x[1], reverse=True)[:10]:
        output_lines.append(f"  {cat or '(none)'}: {fmt(us):.2f} ms")
    output_lines.append("")
    output_lines.append(f"Top {args.top} names by time:")
    for name, us in sorted(by_name.items(), key=lambda x: x[1], reverse=True)[:args.top]:
        output_lines.append(f"  {name[:120]:120s}  {fmt(us):9.2f} ms  ({count_by_name[name]} events)")
    
    output_text = "\n".join(output_lines)
    
    # Write to file or stdout
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"Summary written to: {args.output}")
    else:
        print(output_text)

if __name__ == "__main__":
    main()
