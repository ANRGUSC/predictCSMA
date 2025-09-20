# filter_zero_rows.py
import argparse, ast, pathlib
import pandas as pd

def keep_row(th_list):
    """Return True if at least ONE entry in the list is non-zero."""
    # ast.literal_eval converts the string "[0, 0, …]" to a real Python list
    return any(float(x) != 0 for x in ast.literal_eval(th_list))

def main(src, dst):
    df = pd.read_csv(src)

    # ---- filter ----------------------------------------------------------
    mask = df['saturation_throughput'].map(keep_row)
    cleaned = df.loc[mask].reset_index(drop=True)      # <- no SettingWithCopy warning

    # ---- (optional) serialize lists back to strings ----------------------
    list_cols = ['adj_matrix', 'transmission_prob', 'saturation_throughput']
    cleaned.loc[:, list_cols] = cleaned[list_cols].applymap(lambda x: str(x))

    # ---- save ------------------------------------------------------------
    cleaned.to_csv(dst, index=False)
    print(f"{src}  →  {dst}")
    print(f"kept {mask.sum()}/{len(mask)} rows ({mask.mean():.1%} non-zero)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv_in",  help="original file (e.g., graphs_20nodes.csv)")
    p.add_argument("-o", "--out",
                   help="output file; default adds _filtered before extension")
    args = p.parse_args()

    src_path = pathlib.Path(args.csv_in)
    dst_path = pathlib.Path(
        args.out if args.out else src_path.with_name(src_path.stem + "_filtered2.csv")
    )

    main(src_path, dst_path)
