from pathlib import Path
import pandas as pd
import numpy as np

# ===== アイムジャグラーの定数 =====
COIN_IN_PER_GAME = 3
BIG_PAYOUT       = 252
REG_PAYOUT       = 96

REPLAY_PROB    = 1 / 7.298
REPLAY_PAYOUT  = 3

CHERRY_PROB    = 1 / 53.43
CHERRY_PAYOUT  = 2

BELL_PROB      = 1 / 4915.20
BELL_PAYOUT    = 14

PIERROT_PROB   = 1 / 10865.18
PIERROT_PAYOUT = 10

GRAPE_PAYOUT   = 8


def calc_grape_prob_denom(row):
    """
    1行ぶんから「ぶどう確率の分母」（例: 6.15 → 1/6.15）を計算して返す。
    数値に変換できない行や、計算不可能な行は np.nan を返す。
    """
    # 安全に数値変換（変換できないものは NaN）
    spins = pd.to_numeric(row.get("G数"),  errors="coerce")
    bb    = pd.to_numeric(row.get("BB"),   errors="coerce")
    rb    = pd.to_numeric(row.get("RB"),   errors="coerce")
    diff  = pd.to_numeric(row.get("差枚"), errors="coerce")

    # どれか欠損 or G数が0以下なら計算しない
    if (
        pd.isna(spins) or pd.isna(bb) or pd.isna(rb) or pd.isna(diff) or
        spins <= 0
    ):
        return np.nan

    total_in  = spins * COIN_IN_PER_GAME
    total_out = total_in + diff

    # ボーナス払い出し
    bonus_out       = bb * BIG_PAYOUT + rb * REG_PAYOUT
    # 小役トータル払い出し
    small_out_total = total_out - bonus_out

    # ぶどう以外の小役払い出し（理論値）
    other_small_out = spins * (
        REPLAY_PROB   * REPLAY_PAYOUT +
        CHERRY_PROB   * CHERRY_PAYOUT +
        BELL_PROB     * BELL_PAYOUT   +
        PIERROT_PROB  * PIERROT_PAYOUT
    )

    grape_out = small_out_total - other_small_out
    if grape_out <= 0:
        return np.nan

    grape_hits = grape_out / GRAPE_PAYOUT
    if grape_hits <= 0:
        return np.nan

    grape_prob = spins / grape_hits   # 1/◯ の ◯
    return grape_prob


def process_file(in_path: Path, out_dir: Path):
    """1つのCSVファイルに対して、ブドウ率列を追加したCSVを出力する"""
    print(f"処理中: {in_path.name}")

    # CSV読み込み
    df = pd.read_csv(in_path, encoding="utf-8-sig")

    # ぶどう確率の分母(例: 6.15)を計算
    denoms = df.apply(calc_grape_prob_denom, axis=1)

    # 表示用に "1/6.15" という文字列に変換
    budo_rate_col = denoms.map(
        lambda x: f"1/{x:.2f}" if pd.notna(x) else ""
    )

    # RB率の右隣に「ブドウ率」列を追加したい
    try:
        idx = df.columns.get_loc("RB率") + 1
    except KeyError:
        # RB率 列が無い場合は、とりあえず末尾に追加
        idx = len(df.columns)

    df.insert(idx, "ブドウ率", budo_rate_col)

    # 出力先ディレクトリを作成しておく
    out_dir.mkdir(parents=True, exist_ok=True)

    # 出力ファイルパス（result配下）
    out_path = out_dir / (in_path.stem + "_with_grape" + in_path.suffix)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"  → 出力: {out_path.name}")


def main():
    # ★ 入力元ディレクトリ（data）
    in_dir  = Path(r"C:\Users\akafu\analytics\aim_machi\data")
    # ★ 出力先ディレクトリ（notebook\result）
    out_dir = Path(r"C:\Users\akafu\analytics\aim_machi\notebook\data2")

    # data ディレクトリ内の aim-*.csv をすべて対象にする
    for in_path in in_dir.glob("aim-*.csv"):
        # すでに _with_grape が付いているファイルはスキップ
        if in_path.name.endswith("_with_grape.csv"):
            continue
        process_file(in_path, out_dir)


if __name__ == "__main__":
    main()
