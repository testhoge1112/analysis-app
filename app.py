import os
import glob
import re
import argparse
import datetime
import pandas as pd
import numpy as np

def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    """
    指定された重み（稼働G数など）に基づいて加重平均を計算する。
    単なる平均ではなく、よく回っている台のデータを正当に評価するために使用。
    """
    total_w = weights.sum()
    if total_w == 0 or series.empty:
        return 0.0
    return np.average(series, weights=weights)

def frac_to_float(x: str) -> float:
    """ '1/120' のような分数形式の文字列を小数（確率）に変換する """
    try:
        num, den = str(x).split('/')
        return float(num) / float(den)
    except Exception:
        return 0.0

def load_csv_with_conversion(fp: str) -> pd.DataFrame:
    """
    CSVファイルを読み込み、文字列として取得された数値データ（%, 分数など）を
    Pandasで計算可能な数値型（float/int）にクレンジング（整形）する。
    """
    cols = ['台番', '差枚', 'G数', '出率', 'BB', 'RB', '合成', 'BB率', 'RB率']
    encodings = ['utf-8-sig', 'utf-8', 'cp932']
    
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(fp, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        df = pd.read_csv(fp, encoding='utf-8', errors='replace')

    for col in cols:
        if col not in df.columns:
            continue
        
        # カンマ等の不要な文字を削除
        s = df[col].astype(str).str.strip().str.replace(',', '', regex=False)
        
        if col == '出率':
            s = s.str.replace('%', '', regex=False)
            df[col] = pd.to_numeric(s, errors='coerce').fillna(0) / 100.0
        elif col in ['合成', 'BB率', 'RB率']:
            df[col] = s.apply(frac_to_float)
        else:
            df[col] = pd.to_numeric(s, errors='coerce').fillna(0)
            
    return df

def filter_files(files: list, args, start_date, end_date, parity, month_parity) -> list:
    """
    指定された各種条件（期間、曜日、偶奇など）に基づいて、
    対象となるCSVファイルを1回のループで効率的に絞り込む。
    """
    filtered = []
    target_days = [] if args.dom == ['all'] else [int(d) for d in args.dom]

    for fp in files:
        fn = os.path.basename(fp)
        # 正規表現でファイル名から日付と曜日を安全に抽出 (例: aim-2025-06-13-Fri.csv)
        m = re.match(r'aim-(\d{4})-(\d{2})-(\d{2})-([A-Za-z]{3})', fn)
        if not m:
            continue
            
        y, m_str, d_str, dow = m.groups()
        file_date = datetime.date(int(y), int(m_str), int(d_str))
        
        # 1. 期間フィルタ
        if start_date and end_date and not (start_date <= file_date <= end_date):
            continue
                
        # 2. 日付偶奇フィルタ
        is_even_day = (file_date.day % 2 == 0)
        if parity != 'all' and ((parity == 'even' and not is_even_day) or (parity == 'odd' and is_even_day)):
            continue

        # 3. 日付番号(dom)フィルタ
        if target_days and (file_date.day not in target_days):
            continue

        # 4. 月偶奇フィルタ
        is_even_month = (file_date.month % 2 == 0)
        if month_parity != 'all' and ((month_parity == 'even' and not is_even_month) or (month_parity == 'odd' and is_even_month)):
            continue

        # 5. 曜日フィルタ
        if args.day != 'all' and dow != args.day:
            continue

        filtered.append(fp)
        
    return filtered

def aggregate_data(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    全データを台番ごとにグループ化し、差枚やG数は合計を、
    確率・出率関連はG数を重みとした加重平均を算出する。
    """
    df_all['台番'] = pd.to_numeric(df_all['台番'], errors='coerce').fillna(0).astype(int)
    df_all = df_all[df_all['台番'] > 0]

    sum_cols = ['G数', 'BB', 'RB', '差枚']
    frac_cols = ['出率', '合成', 'BB率', 'RB率']

    records = []
    # Pandasのgroupbyを使って台番ごとに効率よく集計
    for machine, grp in df_all.groupby('台番'):
        rec = {'台番': machine}
        
        # 単純な合計値
        for col in sum_cols:
            rec[col] = grp[col].sum() if col in grp.columns else 0
            
        # 加重平均（G数を重みとする）
        for col in frac_cols:
            if col not in grp.columns or 'G数' not in grp.columns:
                rec[col] = '0'
                continue
                
            avg_val = weighted_mean(grp[col], grp['G数'])
            
            if col == '出率':
                rec[col] = f"{avg_val * 100:.1f}%"
            else:
                # 分析結果を人が見やすい確率表記 (1/xxx) に戻す
                rec[col] = f"1/{int(round(1 / avg_val))}" if avg_val > 0 else '0'
                
        records.append(rec)

    return pd.DataFrame(records).sort_values('台番').reset_index(drop=True)

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='台番別集計スクリプト')
    parser.add_argument('--day', choices=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'all'], default='all')
    parser.add_argument('--dom', nargs='+', choices=[*map(str, range(1, 32)), 'all'], default=['all'])
    args = parser.parse_args()

    # --- 対話型入力セクション ---
    start = input('集計開始日 (YYYYMMDD または all): ').strip()
    start_date, end_date = None, None
    if start.lower() != 'all':
        try:
            start_date = datetime.datetime.strptime(start, '%Y%m%d').date()
            end = input('集計終了日 (YYYYMMDD): ').strip()
            end_date = datetime.datetime.strptime(end, '%Y%m%d').date()
            if end_date < start_date:
                print('エラー: 終了日は開始日以降を指定してください。')
                return
        except ValueError:
            print('エラー: フォーマットが不正です。YYYYMMDD で入力してください。')
            return

    parity = input('日付フィルタ (even/odd/all): ').strip().lower()
    month_parity = input('月フィルタ (even/odd/all): ').strip().lower()
    
    if not {parity, month_parity}.issubset({'even', 'odd', 'all'}):
        print('エラー: フィルタは even, odd, all のいずれかを入力してください。')
        return

    # --- データ読み込みと絞り込み ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    all_files = glob.glob(os.path.join(data_dir, 'aim-*.csv'))

    # 関数化してスッキリしたフィルタリング処理
    target_files = filter_files(all_files, args, start_date, end_date, parity, month_parity)

    if not target_files:
        print("条件に一致する対象ファイルがありませんでした。")
        return

    # --- 集計処理と保存 ---
    print(f"\n{len(target_files)}件のファイルを読み込み、集計を開始します...")
    dfs = [load_csv_with_conversion(fp) for fp in target_files]
    df_all = pd.concat(dfs, ignore_index=True)
    
    # 関数化してスッキリした集計処理
    df_res = aggregate_data(df_all)

    out_fp = os.path.join(base_dir, 'aggregated.csv')
    df_res.to_csv(out_fp, index=False, encoding='utf-8-sig')
    print(f"集計結果を保存しました: {out_fp}")

if __name__ == '__main__':
    main()