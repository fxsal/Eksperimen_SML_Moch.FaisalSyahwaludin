import pandas as pd
import numpy as np
import argparse
import os

def preprocess_to_daily(df_in):
    df = df_in.copy()

    # Hapus kolom yang tidak diperlukan
    drop_cols = [c for c in ['Loud Cover', 'Daily Summary'] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # ---- FIX datetime ----
    # Tangani mixed timezone dan pastikan kolom bertipe datetime
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], errors='coerce', utc=True)
    df['Formatted Date'] = df['Formatted Date'].dt.tz_convert(None)

    # Hapus baris kosong dan urutkan
    df = df.dropna(subset=['Formatted Date'])
    df = df.sort_values('Formatted Date')

    # Tambahkan kolom tanggal (tanpa jam)
    df['Date'] = df['Formatted Date'].dt.date

    # ---- Agregasi harian ----
    num_cols = [
        'Temperature (C)',
        'Apparent Temperature (C)',
        'Humidity',
        'Wind Speed (km/h)',
        'Wind Bearing (degrees)',
        'Visibility (km)',
        'Pressure (millibars)'
    ]
    agg_dict = {c: 'mean' for c in num_cols if c in df.columns}
    daily = df.groupby('Date').agg(agg_dict).reset_index()

    # Kolom kategori (ambil modus harian)
    if 'Precip Type' in df.columns:
        precip = df.groupby('Date')['Precip Type'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ).reset_index()
        daily = daily.merge(precip, on='Date', how='left')

    if 'Summary' in df.columns:
        summary = df.groupby('Date')['Summary'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ).reset_index()
        daily = daily.merge(summary, on='Date', how='left')

    # Isi nilai kosong
    daily = daily.fillna(method='ffill').fillna(method='bfill')

    # ---- Fitur tambahan: lag dan rolling ----
    daily = daily.copy()
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.sort_values('Date').reset_index(drop=True)

    target = 'Temperature (C)'
    if target in daily.columns:
        for lag in range(1, 8):
            daily[f'{target}_lag_{lag}'] = daily[target].shift(lag)
        daily['temp_roll7_mean'] = daily[target].rolling(window=7).mean()
        daily['temp_roll7_std'] = daily[target].rolling(window=7).std()

    # Hapus baris awal yang NaN karena lag/rolling
    daily = daily.dropna().reset_index(drop=True)

    return daily


def main(input_path, output_path):
    # Pastikan folder output ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Reading input file: {input_path}")
    df = pd.read_csv(input_path)

    print("Starting preprocessing ...")
    processed = preprocess_to_daily(df)

    processed.to_csv(output_path, index=False)
    print(f"Preprocessing selesai. Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess weather dataset")
    parser.add_argument("--input", type=str, required=True, help="path to raw csv")
    parser.add_argument("--output", type=str, required=True, help="path to save processed csv")
    args = parser.parse_args()

    main(args.input, args.output)