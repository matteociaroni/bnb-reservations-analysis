from pathlib import Path
import numpy as np
import pandas as pd

BOOKING_FEE = 0.165 # This is calculated from the original amount
AIRBNB_FEE = 0.03 # This is calculated from the original amount
STATE_FEE = 0.21 # This is calculated from the original amount
VAT_TAX = 0.22 # This is calculated from the platform fees

def read_booking_files(directory: Path) -> pd.DataFrame:
    """
    Read all Booking.com CSVs from a directory and compute net amount.
    """
    csv_files = directory.glob("*.csv")
    df_list = [pd.read_csv(f) for f in csv_files]
    booking_df = pd.concat(df_list, ignore_index=True)

    booking_df["Booked on"] = pd.to_datetime(booking_df["Booked on"])
    booking_df["Origin"] = "Booking"

    booking_df["Original amount"] = pd.to_numeric(booking_df["Original amount"])
    booking_df["Net amount"] = booking_df["Original amount"] * (1 - BOOKING_FEE * (1 + VAT_TAX) - STATE_FEE)

    columns = [
        'Reservation number', 'Booked on', 'Arrival', 'Departure',
        'Guest name', 'Persons', 'Room nights', 'Net amount', 'Status', 'Origin'
    ]
    return booking_df[columns]


def preprocess_airbnb(file_path: str) -> pd.DataFrame:
    """
    Read Airbnb CSV and transform to standard format.
    """
    df = pd.read_csv(file_path)
    df["Persons"] = df["N. di adulti"] + df["N. di bambini"] + df["N. di neonati"]
    df["Arrival"] = pd.to_datetime(df["Data di inizio"], format="%d/%m/%Y")
    df["Departure"] = pd.to_datetime(df["Data di fine"], format="%d/%m/%Y")

    # Clean revenue column
    df["Revenue"] = (
        df["Guadagni"]
        .str.replace("\xa0", "", regex=False)  # remove non-breaking space
        .str.replace("€", "", regex=False)    # remove euro symbol
        .str.replace(",", ".", regex=False)   # convert comma to dot
        .str.strip()                          # remove any leading/trailing whitespace
        .astype(float)
    )

    df["Net amount"] = df["Revenue"] * (1 - AIRBNB_FEE * (1 + VAT_TAX) - STATE_FEE)

    df["Status"] = df["Stato"].replace({
        "Ospite precedente": "OK",
        "Cancellata dall'ospite": "CANCELLED"
    })
    df = df[df["Status"] != "Confermata"]
    df["Origin"] = "AirBnB"

    columns = [
        'Codice di conferma', 'Prenotata', 'Arrival', 'Departure',
        "Nome dell'ospite", 'Persons', 'N. di notti', 'Net amount', 'Status', "Origin"
    ]
    df = df[columns]
    df.columns = ['Reservation number', 'Booked on', 'Arrival', 'Departure',
                  'Guest name', 'Persons', 'Room nights', 'Net amount', 'Status', 'Origin']
    return df

def preprocess_direct(file_path: str) -> pd.DataFrame:
    """
    Read direct bookings CSV and standardize format.
    """
    df = pd.read_csv(file_path)
    df["Origin"] = "Direct"
    df["Net amount"] = df["Original amount"] * (1 - STATE_FEE)
    df["Arrival"] = pd.to_datetime(df["Arrival"], format="%d/%m/%Y")
    df["Departure"] = pd.to_datetime(df["Departure"], format="%d/%m/%Y")
    return df

def merge_all_bookings(booking: pd.DataFrame, airbnb: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate all sources and compute derived columns.
    """
    total = pd.concat([booking, airbnb, direct], ignore_index=True)

    # Ensure datetime
    for col in ["Booked on", "Arrival", "Departure"]:
        total[col] = pd.to_datetime(total[col])

    # Derived columns
    total["Days before"] = (total["Arrival"] - total["Booked on"]).dt.days
    total["Average price"] = total["Net amount"] / total["Room nights"]
    return total

def build_daily_booking_fact(
        df: pd.DataFrame,
        arrival_col: str = "Arrival",
        departure_col: str = "Departure",
        booked_on_col: str = "Booked on",
        days_before_col: str = "Days before",
        net_amount_col: str = "Net amount",
        nights_col: str = "Room nights",
        origin_col: str = "Origin",
        persons_col: str = "Persons",
        status_col: str = "Status",
        valid_status: str = "OK"
) -> pd.DataFrame:
    """
    Build a daily booking fact table.
    """
    df = df.copy()
    df = df[(df[status_col] == valid_status) & (df[nights_col] > 0)]

    records = []
    for _, row in df.iterrows():
        nightly_revenue = row[net_amount_col] / row[nights_col]
        persons = row[persons_col]

        days = pd.date_range(
            start=row[arrival_col],
            end=row[departure_col] - pd.Timedelta(days=1)
        )

        for i, day in enumerate(days):
            is_checkin = (i == 0)
            records.append({
                "date": day,
                "year": day.year,
                "month": day.month,
                "day": day.day,
                "origin": row[origin_col],
                "price_per_night": nightly_revenue,
                "occupied": persons,
                "booked_on": row[booked_on_col] if is_checkin else pd.NaT,
                "days_before": row[days_before_col] if is_checkin else np.nan
            })

    return pd.DataFrame(records)

def build_platform_breakdown(daily_df: pd.DataFrame, group_cols=["year", "month"]) -> pd.DataFrame:
    """
    Build per-platform metrics for each grouping (year/month or year).
    Added avg_persons and revenue per platform.
    """
    platform_agg = (
        daily_df
        .groupby(group_cols + ["origin"])
        .agg(
            nights=("date", "count"),
            persons_nights=("occupied", "sum"),
            revenue=("price_per_night", "sum"),
            avg_price=("price_per_night", "mean"),
            avg_lead_time=("days_before", "mean")
        )
        .reset_index()
    )

    # Calculate average persons per platform
    platform_agg["avg_persons"] = (platform_agg["persons_nights"] / platform_agg["nights"]).round(2)

    platform_dict = (
        platform_agg
        .groupby(group_cols)
        .apply(
            lambda g: {
                row["origin"]: {
                    "nights": int(row["nights"]),
                    "persons_nights": int(row["persons_nights"]),
                    "avg_price": float(row["avg_price"]),
                    "avg_persons": float(row["avg_persons"]),
                    "revenue": float(row["revenue"]),
                    "avg_lead_time": round(float(row["avg_lead_time"]), 2) if not pd.isna(row["avg_lead_time"]) else 0
                }
                for _, row in g.iterrows()
            }
        )
        .reset_index(name="by_platform")
    )
    return platform_dict


def build_monthly_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly occupancy and revenue summary.
    """
    monthly_agg = (
        daily_df
        .groupby(["year", "month"])
        .agg(
            occupied_nights=("occupied", "count"),
            total_persons=("occupied", "sum"),
            avg_price=("price_per_night", "mean"),
            avg_lead_time=("days_before", "mean") # Calcolo media giorni anticipo
        )
        .reset_index()
    )

    monthly_agg["days_in_month"] = monthly_agg.apply(
        lambda r: pd.Period(f"{int(r.year)}-{int(r.month):02d}").days_in_month,
        axis=1
    )

    monthly_agg["occupancy_pct"] = (monthly_agg["occupied_nights"] / monthly_agg["days_in_month"] * 100).round(2)
    monthly_agg["avg_persons"] = monthly_agg["total_persons"] / monthly_agg["occupied_nights"]
    monthly_agg["monthly_revenue"] = monthly_agg["occupied_nights"] * monthly_agg["avg_price"]

    return monthly_agg

def build_yearly_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build yearly occupancy and revenue summary with same logic as monthly.
    """
    yearly_agg = (
        daily_df
        .groupby(["year"])
        .agg(
            occupied_nights=("occupied", "count"),
            total_persons=("occupied", "sum"),
            avg_price=("price_per_night", "mean"),
            avg_lead_time=("days_before", "mean") # Calcolo media giorni anticipo
        )
        .reset_index()
    )

    # Calculate days in year (handles leap years)
    yearly_agg["days_in_year"] = yearly_agg["year"].apply(
        lambda y: 366 if (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0) else 365
    )

    yearly_agg["occupancy_pct"] = (yearly_agg["occupied_nights"] / yearly_agg["days_in_year"] * 100).round(2)
    yearly_agg["avg_persons"] = yearly_agg["total_persons"] / yearly_agg["occupied_nights"]
    yearly_agg["yearly_revenue"] = yearly_agg["occupied_nights"] * yearly_agg["avg_price"]

    return yearly_agg

# =========================
# Execution
# =========================
airbnb_df = preprocess_airbnb("../input/reservations.csv")
booking_df = read_booking_files(Path("../input/booking"))
airbnb_df.columns = booking_df.columns
direct_df = preprocess_direct("../input/no_platform.csv")
total_df = merge_all_bookings(booking_df, airbnb_df, direct_df)

# keep only confirmed bookings
confirmed = total_df[total_df["Status"] == "OK"].copy()
daily_occupancy = build_daily_booking_fact(confirmed)

monthly_summary = build_monthly_summary(daily_occupancy)
monthly_summary = monthly_summary.merge(build_platform_breakdown(daily_occupancy, ["year", "month"]), on=["year", "month"], how="left")

yearly_summary = build_yearly_summary(daily_occupancy)
yearly_summary = yearly_summary.merge(build_platform_breakdown(daily_occupancy, ["year"]), on=["year"], how="left")

# --- OUTPUT ---
monthly_summary.to_csv("../output/monthly_summary.csv", index=False)
yearly_summary.to_csv("../output/yearly_summary.csv", index=False)
daily_occupancy.to_csv("../output/daily_bookings.csv", index=False)