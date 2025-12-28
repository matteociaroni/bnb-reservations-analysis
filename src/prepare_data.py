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
        net_amount_col: str = "Net amount",
        nights_col: str = "Room nights",
        origin_col: str = "Origin",
        status_col: str = "Status",
        valid_status: str = "OK"
) -> pd.DataFrame:
    """
    Build a daily booking fact table.

    One row = one occupied night.
    Includes calendar fields, nightly revenue, and 'occupied' flag.
    """
    df = df.copy()
    df[arrival_col] = pd.to_datetime(df[arrival_col])
    df[departure_col] = pd.to_datetime(df[departure_col])

    df = df[(df[status_col] == valid_status) & (df[nights_col] > 0)]

    records = []

    for _, row in df.iterrows():
        nightly_revenue = row[net_amount_col] / row[nights_col]
        days = pd.date_range(start=row[arrival_col], end=row[departure_col] - pd.Timedelta(days=1))

        for day in days:
            records.append({
                "date": day,
                "year": day.year,
                "month": day.month,
                "day": day.day,
                "origin": row[origin_col],
                "price_per_night": nightly_revenue,
                "occupied": 1
            })

    return pd.DataFrame(records)

def compute_price_per_night(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute nightly price, avoiding division by zero.
    """
    df = df.copy()
    df["price_per_night"] = (
        df["Net amount"]
        .div(df["Room nights"])
        .fillna(0)
    )
    return df

def build_platform_breakdown(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-platform dictionary for each year/month.
    """
    platform_agg = (
        daily_df
        .groupby(["year", "month", "origin"])
        .agg(
            nights=("occupied", "sum"),
            avg_price=("price_per_night", "mean")
        )
        .reset_index()
    )

    platform_dict = (
        platform_agg
        .groupby(["year", "month"])
        .apply(
            lambda g: {
                row["origin"]: {
                    "nights": int(row["nights"]),
                    "avg_price": float(row["avg_price"])
                }
                for _, row in g.iterrows()
            }
        )
        .reset_index(name="by_platform")
    )

    return platform_dict


def build_monthly_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly occupancy and revenue summary,
    only including months with actual data.
    """
    # aggregate by year and month
    monthly_agg = (
        daily_df
        .groupby(["year", "month"])
        .agg(
            occupied_nights=("occupied", "sum"),
            avg_price=("price_per_night", "mean")
        )
        .reset_index()
    )

    # compute days in month
    monthly_agg["days_in_month"] = monthly_agg.apply(
        lambda r: pd.Period(f"{int(r.year)}-{int(r.month):02d}").days_in_month,
        axis=1
    )

    # compute occupancy percentage
    monthly_agg["occupancy_pct"] = (
        monthly_agg["occupied_nights"]
        .div(monthly_agg["days_in_month"])
        .mul(100)
        .round(2)
    )

    # compute monthly revenue
    monthly_agg["monthly_revenue"] = (
            monthly_agg["occupied_nights"] * monthly_agg["avg_price"]
    )

    return monthly_agg


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
platform_summary = build_platform_breakdown(daily_occupancy)

monthly_summary = monthly_summary.merge(
    platform_summary,
    on=["year", "month"],
    how="left"
)

monthly_summary["by_platform"] = monthly_summary["by_platform"].apply(
    lambda x: x if isinstance(x, dict) else {}
)

monthly_summary.to_csv("../output/monthly_summary.csv", index=False)
daily_occupancy.to_csv("../output/daily_bookings.csv", index=False)