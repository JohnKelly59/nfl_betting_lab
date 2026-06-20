import os
import pandas as pd

TRACKER_FILE = "billionbetting_tracker.csv"


def init_tracker():
    if not os.path.exists(TRACKER_FILE):
        df = pd.DataFrame(columns=[
            "Date", "Game", "Closing Line", "Model Line", "Edge", 
            "Confidence", "Tier", "Result", "ATS Result", "CLV"
        ])
        df.to_csv(TRACKER_FILE, index=False)
