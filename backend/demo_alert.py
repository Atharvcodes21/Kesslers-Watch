"""Demo alert script for KesslerWatch pitch.
Sends a live high-priority collision alert directly to Telegram to demonstrate autonomous alerting.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure utf-8 output in Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent))

import sentinel

def trigger_demo_alert():
    demo_alert_data = {
        "status": "CONFIRMED GO",
        "primary_name": "ISS (ZARYA)",
        "secondary_name": "COSMOS 1408 DEB",
        "secondary_type": "DEBRIS",
        "tca_utc": datetime.now(timezone.utc).isoformat(),
        "hours_to_tca": 14.2,
        "miss_km": 0.38,
        "pc": 0.000185,
        "decision": "GO",
        "cost_of_acting": {
            "available": True,
            "dv_ms": 0.095,
        }
    }

    message = sentinel.alert_text(demo_alert_data)
    print("Sending live alert to Telegram...")
    print("---------------------------------")
    print(message)
    print("---------------------------------")
    
    success = sentinel.send_telegram(message)
    if success:
        print("✅ Alert delivered to Telegram successfully!")
    else:
        print("❌ Delivery failed. Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in backend/.env")

if __name__ == "__main__":
    trigger_demo_alert()
