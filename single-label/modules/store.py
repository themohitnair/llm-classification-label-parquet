import pyarrow as pa
import pyarrow.parquet as pq
import os
import logging
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def store_to_parquet(
    records: List[Dict[str, Any]], filename: str = "output.parquet"
) -> Optional[str]:
    try:
        valid_records = [r for r in records if r is not None]

        if not valid_records:
            logger.error("No valid records to store")
            return None

        logger.info(f"Storing {len(valid_records)} records")

        # Expect keys: description, purpose, polarity, emotion, delivery, domain
        table = pa.Table.from_pydict(
            {
                "description": [r.get("description") for r in valid_records],
                "purpose": [r.get("purpose") for r in valid_records],
                "polarity": [r.get("polarity") for r in valid_records],
                "emotion": [r.get("emotion") for r in valid_records],
                "delivery": [r.get("delivery") for r in valid_records],
                "domain": [r.get("domain") for r in valid_records],
                # Add other fields as needed
            }
        )

        if os.path.exists(filename):
            os.remove(filename)

        pq.write_table(table, filename)

        logger.info(
            f"✅ Successfully stored {len(valid_records)} records to {filename}"
        )
        return filename

    except Exception as e:
        logger.error(f"Failed to store records: {e}")
        return None