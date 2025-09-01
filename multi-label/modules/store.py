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

        # Convert lists to string representations for storage
        # For multi-label data, we'll store as JSON strings that can be parsed later
        def convert_labels(label_list):
            if isinstance(label_list, list):
                return "|".join(label_list)  # Use pipe separator for multi-labels
            return str(label_list)  # Fallback for single labels

        # Expect keys: description, purpose, polarity, emotion, delivery, domain
        table = pa.Table.from_pydict(
            {
                "description": [r.get("description") for r in valid_records],
                "purpose": [convert_labels(r.get("purpose", [])) for r in valid_records],
                "polarity": [convert_labels(r.get("polarity", [])) for r in valid_records],
                "emotion": [convert_labels(r.get("emotion", [])) for r in valid_records],
                "delivery": [convert_labels(r.get("delivery", [])) for r in valid_records],
                "domain": [convert_labels(r.get("domain", [])) for r in valid_records],
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