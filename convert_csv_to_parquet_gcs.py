import argparse
from io import BytesIO

import pandas as pd
from google.cloud import storage


DEFAULT_BUCKET = "acpe-dev-uc-ml-promociones"
DEFAULT_BLOBS = (
    "tmp/bo_sp.csv",
    "tmp/bo_cp.csv",
    "tmp/as.csv",
    "tmp/cmp.csv",
    "tmp/escalamiento_promociones_ticket_cmp.csv",
    "tmp/escalamiento_promociones_ticket_b2b.csv",
)


def csv_to_parquet_blob_name(csv_blob_name):
    return csv_blob_name.rsplit(".", 1)[0] + ".parquet"


def convert_blob(bucket, csv_blob_name):
    csv_blob = bucket.blob(csv_blob_name)
    if not csv_blob.exists():
        print(f"[SKIP] No existe: gs://{bucket.name}/{csv_blob_name}")
        return

    print(f"[READ] gs://{bucket.name}/{csv_blob_name}")
    csv_bytes = csv_blob.download_as_bytes()
    df = pd.read_csv(BytesIO(csv_bytes))

    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False, compression="snappy")
    parquet_buffer.seek(0)

    parquet_blob_name = csv_to_parquet_blob_name(csv_blob_name)
    parquet_blob = bucket.blob(parquet_blob_name)
    parquet_blob.upload_from_file(parquet_buffer, content_type="application/octet-stream")
    print(f"[OK] gs://{bucket.name}/{parquet_blob_name}")


def main():
    parser = argparse.ArgumentParser(description="Convierte CSV de GCS a Parquet en el mismo bucket.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="Nombre del bucket GCS.")
    parser.add_argument(
        "--blobs",
        nargs="*",
        default=list(DEFAULT_BLOBS),
        help="Rutas de blobs CSV a convertir (ejemplo: tmp/bo_cp.csv).",
    )
    args = parser.parse_args()

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    for blob_name in args.blobs:
        try:
            convert_blob(bucket, blob_name)
        except Exception as exc:
            print(f"[ERROR] {blob_name}: {exc}")


if __name__ == "__main__":
    main()

