from io import BytesIO
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from google.cloud import bigquery
from google.cloud import storage
from google.oauth2 import service_account


CLIENTES_PILOTOS_TABLE = "acpe-dev-uc-ml.dev.clientes_pilotos"
BOCA_SALIDA_BUCKET = "acpe-dev-uc-ml-promociones"
BO_SP_BLOB = "tmp/bo_sp.parquet"
BO_CP_BLOB = "tmp/bo_cp.parquet"
AS_BLOB = "tmp/as.parquet"
CMP_BLOB = "tmp/cmp.parquet"
TICKET_CMP_BLOB = "tmp/escalamiento_promociones_ticket_cmp.csv"
TICKET_B2B_BLOB = "tmp/escalamiento_promociones_ticket_b2b.csv"

COMMON_FILTER_COLS = (
    "nom_compania",
    "nom_sucursal",
    "des_region",
    "des_zona_venta",
    "des_oficina_venta",
    "boca_salida",
    "des_segmento_transaccional",
    "des_segmento_estrategica",
    "flg_potential",
    "negocio",
)

REQUIRED_MULTI_FILTER_COLS = (
    "boca_salida",
    "des_segmento_transaccional",
    "flg_potential",
)

EXCLUSION_FILTER_COLS = (
    "des_categoria",
    "des_familia",
    "des_marca_material",
    "tipo_promocion",
)

CLIENT_KEY_CANDIDATES = (
    "cod_cliente",
    "cod_cliente_alicorp_actual",
    "codigo_cliente",
    "cliente",
)

TICKET_COL_CANDIDATES = (
    "ticket_pedido",
    "ticket",
    "ticket_promedio",
    "monto_ticket",
    "avg_ticket",
)

_STORAGE_CLIENT = None
_BQ_CLIENT = None


def _coerce_float_cols(df_in, cols):
    df = df_in.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _bq_read_table(client, table_fqn):
    query = f"SELECT * FROM `{table_fqn}`"
    return client.query(query).to_dataframe(create_bqstorage_client=False)


def _get_gcp_credentials_and_project():
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
    creds = None

    # 1) Credenciales JSON completas en env var.
    raw_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if raw_json:
        info = json.loads(raw_json)
        creds = service_account.Credentials.from_service_account_info(info)
        project = project or info.get("project_id")
        return creds, project

    # 2) Credenciales desde Streamlit secrets.
    try:
        import streamlit as st  # import local para no forzar dependencia fuera de Streamlit

        if "gcp_service_account" in st.secrets:
            info = dict(st.secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(info)
            project = project or info.get("project_id")
            return creds, project
    except Exception:
        pass

    # 3) Fallback a Application Default Credentials.
    return None, project


def _get_storage_client():
    global _STORAGE_CLIENT
    if _STORAGE_CLIENT is not None:
        return _STORAGE_CLIENT

    creds, project = _get_gcp_credentials_and_project()
    if creds is not None:
        _STORAGE_CLIENT = storage.Client(project=project, credentials=creds)
    else:
        _STORAGE_CLIENT = storage.Client(project=project)
    return _STORAGE_CLIENT


def _get_bigquery_client():
    global _BQ_CLIENT
    if _BQ_CLIENT is not None:
        return _BQ_CLIENT

    creds, project = _get_gcp_credentials_and_project()
    if creds is not None:
        _BQ_CLIENT = bigquery.Client(project=project, credentials=creds)
    else:
        _BQ_CLIENT = bigquery.Client(project=project)
    return _BQ_CLIENT


def _read_parquet_from_gcs(bucket_name, blob_name):
    storage_client = _get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_bytes()
    return pd.read_parquet(BytesIO(content))


def _read_csv_from_gcs(bucket_name, blob_name):
    storage_client = _get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_bytes()
    return pd.read_csv(BytesIO(content))


def _read_table_from_gcs(bucket_name, blob_name):
    if blob_name.lower().endswith(".parquet"):
        return _read_parquet_from_gcs(bucket_name, blob_name)
    if blob_name.lower().endswith(".csv"):
        return _read_csv_from_gcs(bucket_name, blob_name)
    raise ValueError(f"Extension no soportada para blob: {blob_name}")


def _read_parquet_filtered_in_batches(
    bucket_name,
    blob_name,
    include_filters=None,
    exclude_filters=None,
    columns=None,
    batch_size=200_000,
):
    storage_client = _get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    frames = []
    with blob.open("rb") as fobj:
        pf = pq.ParquetFile(fobj)
        parquet_cols = set(pf.schema.names)
        use_cols = [c for c in (columns or pf.schema.names) if c in parquet_cols]

        for batch in pf.iter_batches(columns=use_cols, batch_size=batch_size):
            chunk = batch.to_pandas()
            if include_filters or exclude_filters:
                chunk = _apply_local_filters(
                    chunk,
                    include_filters=include_filters,
                    exclude_filters=exclude_filters,
                )
            if not chunk.empty:
                frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=columns if columns is not None else None)
    return pd.concat(frames, ignore_index=True)


def _apply_local_filters(df_in, include_filters=None, exclude_filters=None):
    df = df_in.copy()
    include_filters = include_filters or {}
    exclude_filters = exclude_filters or {}

    def _normalize_values_for_col(col, values):
        if col != "flg_potential":
            return {str(v).strip() for v in values}

        normalized = set()
        for v in values:
            s = str(v).strip().lower()
            if s in {"con potencial", "true", "1", "si", "sí"}:
                normalized.add(True)
                continue
            if s in {"sin potencial", "false", "0", "no"}:
                normalized.add(False)
                continue
            normalized.add(v)
        return normalized

    def _to_bool_like(value):
        s = str(value).strip().lower()
        if s in {"true", "1", "si", "sí", "con potencial"}:
            return True
        if s in {"false", "0", "no", "sin potencial"}:
            return False
        return None

    for col, values in include_filters.items():
        if not values or col not in df.columns:
            continue
        allowed = _normalize_values_for_col(col, values)
        if col == "flg_potential":
            col_values = df[col].map(_to_bool_like)
            df = df[col_values.isin(allowed)].copy()
        else:
            df = df[df[col].astype(str).str.strip().isin({str(v).strip() for v in allowed})].copy()

    for col, values in exclude_filters.items():
        if not values or col not in df.columns:
            continue
        blocked = _normalize_values_for_col(col, values)
        if col == "flg_potential":
            col_values = df[col].map(_to_bool_like)
            df = df[~col_values.isin(blocked)].copy()
        else:
            df = df[~df[col].astype(str).str.strip().isin({str(v).strip() for v in blocked})].copy()

    return df


def _norm_str(value):
    return str(value).strip().upper()


def _resolve_base_blob(boca_salida, flg_potential=None):
    boca = _norm_str(boca_salida)
    flg = _norm_str(flg_potential) if flg_potential is not None else ""

    if boca == "BO":
        if "SIN" in flg and "POT" in flg:
            return BO_SP_BLOB
        if "CON" in flg and "POT" in flg:
            return BO_CP_BLOB
        raise ValueError("Para boca_salida='BO' debes elegir flg_potential 'Con potencial' o 'Sin Potencial'.")
    if boca in {"LV", "PA", "GA"}:
        return AS_BLOB
    if boca in {"PM", "MM"}:
        return CMP_BLOB
    raise ValueError(f"boca_salida no soportada: {boca_salida}")


def _resolve_ticket_blob(boca_salida):
    boca = _norm_str(boca_salida)
    if boca in {"BO", "PM", "MM"}:
        return TICKET_CMP_BLOB
    return TICKET_B2B_BLOB


def load_filter_options_from_storage(boca_salida, flg_potential=None):
    base_blob = _resolve_base_blob(boca_salida, flg_potential)
    cols = list(dict.fromkeys([*COMMON_FILTER_COLS, *EXCLUSION_FILTER_COLS]))
    storage_client = _get_storage_client()
    bucket = storage_client.bucket(BOCA_SALIDA_BUCKET)
    blob = bucket.blob(base_blob)

    options = {}
    for col in cols:
        options[col] = []

    # Para parquet grande: leer por lotes y limitar filas para evitar error de memoria.
    if base_blob.lower().endswith(".parquet"):
        seen_by_col = {c: set() for c in cols}
        rows_seen = 0
        max_rows_for_options = 400_000

        with blob.open("rb") as fobj:
            pf = pq.ParquetFile(fobj)
            for batch in pf.iter_batches(columns=cols, batch_size=200_000):
                bdf = batch.to_pandas()
                rows_seen += len(bdf)
                for col in cols:
                    if col not in bdf.columns:
                        continue
                    uniq = pd.unique(bdf[col].dropna())
                    for v in uniq:
                        s = str(v).strip()
                        if s:
                            seen_by_col[col].add(s)
                if rows_seen >= max_rows_for_options:
                    break

        for col in cols:
            options[col] = sorted(seen_by_col[col])
        return options

    # Fallback (csv u otro): comportamiento previo
    df = _read_table_from_gcs(BOCA_SALIDA_BUCKET, base_blob)
    for col in cols:
        if col not in df.columns:
            continue
        uniq = pd.unique(df[col].dropna())
        cleaned = []
        seen = set()
        for v in uniq:
            s = str(v).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            cleaned.append(s)
        options[col] = sorted(cleaned)
    return options


def _normalize_client_key(df_in):
    df = df_in.copy()
    for col in CLIENT_KEY_CANDIDATES:
        if col in df.columns:
            if col != "cod_cliente":
                df = df.rename(columns={col: "cod_cliente"})
            return df
    raise ValueError(
        f"No se encontro columna de cliente en dataframe. Esperadas: {CLIENT_KEY_CANDIDATES}"
    )


def _count_unique_clients(df_in):
    col = _find_column_case_insensitive(df_in, CLIENT_KEY_CANDIDATES)
    if col is None:
        return 0
    return int(df_in[col].dropna().astype(str).nunique())


def _find_column_case_insensitive(df, candidates):
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        key = str(c).strip().lower()
        if key in col_map:
            return col_map[key]
    return None


def _normalize_ticket_column(df_in):
    df = df_in.copy()
    found = _find_column_case_insensitive(df, TICKET_COL_CANDIDATES)
    if found is None:
        raise ValueError(
            f"No se encontro columna de ticket en dataframe. Esperadas: {TICKET_COL_CANDIDATES}"
        )
    if found != "ticket_pedido":
        df = df.rename(columns={found: "ticket_pedido"})
    return df


def _build_ticket_region(ticket_df_in):
    ticket_df = ticket_df_in.copy()

    if "des_region" not in ticket_df.columns:
        ticket_df["des_region"] = "SIN_REGION"

    for col in ("volumen", "venta", "inv_promo_neta"):
        if col not in ticket_df.columns:
            ticket_df[col] = 0.0

    return (
        ticket_df.groupby(["cod_cliente", "des_region"], as_index=False)
        .agg(
            {
                "ticket_pedido": "mean",
                "volumen": "sum",
                "venta": "sum",
                "inv_promo_neta": "sum",
            }
        )
    )


def load_inputs(inputs_dir="inputs", include_filters=None, exclusion_filters=None, return_debug=False):
    _ = inputs_dir  # compatibilidad con firma previa

    client = _get_bigquery_client()

    include_filters = include_filters or {}
    exclusion_filters = exclusion_filters or {}

    boca_vals = include_filters.get("boca_salida", [])
    flg_vals = include_filters.get("flg_potential", [])
    if not boca_vals:
        raise ValueError("Debe seleccionar boca_salida.")

    boca_salida = boca_vals[0]
    flg_potential = flg_vals[0] if flg_vals else None
    if _norm_str(boca_salida) == "BO" and not flg_potential:
        raise ValueError("Para boca_salida='BO' debe seleccionar flg_potential.")

    base_blob = _resolve_base_blob(boca_salida, flg_potential)
    ticket_blob = _resolve_ticket_blob(boca_salida)

    debug_data = {}

    if base_blob.lower().endswith(".parquet"):
        df = _read_parquet_filtered_in_batches(
            BOCA_SALIDA_BUCKET,
            base_blob,
            include_filters=include_filters,
            exclude_filters=exclusion_filters,
            columns=None,
            batch_size=200_000,
        )
    else:
        df = _read_table_from_gcs(BOCA_SALIDA_BUCKET, base_blob)
        df = _apply_local_filters(
            df,
            include_filters=include_filters,
            exclude_filters=exclusion_filters,
        )
    debug_data["df_loaded_head"] = df.head(5).copy()
    debug_data["df_loaded_rows"] = int(len(df))
    debug_data["df_loaded_clientes"] = _count_unique_clients(df)
    debug_data["df_filtered_head"] = df.head(5).copy()
    debug_data["df_filtered_rows"] = int(len(df))
    debug_data["df_filtered_clientes"] = _count_unique_clients(df)

    ticket_pedido = _read_table_from_gcs(BOCA_SALIDA_BUCKET, ticket_blob)
    debug_data["ticket_loaded_head"] = ticket_pedido.head(5).copy()
    ticket_pedido = _apply_local_filters(
        ticket_pedido,
        include_filters=include_filters,
        exclude_filters=None,
    )
    debug_data["ticket_filtered_head"] = ticket_pedido.head(5).copy()
    clientes_pilotos = _bq_read_table(client, CLIENTES_PILOTOS_TABLE)
    debug_data["clientes_pilotos_head"] = clientes_pilotos.head(5).copy()

    df = _normalize_client_key(df)
    ticket_pedido = _normalize_client_key(ticket_pedido)
    ticket_pedido = _normalize_ticket_column(ticket_pedido)
    clientes_pilotos = _normalize_client_key(clientes_pilotos)

    clientes_excluir = set(clientes_pilotos["cod_cliente"].dropna().astype(str))
    if clientes_excluir:
        df = df[~df["cod_cliente"].astype(str).isin(clientes_excluir)].copy()
        ticket_pedido = ticket_pedido[
            ~ticket_pedido["cod_cliente"].astype(str).isin(clientes_excluir)
        ].copy()
    debug_data["df_post_pilotos_head"] = df.head(5).copy()
    debug_data["ticket_post_pilotos_head"] = ticket_pedido.head(5).copy()
    debug_data["df_post_pilotos_clientes"] = int(df["cod_cliente"].nunique()) if "cod_cliente" in df.columns else 0

    df = _coerce_float_cols(df, ["inv_promo_neta", "volumen", "venta", "ticket_pedido"])
    ticket_pedido = _coerce_float_cols(ticket_pedido, ["inv_promo_neta", "volumen", "venta", "ticket_pedido"])

    ticket_region = _build_ticket_region(ticket_pedido)
    debug_data["ticket_region_head"] = ticket_region.head(5).copy()

    if return_debug:
        return df, ticket_pedido, ticket_region, debug_data
    return df, ticket_pedido, ticket_region

def _add_decil_monto(ticket_region):
    # Cortes de 10 en 10: 0?10, 10?20, ... 190?200, y >200
    bins = list(range(-1, 201, 10)) + [np.inf]  # [0,10,20,...,200,inf)
    labels = [f"{i}-{i+10}" for i in range(0, 200, 10)] + [">200"]

    ticket_region = ticket_region.copy()
    ticket_region["decil_monto"] = pd.cut(
        ticket_region["ticket_pedido"],
        bins=bins,
        labels=labels,
        right=False,  # incluye el l?mite izquierdo: [0,10)
        include_lowest=True,
    )
    return ticket_region

def build_base(df, ticket_region):
    ticket_region = _add_decil_monto(ticket_region)
    base = df.merge(
        ticket_region[["cod_cliente", "decil_monto"]],
        on="cod_cliente",
        how="left",
    )
    return base

# Targets (campana mensual)
targets_2026 = pd.DataFrame({
    "mes": ["2026-03","2026-04","2026-05","2026-06","2026-07","2026-08","2026-09","2026-10","2026-11","2026-12"],
    "objetivo_mensual": [102000, 105000, 108000, 108000, 120000, 123000, 126000, 126000, 135000, 147000],
})
targets_2026["objetivo_acumulado_flujo"] = targets_2026["objetivo_mensual"].cumsum()


def build_plan_ahorro_mensual_flujo(
    df,
    ticket_pedido,  # <-- periodo, cliente, ticket_pedido

    # filtros dinámicos
    marcas_excluir=None,
    marcas_incluir=None,
    df_filter=None,  # function(df)->mask (opcional)

    # columnas base
    col_cliente="cod_cliente",
    col_marca="des_marca_material",
    col_periodo="periodo",
    col_decil="decil_monto",
    col_inversion="inv_promo_neta",  # positivo = ahorro; negativo = 0

    # columnas de negocio (para venta/volumen)
    col_venta="venta",
    col_volumen="volumen",

    # ticket por periodo/cliente
    col_ticket="ticket_pedido",
    col_ticket_cliente=None,   # si ticket_df usa otro nombre de cliente
    col_ticket_periodo=None,   # si ticket_df usa otro nombre de periodo

    # lógica meses activos
    meses_activos_solo_si_ahorro=True,

    targets=targets_2026
):
    d = df.copy()

    # filtro adicional
    if df_filter is not None:
        d = d.loc[df_filter(d)].copy()

    # filtros de marca
    if marcas_incluir is not None:
        d = d[d[col_marca].isin(marcas_incluir)]
    if marcas_excluir is not None:
        d = d[~d[col_marca].isin(marcas_excluir)]

    # normaliza periodo a YYYYMM
    d[col_periodo] = d[col_periodo].astype(str).str.replace("-", "").str[:6].astype(int)

    # ahorro: mantiene positivos y negativos
    inv = pd.to_numeric(d[col_inversion], errors="coerce").fillna(0.0)
    d["_ahorro_2025"] = inv

    # venta/volumen
    d["_venta"] = pd.to_numeric(d[col_venta], errors="coerce").fillna(0.0)
    d["_volumen"] = pd.to_numeric(d[col_volumen], errors="coerce").fillna(0.0)

    # -------------------------
    # Ticket por periodo-cliente
    # -------------------------
    tdf = ticket_pedido.copy()

    # renombres si vienen distinto
    if col_ticket_cliente is not None and col_ticket_cliente != col_cliente:
        tdf = tdf.rename(columns={col_ticket_cliente: col_cliente})
    if col_ticket_periodo is not None and col_ticket_periodo != col_periodo:
        tdf = tdf.rename(columns={col_ticket_periodo: col_periodo})

    tdf[col_periodo] = tdf[col_periodo].astype(str).str.replace("-", "").str[:6].astype(int)
    tdf["_ticket"] = pd.to_numeric(tdf[col_ticket], errors="coerce")

    # si hay duplicados por cliente-periodo, promediamos (por seguridad)
    tdf = (tdf.groupby([col_periodo, col_cliente], as_index=False)
             .agg(_ticket=("_ticket", "mean")))

    # -------------------------
    # meses activos por cliente (para runrate por cliente)
    # -------------------------
    base_meses = d  # <-- en vez de d[d["_ahorro_2025"] > 0]
    meses_activos = (base_meses.groupby(col_cliente)[col_periodo]
                           .nunique()
                           .rename("meses_activos"))

    # decil único por cliente
    decil_cli = d.groupby(col_cliente)[col_decil].max().rename("decil_monto")

    # total ahorro 2025 por cliente
    by_cli = (d.groupby(col_cliente, as_index=False)
                .agg(ahorro_total_2025=("_ahorro_2025", "sum"))
             ).merge(meses_activos, on=col_cliente, how="left") \
              .merge(decil_cli, on=col_cliente, how="left")

    by_cli["meses_activos"] = by_cli["meses_activos"].fillna(0).astype(int)

    # promedio mensual del cliente (run-rate por cliente)
    by_cli["runrate_mensual_cliente"] = np.where(
        by_cli["meses_activos"] > 0,
        by_cli["ahorro_total_2025"] / by_cli["meses_activos"],
        0.0
    )

    # orden: pequeños en compra primero (decil)
    by_cli = by_cli.sort_values(["decil_monto", col_cliente], ascending=[True, True]).reset_index(drop=True)

    # Totales del universo (para porcentajes)
    total_venta_universo = d["_venta"].sum()
    total_vol_universo = d["_volumen"].sum()

    # --- rollout ---
    selected = set()
    ahorro_mes_total = 0.0
    idx = 0
    n_total = len(by_cli)

    plan_rows = []
    for _, t in targets.iterrows():
        mes_str = t["mes"]           # '2026-03'
        mes_yyyymm = int(mes_str.replace("-", ""))  # 202603

        objetivo_mes = float(t["objetivo_mensual"])

        nuevos = []
        ahorro_nuevo_mes = 0.0

        while ahorro_mes_total + 1e-9 < objetivo_mes and idx < n_total:
            cli = by_cli.loc[idx, col_cliente]
            rr  = float(by_cli.loc[idx, "runrate_mensual_cliente"])
            idx += 1

            if rr <= 0:
                continue

            if cli not in selected:
                selected.add(cli)
                ahorro_mes_total += rr
                ahorro_nuevo_mes += rr
                nuevos.append(cli)

        # ---- venta/volumen de la tanda y activos (OJO: esto está como TOTAL 2025 del set) ----
        if len(nuevos) > 0:
            venta_tanda = d.loc[d[col_cliente].isin(nuevos), "_venta"].sum()
            vol_tanda   = d.loc[d[col_cliente].isin(nuevos), "_volumen"].sum()
        else:
            venta_tanda = 0.0
            vol_tanda   = 0.0

        if len(selected) > 0:
            venta_activos = d.loc[d[col_cliente].isin(selected), "_venta"].sum()
            vol_activos   = d.loc[d[col_cliente].isin(selected), "_volumen"].sum()
        else:
            venta_activos = 0.0
            vol_activos   = 0.0

        # -------------------------
        # NUEVO: ticket promedio por MES de la tanda
        # (toma ticket de esos clientes en el mes de la tanda)
        # -------------------------
        if len(nuevos) > 0:
            ticket_tanda_mes = tdf.loc[(tdf[col_cliente].isin(nuevos)), "_ticket"].mean()
        else:
            ticket_tanda_mes = np.nan

        # (Opcional útil) ticket promedio por MES de activos (clientes ya sin promo a ese mes)
        if len(selected) > 0:
            ticket_activos_mes = tdf.loc[(tdf[col_cliente].isin(selected)),"_ticket"].mean()
        else:
            ticket_activos_mes = np.nan

        plan_rows.append({
            "mes": mes_str,
           # "periodo_yyyymm": mes_yyyymm,
            "objetivo_mensual": objetivo_mes,

            # ahorro (flujo mensual con base activa)
            "ahorro_mes_total": ahorro_mes_total,
            "ahorro_nuevo_mes": ahorro_nuevo_mes,

            "bodegas_nuevas_mes": len(nuevos),
            "bodegas_acum": len(selected),
            "%_bodegas_acum": (len(selected)/n_total) if n_total else 0.0,

            # listas
            "clientes_nuevos_lista": nuevos,
            "clientes_activos_lista": sorted(selected),

           # # métricas de negocio (TOTAL 2025 del set)
           # "venta_tanda_total_2025": venta_tanda,
           # "venta_tanda_%_universo": (venta_tanda / total_venta_universo) if total_venta_universo else 0.0,
           # "vol_tanda_total_2025": vol_tanda,
           # "vol_tanda_%_universo": (vol_tanda / total_vol_universo) if total_vol_universo else 0.0,
#
           # "venta_activos_total_2025": venta_activos,
           # "venta_activos_%_universo": (venta_activos / total_venta_universo) if total_venta_universo else 0.0,
           # "vol_activos_total_2025": vol_activos,
           # "vol_activos_%_universo": (vol_activos / total_vol_universo) if total_vol_universo else 0.0,
#
           # # TICKET por mes (lo que pediste)
           # "ticket_tanda_mes": ticket_tanda_mes,
           # "ticket_activos_mes": ticket_activos_mes,  # opcional pero suele servir
        })

    plan = pd.DataFrame(plan_rows)

    # validación vs objetivo mensual
    plan["gap_vs_obj_mensual"] = plan["objetivo_mensual"] - plan["ahorro_mes_total"]
    plan["cumple_obj_mensual"] = plan["gap_vs_obj_mensual"] <= 0

    # acumulado Mar–Dic como flujo
    plan["ahorro_acumulado_mar_dic"] = plan["ahorro_mes_total"].cumsum()

    seleccionadas = by_cli[by_cli[col_cliente].isin(selected)].copy()

    resumen_decil = (seleccionadas.groupby("decil_monto", as_index=False, observed=False)
                       .agg(
                           bodegas=("decil_monto", "size"),
                           runrate=("runrate_mensual_cliente", "sum"),
                           ahorro_total_2025=("ahorro_total_2025", "sum")
                       )
                     ).sort_values("decil_monto")

    return plan, seleccionadas, resumen_decil, by_cli


if __name__ == "__main__":
    df, ticket_pedido, ticket_region = load_inputs()
    base = build_base(df, ticket_region)

    plan, sel, resumen, by_cli = build_plan_ahorro_mensual_flujo(
        base,
        ticket_pedido=ticket_pedido,
        col_venta="venta",
        col_volumen="volumen",
        col_ticket="ticket_pedido",
    )

    #print(plan.head())
