import pandas as pd
import streamlit as st

from calculadora import (
    COMMON_FILTER_COLS,
    EXCLUSION_FILTER_COLS,
    REQUIRED_MULTI_FILTER_COLS,
    build_base,
    build_plan_ahorro_mensual_flujo,
    load_filter_options_from_storage,
    load_inputs,
    targets_2026,
)


st.set_page_config(page_title="Calculadora Promos", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --brand-1: #0f766e;
      --brand-2: #1d4ed8;
      --bg-soft: #f4f7fb;
      --card: #ffffff;
      --text-main: #0f172a;
      --text-soft: #475569;
      --line: #dbe4f0;
    }
    .stApp {
      background: radial-gradient(1200px 500px at 5% -10%, #e7f6f4 0%, transparent 45%),
                  radial-gradient(1200px 500px at 95% -10%, #e8efff 0%, transparent 45%),
                  var(--bg-soft);
    }
    .block-container {
      padding-top: 1.4rem;
      padding-bottom: 2rem;
    }
    h1, h2, h3 {
      color: var(--text-main);
      letter-spacing: 0.1px;
    }
    [data-testid="stMetric"] {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
    }
    [data-testid="stMetricLabel"] p {
      color: var(--text-soft);
      font-weight: 600;
    }
    [data-testid="stMetricValue"] {
      color: var(--brand-2);
      font-weight: 700;
    }
    div[data-testid="stExpander"] {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--card);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_data(include_filters_key, exclusion_filters_key):
    include_filters = {k: list(v) for k, v in include_filters_key}
    exclusion_filters = {k: list(v) for k, v in exclusion_filters_key}
    df, ticket_pedido, ticket_region, debug_data = load_inputs(
        include_filters=include_filters,
        exclusion_filters=exclusion_filters,
        return_debug=True,
    )
    base = build_base(df, ticket_region)
    debug_data["base_head"] = base.head(5).copy()
    return base, ticket_pedido, ticket_region, debug_data


def load_options(boca_salida, flg_potential):
    return load_filter_options_from_storage(boca_salida, flg_potential)


def _pretty_label(col_name):
    custom_labels = {
        "nom_compania": "DEX",
        "nom_sucursal": "Sucursal",
        "des_region": "Region",
    }
    if col_name in custom_labels:
        return custom_labels[col_name]

    label = col_name.strip()
    if label.startswith("des_"):
        label = label[4:]
    return label.replace("_", " ").strip().title()


def _render_multiselect_grid(cols, filter_options, key_prefix, required_cols=None, n_cols=3):
    selected_map = {}
    missing_required = []
    required_cols = required_cols or set()
    grid_cols = st.columns(n_cols)

    for i, col in enumerate(cols):
        is_required = col in required_cols
        with grid_cols[i % n_cols]:
            selected = st.multiselect(
                f"{_pretty_label(col)}{' *' if is_required else ''}",
                options=filter_options.get(col, []),
                default=[],
                key=f"{key_prefix}_{col}",
            )
        if selected:
            selected_map[col] = selected
        elif is_required:
            missing_required.append(col)
    return selected_map, missing_required


def _build_targets_for_months(meses_n):
    base = targets_2026[["mes", "objetivo_mensual"]].copy().reset_index(drop=True)
    if meses_n <= len(base):
        return base.tail(meses_n).reset_index(drop=True)

    extra = meses_n - len(base)
    last_period = pd.Period(base.iloc[-1]["mes"], freq="M")
    last_obj = float(base.iloc[-1]["objetivo_mensual"])
    extra_rows = []
    for i in range(1, extra + 1):
        p = last_period + i
        extra_rows.append({"mes": str(p), "objetivo_mensual": last_obj})

    return pd.concat([base, pd.DataFrame(extra_rows)], ignore_index=True).tail(meses_n).reset_index(drop=True)


def _style_numbers(df):
    if df.empty:
        return df
    fmt = {}
    for col in df.columns:
        col_l = col.lower()
        if "%_" in col_l or col_l.startswith("pct_") or col_l.endswith("_pct"):
            fmt[col] = "{:.2%}"
        elif pd.api.types.is_numeric_dtype(df[col]):
            fmt[col] = "{:,.2f}"
    return df.style.format(fmt)


st.title("Calculadora Promos")
st.caption("Configura filtros, ajusta objetivos y genera el plan de forma guiada.")

st.subheader("Seleccion de Dataset")
ds_col1, ds_col2 = st.columns(2)
with ds_col1:
    boca_salida_sel = st.selectbox(
        "Boca Salida *",
        options=["BO", "LV", "PA", "GA", "PM", "MM"],
        index=None,
        placeholder="Selecciona boca salida",
    )
with ds_col2:
    flg_potential_sel = st.selectbox(
        "Flg Potential (obligatorio solo para BO)",
        options=["Con potencial", "Sin Potencial"],
        index=None,
        placeholder="Selecciona flg potential",
    )

include_filters = {}
required_missing = []

if not boca_salida_sel:
    required_missing.append("boca_salida")
elif boca_salida_sel == "BO" and not flg_potential_sel:
    required_missing.append("flg_potential")

if required_missing:
    exclusion_filters = {}
else:
    include_filters["boca_salida"] = [boca_salida_sel]
    if flg_potential_sel:
        include_filters["flg_potential"] = [flg_potential_sel]

    with st.spinner("Cargando filtros desde dataset seleccionado..."):
        filter_options = load_options(boca_salida_sel, flg_potential_sel)

    st.subheader("Filtros de Inclusion")
    include_cols = [c for c in COMMON_FILTER_COLS if c not in {"boca_salida", "flg_potential"}]
    include_required = {c for c in REQUIRED_MULTI_FILTER_COLS if c not in {"boca_salida", "flg_potential"}}
    include_selected, include_missing = _render_multiselect_grid(
        include_cols,
        filter_options,
        key_prefix="include",
        required_cols=include_required,
        n_cols=3,
    )
    include_filters.update(include_selected)
    required_missing.extend(include_missing)

    st.subheader("Filtros de Exclusion")
    exclusion_filters = {}
    exclusion_selected, _ = _render_multiselect_grid(
        list(EXCLUSION_FILTER_COLS),
        filter_options,
        key_prefix="exclude",
        required_cols=set(),
        n_cols=3,
    )
    exclusion_filters.update(exclusion_selected)

meses_n = st.number_input(
    "Cantidad de meses (ultimos)",
    min_value=1,
    max_value=12,
    value=min(12, int(len(targets_2026))),
    step=1,
)

targets_for_plan = _build_targets_for_months(int(meses_n))
target_overrides = {}
target_input_error = False

st.subheader("Objetivo Mensual (Opcional)")
st.caption("Si dejas vacio un mes, se usa el valor por defecto.")
target_cols = st.columns(3)
for i, row in targets_for_plan.iterrows():
    mes = str(row["mes"])
    default_obj = float(row["objetivo_mensual"])
    with target_cols[i % 3]:
        raw = st.text_input(
            mes,
            value="",
            placeholder=f"Default: {default_obj:,.0f}",
            key=f"target_override_{mes}",
        )
    if raw.strip():
        try:
            target_overrides[mes] = float(raw.replace(",", "").strip())
        except ValueError:
            target_input_error = True

if target_input_error:
    st.warning("Hay valores invalidos en Objetivo Mensual. Usa solo numeros.")

if target_overrides:
    targets_for_plan = targets_for_plan.copy()
    targets_for_plan["objetivo_mensual"] = targets_for_plan.apply(
        lambda r: target_overrides.get(str(r["mes"]), float(r["objetivo_mensual"])),
        axis=1,
    )
    targets_for_plan["objetivo_acumulado_flujo"] = targets_for_plan["objetivo_mensual"].cumsum()

if required_missing:
    missing_labels = ", ".join(_pretty_label(col) for col in required_missing)
    st.warning(f"Completa filtros obligatorios: {missing_labels}")

run = st.button("Generar plan", disabled=bool(required_missing or target_input_error))

if run:
    include_filters_key = tuple((k, tuple(v)) for k, v in include_filters.items())
    exclusion_filters_key = tuple((k, tuple(v)) for k, v in exclusion_filters.items())

    with st.spinner("Leyendo Storage/BigQuery y calculando plan..."):
        base, ticket_pedido, ticket_region, debug_data = load_data(
            include_filters_key,
            exclusion_filters_key,
        )
        plan, sel, resumen, by_cli = build_plan_ahorro_mensual_flujo(
            base,
            ticket_pedido=ticket_pedido,
            col_venta="venta",
            col_volumen="volumen",
            col_ticket="ticket_pedido",
            targets=targets_for_plan,
        )
    clientes_filtrados = int(debug_data.get("df_filtered_clientes", 0))
    clientes_total = int(debug_data.get("df_post_pilotos_clientes", 0))
    clientes_excluidos = max(clientes_filtrados - clientes_total, 0)

    kpi1, kpi2 = st.columns(2)
    kpi1.metric("Nro clientes excluidos", f"{clientes_excluidos:,}")
    kpi2.metric("Nro clientes total", f"{clientes_total:,}")

    st.subheader("Plan")
    st.dataframe(_style_numbers(plan), use_container_width=True, hide_index=True)

    st.subheader("Clientes por periodo")
    periodos = plan["mes"].dropna().astype(str).tolist()
    periodo_sel = st.selectbox("Periodo", options=periodos)

    clientes = []
    if periodo_sel:
        fila = plan.loc[plan["mes"] == periodo_sel, "clientes_activos_lista"]
        if not fila.empty:
            clientes = fila.iloc[0] or []

    st.caption(f"Clientes activos en {periodo_sel}: {len(clientes)}")

    periodo_yyyymm = int(periodo_sel.replace("-", "")) if periodo_sel else None
    if clientes and periodo_yyyymm is not None:
        base_fil = base.loc[base["cod_cliente"].isin(clientes)].copy()
        base_fil["periodo_yyyymm"] = (
            base_fil["periodo"].astype(str).str.replace("-", "").str[:6].astype(int)
        )
        base_fil["inv_calc"] = pd.to_numeric(base_fil["inv_promo_neta"], errors="coerce").fillna(0.0)

        # Mismo calculo que el plan: runrate mensual por cliente = ahorro_total / meses_activos.
        by_cli_view = (
            base_fil.groupby("cod_cliente", as_index=False)
            .agg(
                {
                    "inv_calc": "sum",
                    "periodo_yyyymm": "nunique",
                    "venta": "sum",
                    "volumen": "sum",
                    "decil_monto": "max",
                    "nom_compania": "max",
                    "nom_sucursal": "max",
                    "boca_salida": "max",
                    "des_segmento_transaccional": "max",
                    "flg_potential": "max",
                }
            )
            .rename(columns={"periodo_yyyymm": "meses_activos"})
        )
        by_cli_view["meses_activos"] = by_cli_view["meses_activos"].replace(0, pd.NA)
        by_cli_view["inv_promo_neta"] = (
            by_cli_view["inv_calc"] / by_cli_view["meses_activos"]
        ).fillna(0.0)
        base_prom = by_cli_view.drop(columns=["inv_calc"])

        base_prom = base_prom.merge(
            ticket_region[["cod_cliente", "ticket_pedido"]],
            on="cod_cliente",
            how="left",
        )

        clientes_df = (
            base_prom.groupby("decil_monto", as_index=False)
            .agg(
                {
                    "ticket_pedido": "mean",
                    "cod_cliente": "nunique",
                    "inv_promo_neta": "sum",
                    "venta": "sum",
                    "volumen": "sum",
                }
            )
            .rename(columns={"cod_cliente": "clientes_unicos"})
        )
        clientes_df = clientes_df.sort_values("decil_monto").reset_index(drop=True)
        clientes_df["inv_promo_neta_acum"] = clientes_df["inv_promo_neta"].cumsum()
        clientes_df = clientes_df.rename(
            columns={
                "decil_monto": "decil_ticket",
                "inv_promo_neta": "Inv. Promociones Mensual",
                "venta": "Venta (S/.)",
                "volumen": "Volumen (Kg)",
                "inv_promo_neta_acum": "Inv. Promociones Acumulada",
            }
        )
    else:
        base_prom = base.head(0)
        clientes_df = base.head(0)[["decil_monto", "inv_promo_neta", "venta", "volumen"]].rename(
            columns={
                "decil_monto": "decil_ticket",
                "inv_promo_neta": "Inv. Promociones Mensual",
                "venta": "Venta (S/.)",
                "volumen": "Volumen (Kg)",
            }
        )

    st.dataframe(_style_numbers(clientes_df), use_container_width=True, hide_index=True)

    st.subheader("Clientes por periodo (filtro ticket)")
    ticket_umbral = st.number_input(
        "Mostrar ticket_pedido menor a",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )
    clientes_df_filtrado = base_prom.loc[
        (base_prom["ticket_pedido"] >= 0) & (base_prom["ticket_pedido"] < ticket_umbral)
    ] if "ticket_pedido" in base_prom.columns else base_prom

    clientes_df_filtrado_group = (
        clientes_df_filtrado.groupby("decil_monto", as_index=False)
        .agg(
            {
                "ticket_pedido": "mean",
                "cod_cliente": "nunique",
                "inv_promo_neta": "sum",
                "venta": "sum",
                "volumen": "sum",
            }
        )
        .rename(columns={"cod_cliente": "clientes_unicos"})
    )
    clientes_df_filtrado_group = clientes_df_filtrado_group.sort_values("decil_monto").reset_index(drop=True)
    clientes_df_filtrado_group["inv_promo_neta_acum"] = clientes_df_filtrado_group["inv_promo_neta"].cumsum()
    clientes_df_filtrado_group = clientes_df_filtrado_group.rename(
        columns={
            "decil_monto": "decil_ticket",
            "inv_promo_neta": "Inv. Promociones Mensual",
            "venta": "Venta (S/.)",
            "volumen": "Volumen (Kg)",
            "inv_promo_neta_acum": "Inv. Promociones Acumulada",
        }
    )

    st.dataframe(_style_numbers(clientes_df_filtrado_group), use_container_width=True, hide_index=True)
