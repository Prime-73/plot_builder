import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit  

def parse_array_text(text: str) -> np.ndarray:
    if not text or not text.strip():
        return np.array([])
    cleaned = (
        text.replace("[", " ")
            .replace("]", " ")
            .replace("(", " ")
            .replace(")", " ")
            .replace("\\n", " ")
            .replace("\\t", " ")
            .replace(";", " ")
    )
    parts = [p for p in cleaned.replace(",", " ").split(" ") if p.strip()]
    try:
        arr = np.array([float(x) for x in parts], dtype=float)
    except ValueError:
        import ast
        try:
            lit = ast.literal_eval(text)
            arr = np.array(lit, dtype=float).ravel()
        except Exception:
            st.error("Could not parse the pasted array.")
            return np.array([])
    return arr.ravel()


def color_swatch_html(name: str, hexval: str) -> str:
    return f'''
    <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
      <div style="width:18px;height:18px;border-radius:4px;border:1px solid #ccc;background:{hexval};"></div>
      <code style="font-size:0.85rem;">{name}</code>
    </div>
    '''


def get_named_colors():
    colors = dict(mcolors.CSS4_COLORS)
    colors.update(mcolors.TABLEAU_COLORS)
    return colors


def linfunc(x, m, b):
    return m * x + b


st.set_page_config(page_title="Plot Builder", layout="wide")
st.title("Simple Plot Builder")

left, right = st.columns([1, 1])

with left:
    st.subheader("1) Data Input")
    source_type = st.radio("Choose data source", ["CSV file", "Pasted arrays"], horizontal=True)
    df = None
    x = None
    y_arrays = []
    xerr_arrays = []
    yerr_arrays = []
    csv_cols = []

    if source_type == "CSV file":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file is not None:
            try:
                df = pd.read_csv(file)
                csv_cols = df.columns.tolist()
                st.caption("Preview:")
                st.dataframe(df.head(12), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

        if df is not None and len(csv_cols) >= 2:
            xcol = st.selectbox("X column", options=csv_cols, index=0)
            ycols = st.multiselect("Y columns", options=[c for c in csv_cols if c != xcol])
            for yc in ycols:
                y_arrays.append(df[yc].to_numpy(dtype=float))
                xe_col = st.selectbox(f"xerr column for {yc} (optional)", options=["(none)"] + csv_cols, index=0)
                ye_col = st.selectbox(f"yerr column for {yc} (optional)", options=["(none)"] + csv_cols, index=0)
                xerr_arrays.append(df[xe_col].to_numpy(dtype=float) if xe_col != "(none)" else None)
                yerr_arrays.append(df[ye_col].to_numpy(dtype=float) if ye_col != "(none)" else None)
            x = df[xcol].to_numpy(dtype=float)

    else:
        st.write("Paste arrays below. Provide one X array and one or more Y arrays.")
        x_text = st.text_area("X array", value="1, 2, 3, 4, 5")
        num_series = st.number_input("How many Y series?", min_value=1, max_value=10, value=1, step=1)
        for i in range(int(num_series)):
            y_text = st.text_area(f"Y array #{i+1}", value="1, 4, 9, 16, 25" if i == 0 else "")
            y_arrays.append(parse_array_text(y_text))
            xe_text = st.text_area(f"xerr for series #{i+1} (array or single value)", value="")
            ye_text = st.text_area(f"yerr for series #{i+1} (array or single value)", value="")
            arr_xe = parse_array_text(xe_text) if xe_text.strip() else np.array([])
            arr_ye = parse_array_text(ye_text) if ye_text.strip() else np.array([])
            if arr_xe.size == 1:
                arr_xe = np.full_like(parse_array_text(x_text), arr_xe.item())
            if arr_ye.size == 1:
                arr_ye = np.full_like(parse_array_text(x_text), arr_ye.item())
            xerr_arrays.append(arr_xe if arr_xe.size else None)
            yerr_arrays.append(arr_ye if arr_ye.size else None)
        x = parse_array_text(x_text)

with right:
    st.subheader("2) Plot Settings")

    scale = st.selectbox("Axis scale", ["linear", "logarithmic x", "logarithmic y", "both logarithmic"])

    title = st.text_input("Title", value="")
    xlabel = st.text_input("X label", value="x")
    ylabel = st.text_input("Y label", value="y")
    legend_enabled = st.checkbox("Enable legend", value=True)

    named_colors = get_named_colors()
    with st.expander("Show named colors"):
        cols = st.columns(4)
        items = list(named_colors.items())
        for i, (nm, hx) in enumerate(items):
            col = cols[i % 4]
            col.markdown(color_swatch_html(nm, hx), unsafe_allow_html=True)

    st.subheader("Series customization")
    series_settings = []
    for i in range(len(y_arrays)):
        st.markdown(f"**Series {i+1} settings**")
        name = st.text_input(f"Name for series {i+1}", value=f"series_{i+1}")
        color = st.selectbox(f"Color for series {i+1}", options=list(named_colors.keys()), index=10)
        fit_enabled = st.checkbox(f"Enable linear fit for series {i+1}", value=False)
        fit_color = None
        if fit_enabled:
            fit_color = st.selectbox(f"Fit line color for series {i+1}", options=list(named_colors.keys()), index=9)
        series_settings.append({"name": name, "color": color, "fit": fit_enabled, "fit_color": fit_color})

    st.subheader("3) Plot Preview")
    if x is not None and len(x) > 0 and any(len(y) > 0 for y in y_arrays):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        code_lines = [
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "from scipy.optimize import curve_fit",
            "",
            f"x = np.array({x.tolist()})"
        ]

        for i, y in enumerate(y_arrays):
            yname = f"y{i+1}"
            code_lines.append(f"{yname} = np.array({y.tolist()})")
            label = series_settings[i]["name"]
            color_choice = series_settings[i]["color"]
            xe = xerr_arrays[i]
            ye = yerr_arrays[i]

            if xe is not None or ye is not None:
                ax.errorbar(x, y, yerr=ye, xerr=xe, fmt='o', label=label, color=named_colors[color_choice])
                if ye is not None:
                    code_lines.append(f"{yname}_err = np.array({ye.tolist()})")
                if xe is not None:
                    code_lines.append(f"{yname}_xerr = np.array({xe.tolist()})")
                code_lines.append(f"plt.errorbar(x, {yname}, yerr={yname+"_err" if ye is not None else 'None'}, xerr={yname+"_xerr" if xe is not None else 'None'}, fmt='o', label='{label}', color='{color_choice}')")
            else:
                ax.scatter(x, y, label=label, color=named_colors[color_choice])
                code_lines.append(f"plt.scatter(x, {yname}, label='{label}', color='{color_choice}')")

            if series_settings[i]["fit"]:
                try:
                    code_lines.append("def linfunc(m, x, b): return m*x + b")
                    if ye is not None:
                        popt, pcov = curve_fit(linfunc, x, y, sigma=ye, absolute_sigma=True)
                        code_lines.append(f"popt, pcov = curve_fit(linfunc, x, {yname}, sigma={yname}_err, absolute_sigma=True)")
                    else:
                        popt, pcov = curve_fit(linfunc, x, y)
                        code_lines.append(f"popt, pcov = curve_fit(linfunc, x, {yname})")
                    m, b = popt
                    perr = np.sqrt(np.diag(pcov))
                    xi = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                    yi = linfunc(xi, m, b)
                    ax.plot(xi, yi, '-', label=f"Fit: {label}", color=named_colors[series_settings[i]["fit_color"]])
                    code_lines.extend([
                        "m, b = popt",
                        "perr = np.sqrt(np.diag(pcov))",
                        "xi = np.linspace(np.nanmin(x), np.nanmax(x), 200)",
                        "yi = linfunc(xi, m, b)",
                        f"plt.plot(xi, yi, '-', label=f'Fit: {label}', color='{named_colors[series_settings[i]["fit_color"]]}')"
                    ])
                    st.markdown(f"**Fit results for {label}:** slope = {m:.3g} ± {perr[0]:.2g}, intercept = {b:.3g} ± {perr[1]:.2g}")
                except Exception as e:
                    st.error(f"Fit failed for {label}: {e}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        if title:
            ax.set_title(title)
            code_lines.append(f"plt.title('{title}')")
        if legend_enabled:
            ax.legend()
            code_lines.append("plt.legend()")
        if scale == "logarithmic x" or scale == "both logarithmic":
            ax.set_xscale("log")
            code_lines.append("plt.xscale('log')")
        if scale == "logarithmic y" or scale == "both logarithmic":
            ax.set_yscale("log")
            code_lines.append("plt.yscale('log')")

        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        code_lines.append(f"plt.xlabel('{xlabel}')")
        code_lines.append(f"plt.ylabel('{ylabel}')")
        code_lines.append("plt.grid(True, which='both', linestyle='--', alpha=0.3)")
        code_lines.append("plt.show()")

        code_text = "\n".join(code_lines)
        st.subheader("Generated Python code")
        st.text_area("Python code", value=code_text, height=400)
            
        # JavaScript for the copy button
        copy_js = f"""
        <script>
        function copyToClipboard() {{
            navigator.clipboard.writeText(`{code_text}`);
            alert('✅ Copied code to clipboard!');
        }}
        </script>

        <button
            style="
                background-color: #f0f2f6;
                color: black;
                border: 1px solid #d3d6db;
                border-radius: 4px;
                padding: 0.25em 0.75em;
                font-size: 14px;
                cursor: pointer;
            "
            onclick="copyToClipboard()"
        >
            📋 Copy to Clipboard
        </button>
        """

        # Render the button
        st.components.v1.html(copy_js, height=50)
        st.download_button("Download Python code", data=code_text, file_name="plot_code.py", mime="text/x-python")

        