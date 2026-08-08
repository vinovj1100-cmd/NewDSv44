"""Report Generation Engine v4.3
Generates PDF, Excel, CSV reports with charts, headers, watermarks.
Uses fpdf2 + matplotlib. Streamlit download-ready.
"""
import io
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
from typing import Dict, List, Optional

from db import get_inventory_full, get_orders, get_labor_summary

class ReportGenerator:
    def generate_executive_pdf(self, title: str = "Warehouse Operations Report") -> bytes:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        # FIX: use custom family name, not "Helvetica"
        pdf.add_font("DejaVu", "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=16)
        pdf.cell(200, 10, title, ln=True, align='C')
        pdf.set_font("DejaVu", size=10)
        pdf.cell(200, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
        pdf.ln(10)

        # Inventory Summary
        inv = get_inventory_full()
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(200, 10, "Inventory Overview", ln=True)
        pdf.set_font("Helvetica", size=9)
        pdf.cell(60, 8, "SKU")
        pdf.cell(40, 8, "Product")
        pdf.cell(20, 8, "Stock")
        pdf.cell(20, 8, "Class")
        pdf.cell(30, 8, "Location")
        pdf.ln()
        for _, row in inv.head(20).iterrows():
            pdf.cell(60, 6, str(row["sku"]))
            pdf.cell(40, 6, str(row["product"] or ""))
            pdf.cell(20, 6, str(row["stock"]))
            pdf.cell(20, 6, str(row["abc_class"]))
            pdf.cell(30, 6, str(row["location"] or ""))
            pdf.ln()

        # Chart
        fig, ax = plt.subplots(figsize=(4, 2.5))
        if not inv.empty:
            inv["stock"].plot(kind="hist", bins=15, ax=ax, color="#00b4db", edgecolor="black")
        ax.set_title("Stock Distribution")
        ax.set_xlabel("Units")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        pdf.image(buf, x=20, y=pdf.get_y()+5, w=170)
        plt.close()

        buf_pdf = io.BytesIO()
        pdf.output(buf_pdf)
        return buf_pdf.getvalue()

    def generate_labor_csv(self) -> bytes:
        df = get_labor_summary()
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        return buf.getvalue()

    def generate_forecast_excel(self, forecast_data: Dict) -> bytes:
        df = pd.DataFrame({
            "Date": forecast_data["dates"],
            "Forecast": forecast_data["forecast"],
            "Lower_CI": forecast_data["confidence_lower"],
            "Upper_CI": forecast_data["confidence_upper"]
        })
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Forecast", index=False)
        return buf.getvalue()
