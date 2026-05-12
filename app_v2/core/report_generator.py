import os
import base64
import csv
from datetime import datetime

# A simple CSS-styled SVG text fallback if ui/logo.png is missing.
FALLBACK_LOGO_SVG = """
<svg width="250" height="40" xmlns="http://www.w3.org/2000/svg">
  <text x="0" y="30" font-family="Arial, sans-serif" font-size="28" font-weight="bold" fill="#003366" letter-spacing="4">STELLANTIS</text>
</svg>
"""

def _image_to_base64(filepath):
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        ext = os.path.splitext(filepath)[1][1:].lower()
        if ext == 'jpg':
            ext = 'jpeg'
        elif ext == 'svg':
            ext = 'svg+xml'
        return f"data:image/{ext};base64,{encoded}"
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None


def _get_logo_html(base_dir):
    """Attempt to load ui/logo.svg or ui/logo.png, otherwise use fallback SVG."""
    logo_svg = os.path.join(base_dir, "ui", "logo.svg")
    logo_png = os.path.join(base_dir, "ui", "logo.png")
    
    if os.path.exists(logo_svg):
        b64 = _image_to_base64(logo_svg)
    else:
        b64 = _image_to_base64(logo_png)
        
    if b64:
        return f'<img src="{b64}" alt="Stellantis Logo" style="max-height: 60px; display: block; margin: 0 auto;" />'
    else:
        return FALLBACK_LOGO_SVG


def generate_car_report(vin, date_str, time_str, auto_result, manual_confirm, photo_path, reports_dir, base_dir):
    """
    Generates an individual HTML report for a specific inspection.
    """
    logo_html = _get_logo_html(base_dir)
    photo_b64 = _image_to_base64(photo_path)
    
    if photo_b64:
        photo_html = f'<img src="{photo_b64}" alt="Capture Photo" style="max-width: 100%; max-height: 400px; display: block; margin: 0 auto; border: 1px solid #ccc; border-radius: 4px;" />'
    else:
        photo_html = '<div style="width: 100%; height: 300px; background-color: #f0f0f0; display: flex; align-items: center; justify-content: center; color: #666;">No Photo Available</div>'

    is_auto = not manual_confirm or manual_confirm.startswith("Pending")
    if is_auto:
        is_pass = ("NO LEAK" in auto_result.upper())
        final_text = "PASS (Auto)" if is_pass else "FAIL (Auto)"
    else:
        is_pass = ("No Leak" in manual_confirm)
        final_text = "PASS" if is_pass else "FAIL"

    css_class = "match" if is_pass else "mismatch"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Inspection Report - {vin}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #ffffff;
            color: #000000;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            border: 2px solid #003366;
            padding: 40px;
            position: relative;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .title-box {{
            border: 2px solid #000;
            padding: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 16px;
        }}
        th, td {{
            border: 2px solid #000;
            padding: 12px;
            text-align: left;
        }}
        th {{
            width: 30%;
            background-color: #f8f9fa;
        }}
        .photo-section {{
            margin-top: 30px;
            border: 2px solid #000;
            padding: 20px;
            background-color: #fdfdfd;
        }}
        .result-box {{
            text-align: center;
            margin-top: 20px;
            padding: 15px;
            font-size: 24px;
            font-weight: bold;
            border: 2px solid #000;
            text-transform: uppercase;
        }}
        .match {{ background-color: #d4edda; color: #155724; }}
        .mismatch {{ background-color: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            {logo_html}
            <div class="title-box">UV LIGHT INSPECTION</div>
        </div>

        <table>
            <tr>
                <th>Department</th>
                <td><strong>Under body</strong></td>
            </tr>
            <tr>
                <th>Location</th>
                <td><strong>QCP</strong></td>
            </tr>
        </table>

        <table>
            <tr>
                <th>Vin Number</th>
                <td>{vin}</td>
            </tr>
            <tr>
                <th>Date</th>
                <td>{date_str}</td>
            </tr>
            <tr>
                <th>Time</th>
                <td>{time_str}</td>
            </tr>
            <tr>
                <th>Result</th>
                <td class="{css_class}"><strong>{final_text}</strong></td>
            </tr>
        </table>

        <div class="photo-section">
            {photo_html}
        </div>

        <div class="result-box {css_class}">
            FINAL RESULT: {final_text}
        </div>
    </div>
</body>
</html>
"""
    # Save the report
    safe_vin = "".join([c for c in vin if c.isalpha() or c.isdigit() or c in ('-', '_')]).rstrip()
    filename = f"{safe_vin}_{date_str.replace('-','')}_{time_str.replace(':','')}_report.html"
    out_path = os.path.join(reports_dir, filename)
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    return out_path


def generate_summary_report(csv_path, reports_dir, base_dir, target_date=None):
    """
    Parses the inspection_log.csv and generates a summary HTML report for target_date.
    """
    if not os.path.exists(csv_path):
        return None

    # 1. Read and deduplicate by VIN (keep last entry)
    cars_data = {}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = row.get("timestamp", "")
            if target_date and not timestamp.startswith(target_date):
                continue
            vin = row.get("vin_id", "")
            if vin:
                cars_data[vin] = row

    total_cars = len(cars_data)
    total_passed = 0
    total_failed = 0
    total_pending = 0
    
    table_rows = ""
    
    for vin, row in cars_data.items():
        timestamp = row.get("timestamp", "")
        auto_res  = row.get("auto_result", "")
        manual    = row.get("manual_label", row.get("manual_confirm", ""))

        if not manual or manual in ("—", "") or manual.startswith("Pending"):
            final_text = f"Not Verified (Auto: {auto_res})"
            total_pending += 1
            row_class = "style='background-color: #fff3cd;'" # Warning yellow
        else:
            is_pass = ("No Leak" in manual)
            final_text = "PASS" if is_pass else "FAIL"
            if is_pass:
                total_passed += 1
                row_class = "style='background-color: #d4edda;'"
            else:
                total_failed += 1
                row_class = "style='background-color: #f8d7da;'"

        table_rows += f"""
        <tr {row_class}>
            <td>{vin}</td>
            <td>{timestamp}</td>
            <td><strong>{final_text}</strong></td>
        </tr>
        """

    logo_html = _get_logo_html(base_dir)
    generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Inspection Summary Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f6f9;
            color: #333;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: #fff;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 40px;
            border-radius: 8px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-box {{
            padding: 15px;
            text-align: center;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
        }}
        .stat-box h3 {{ margin: 0 0 5px 0; color: #6c757d; font-size: 13px; text-transform: uppercase; }}
        .stat-box .value {{ font-size: 26px; font-weight: bold; color: #003366; }}
        .stat-box.success .value {{ color: #28a745; }}
        .stat-box.danger .value {{ color: #dc3545; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            border: 1px solid #dee2e6;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #003366;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            {logo_html}
            <p style="color: #666; margin-top: 20px;">Date Generated On: {generated_time}</p>
        </div>

        <div class="stats-grid">
            <div class="stat-box">
                <h3>Total Inspected</h3>
                <div class="value">{total_cars}</div>
            </div>
            <div class="stat-box success">
                <h3>Passed</h3>
                <div class="value">{total_passed}</div>
            </div>
            <div class="stat-box danger">
                <h3>Failed</h3>
                <div class="value">{total_failed}</div>
            </div>
        </div>

        <h2>Detailed Log</h2>
        <table>
            <thead>
                <tr>
                    <th>VIN Number</th>
                    <th>Timestamp</th>
                    <th>Result</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    date_tag = target_date.replace("-", "") if target_date else datetime.now().strftime("%Y%m%d")
    filename  = f"session_summary_{date_tag}_{datetime.now().strftime('%H%M%S')}.html"
    out_path  = os.path.join(reports_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    return out_path

