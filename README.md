# AWS Cost Report (MVP)

This project is the **Minimum Viable Product (MVP)** for **Alpenglow Cloud Consulting**, showcasing an automated **AWS Cost Reporting tool**.

The tool pulls billing data via AWS Cost Explorer and generates **clean, client-ready reports (HTML + PDF)**.  
Reports summarize both **short-term (30-day)** and **long-term (12-month)** usage and costs, highlighting **potential savings opportunities**.

## Features
- **30-Day Daily View**: Spot cost spikes or anomalies  
- **12-Month Monthly View**: Track long-term trends  
- **High-Impact Cost Checks**: Early FinOps savings heuristics (EC2, EBS, S3, Data Transfer in development)  
- **Custom Templates**: HTML via Jinja2 + PDF rendering via WeasyPrint  
- **Quickstart** with Python virtual environment  

## Quickstart
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python run_report.py
Ensure you’ve set up AWS CLI credentials (the script uses Cost Explorer).

## Tech Stack
Python 3.10+

Boto3 (AWS SDK for Python)

Jinja2 (templating)

WeasyPrint (HTML → PDF rendering)

Matplotlib (visualizations)

## Roadmap
Expand High-Impact Checks (S3, EBS, Data Transfer)

Add anomaly detection & heuristics

Enhance narrative insights for executive summaries

## License
MIT License — see LICENSE for details.

This project is part of my portfolio demonstrating applied FinOps and AWS reporting skills.
