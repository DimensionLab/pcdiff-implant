"""Report prompt templates v1.0 for AI-assisted report generation.

Version: v1.0
Created: 2026-02-03
Description: Initial prompt templates for cranioplasty case reports.
"""

PROMPT_VERSION = "v1.0"

# System prompt for the AI model
SYSTEM_PROMPT = """You are a medical report generation assistant for DimensionLab CrAInial, 
an AI-driven cranial implant design system. You generate professional, regulatory-compliant 
case reports for cranioplasty procedures.

Your reports should be:
1. Clear and professional in tone
2. Structured for clinical documentation
3. Include all relevant technical details
4. Suitable for medical records and regulatory compliance (MDR, FDA)
5. Formatted in clean HTML suitable for PDF conversion

Always maintain patient confidentiality and use only the data provided.
Do not make assumptions about medical details not explicitly provided."""

# Template for generating a comprehensive case report
CASE_REPORT_TEMPLATE = """Generate a comprehensive cranioplasty case report based on the following data:

## Patient Information
- Patient Code: {patient_code}
- Patient Name: {patient_name}
- Date of Birth: {date_of_birth}
- Sex: {sex}
- Medical Record Number: {medical_record_number}

## Case Details
- Project/Case Name: {project_name}
- Project Description: {project_description}
- Reconstruction Type: {reconstruction_type}
- Implant Material: {implant_material}
- Region Code: {region_code}
- Case Notes: {case_notes}

## CT Scan Information
{scan_info}

## AI-Generated Implant Information
{implant_info}

## Generation Job Details
{generation_details}

## Quality Metrics
{quality_metrics}

---

Please generate a formal clinical report in HTML format that includes:
1. **Executive Summary** - Brief overview of the case and implant design
2. **Patient Information** - Anonymized patient details
3. **Clinical Indication** - Reason for cranioplasty
4. **Imaging Data** - Summary of CT scan specifications used
5. **Implant Design Process** - AI generation method, parameters, and model used
6. **Quality Assurance** - Fit metrics, geometric accuracy, and validation results
7. **Manufacturing Specifications** - Material, recommended printing parameters
8. **Regulatory Compliance Notes** - Region-specific compliance information
9. **Conclusion and Recommendations**

Format the HTML with proper semantic tags (<h1>, <h2>, <p>, <table>, <ul>) for clean PDF conversion.
Use professional medical terminology appropriate for clinical documentation."""

# Region-specific compliance notes
REGION_COMPLIANCE = {
    "US": """
### FDA Compliance Notes (United States)
- This report is generated as design-assist documentation per FDA guidance on computational modeling and simulation
- Final clinical decisions must be made by qualified medical professionals
- Device classification: Patient-specific cranial implant (Class II)
- Reference: FDA-2023-D-0045 (Modeling & Simulation Credibility)
""",
    "EU": """
### MDR Compliance Notes (European Union)
- Documentation prepared in accordance with MDR 2017/745 requirements
- Technical documentation per Annex II and III requirements
- Risk management per ISO 14971:2019
- Quality management per ISO 13485:2016
- Clinical evaluation per MEDDEV 2.7/1 rev 4
""",
    "SK": """
### Regulatory Notes (Slovakia / EU)
- Documentation prepared in accordance with MDR 2017/745 requirements
- Applicable to EU medical device regulations
- ŠÚKL (State Institute for Drug Control) oversight applies
- Quality management per ISO 13485:2016
""",
    "DEFAULT": """
### Regulatory Notes
- This report is generated for informational and documentation purposes
- Final clinical decisions must be made by qualified medical professionals
- Local regulatory requirements may apply based on jurisdiction
"""
}


def get_compliance_section(region_code: str | None) -> str:
    """Get the appropriate compliance section for a region."""
    if region_code and region_code.upper() in REGION_COMPLIANCE:
        return REGION_COMPLIANCE[region_code.upper()]
    return REGION_COMPLIANCE["DEFAULT"]
