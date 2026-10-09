"""
Custom CSS and Styling Components for the Streamlit Web Application.
"""

CUSTOM_CSS = """
<style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header styling */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 0.8rem;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(14, 165, 233, 0.25);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38bdf8;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }

    /* Card Containers */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }

    /* Winner Banner */
    .winner-banner {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px -4px rgba(2, 132, 199, 0.35);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .winner-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.9;
        font-weight: 600;
    }
    .winner-class {
        font-size: 1.85rem;
        font-weight: 800;
        text-transform: capitalize;
        margin-top: 0.2rem;
    }
    .winner-confidence {
        font-size: 1.9rem;
        font-weight: 800;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.4rem 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Status indicator */
    .status-badge-online {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-badge-local {
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
"""
