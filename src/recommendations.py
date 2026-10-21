import pandas as pd
import numpy as np


def generate_customer_recommendations(customer: dict, prob: float, risk_tier: str) -> dict:
    """
    Generates tailored, attribute-specific retention strategies based on customer profile,
    risk category, and feature drivers.
    
    Returns structured recommendations containing:
    - primary_strategy: Title of high-level strategy
    - business_priority: Strategic urgency level
    - tactical_actions: List of concrete action items with specific business rationales
    - estimated_objective: Targeted business outcome
    """
    tactical_actions = []
    
    # 1. Contract Analysis
    contract = customer.get("Contract", "Month-to-month")
    if contract == "Month-to-month":
        tactical_actions.append({
            "dimension": "Contractual Commitment",
            "action": "Annual Contract Migration Incentive",
            "detail": "Offer a 15% promotional credit or modern equipment upgrade in exchange for transitioning to a 12-month or 24-month contract agreement.",
            "rationale": "Month-to-month subscribers exhibit the highest churn propensity in historical data due to zero switching friction."
        })
    elif contract == "One year" and customer.get("tenure", 0) > 10:
        tactical_actions.append({
            "dimension": "Contract Renewal",
            "action": "Early Multi-Year Renewal Bonus",
            "detail": "Proactively reach out 60 days before contract expiry with a VIP 2-year renewal rate lock.",
            "rationale": "Mitigates contract expiration churn spike."
        })

    # 2. Tech Support & Cybersecurity Add-on Gaps
    has_internet = customer.get("InternetService") in ["Fiber optic", "DSL"]
    tech_support = customer.get("TechSupport") == "Yes"
    online_sec = customer.get("OnlineSecurity") == "Yes"
    
    if has_internet and (not tech_support or not online_sec):
        missing = []
        if not tech_support:
            missing.append("Dedicated Tech Support")
        if not online_sec:
            missing.append("Online Cyber Protection")
        tactical_actions.append({
            "dimension": "Service Ecosystem",
            "action": f"Complimentary 90-Day {' & '.join(missing)} Trial",
            "detail": f"Activate 3 months of free {' and '.join(missing)} followed by bundled loyalty pricing ($3/mo vs $8/mo standard).",
            "rationale": "Subscribers with security and support attachments show over 40% lower churn due to higher perceived platform utility."
        })

    # 3. Payment Method & Billing Friction
    payment_method = customer.get("PaymentMethod", "")
    if payment_method == "Electronic check":
        tactical_actions.append({
            "dimension": "Payment Channel Friction",
            "action": "Automated Billing Incentive Credit",
            "detail": "Provide a recurring $5 monthly bill discount for enrolling in automated Credit Card or Bank ACH transfer.",
            "rationale": "Electronic check payments have the highest billing failure and active churn rate among all payment channels in the dataset."
        })

    # 4. Monthly Spend Optimization
    monthly = float(customer.get("MonthlyCharges", 0.0))
    tenure = float(customer.get("tenure", 0.0))
    if monthly > 85.0 and tenure < 18:
        tactical_actions.append({
            "dimension": "Price Sensitivity & Value Perception",
            "action": "Plan Right-Sizing & Value Bundle",
            "detail": "Perform an account usage review to optimize bundle services or include a free premium streaming tier for 6 months.",
            "rationale": "High monthly charges combined with relatively low tenure create premium-price vulnerability against competitors."
        })

    # 5. Early Lifecycle Onboarding
    if tenure < 6:
        tactical_actions.append({
            "dimension": "Onboarding Lifecycle",
            "action": "Proactive Concierge Check-in",
            "detail": "Assign account to Customer Success for a 30-day proactive satisfaction call to resolve any connectivity or speed issues.",
            "rationale": "The first 6 months represent the most critical churn window; early dissatisfaction is the leading driver of premature cancellation."
        })

    # 6. Overall Strategy Synthesis based on Risk Tier
    if risk_tier == "High Risk":
        primary_strategy = "Priority Retention Escalation & Retention Offer"
        priority_level = "High Priority (Immediate Outreach)"
        retention_objective = "Prevent imminent cancellation by resolving service pain-points and providing contractual incentives."
    elif risk_tier == "Medium Risk":
        primary_strategy = "Proactive Service Engagement & Bundle Optimization"
        priority_level = "Medium Priority (Engagement within 14 Days)"
        retention_objective = "Address friction points and deepen product adoption before satisfaction degrades."
    else:
        primary_strategy = "Customer Loyalty Nurturing & Expansion"
        priority_level = "Standard Priority (Routine Lifecycle Marketing)"
        retention_objective = "Reinforce brand value, maintain high NPS, and identify cross-selling opportunities."
        if not tactical_actions:
            tactical_actions.append({
                "dimension": "Brand Advocacy",
                "action": "Loyalty Appreciation & Referral Reward",
                "detail": "Enroll customer in the VIP Tier and send a referral reward offering $50 account credits for referring business peers.",
                "rationale": "Low-risk, highly loyal customers generate the highest ROI through advocacy and organic referral acquisition."
            })

    return {
        "primary_strategy": primary_strategy,
        "business_priority": priority_level,
        "retention_objective": retention_objective,
        "tactical_actions": tactical_actions,
        "disclaimer": "Note: Recommendations represent strategic business heuristics based on historical correlation analysis. They do not constitute guaranteed churn prevention."
    }


def analyze_portfolio_risk_cohorts(df_scored: pd.DataFrame) -> dict:
    """
    Analyzes high-risk cohorts across the customer portfolio to identify
    aggregate business risks and revenue exposure.
    """
    high_risk_df = df_scored[df_scored["Risk_Tier"] == "High Risk"]
    
    total_customers = len(df_scored)
    high_risk_count = len(high_risk_df)
    
    # Revenue at risk
    total_monthly_revenue = df_scored["MonthlyCharges"].sum()
    monthly_rev_at_risk = high_risk_df["MonthlyCharges"].sum()
    pct_revenue_at_risk = (monthly_rev_at_risk / total_monthly_revenue * 100) if total_monthly_revenue > 0 else 0
    
    # Cohort 1: Month-to-month in high risk
    m2m_high = (high_risk_df["Contract"] == "Month-to-month").sum()
    
    # Cohort 2: Fiber optic without tech support
    fiber_no_support = ((high_risk_df["InternetService"] == "Fiber optic") & 
                        (high_risk_df["TechSupport"] == "No")).sum()
                        
    # Cohort 3: Electronic check users
    e_check_high = (high_risk_df["PaymentMethod"] == "Electronic check").sum()
    
    return {
        "total_customers": total_customers,
        "high_risk_count": high_risk_count,
        "high_risk_pct": round(high_risk_count / total_customers * 100, 2) if total_customers > 0 else 0,
        "monthly_revenue_at_risk": round(monthly_rev_at_risk, 2),
        "pct_revenue_at_risk": round(pct_revenue_at_risk, 2),
        "month_to_month_count": m2m_high,
        "fiber_no_support_count": fiber_no_support,
        "e_check_count": e_check_high
    }
